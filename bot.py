import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

# =========== consts ============
from typing import Optional
# ========================================
#             Model Initialization
# ========================================

print('Initializing Bot')
cfg = Config(parse_args())

BOT_TOKEN = cfg.config.bot.bot_token
CLIENT_ID = cfg.config.bot.client_id
ALLOWED_SERVER_IDS = [int(s) for s in cfg.config.bot.allowed_server_ids.split(",")]
# Send Messages, Create Public Threads, Send Messages in Threads, Manage Messages, Manage Threads, Read Message History, Use Slash Command
BOT_INVITE_URL = f"https://discord.com/api/oauth2/authorize?client_id={CLIENT_ID}&permissions=328565073920&scope=bot"
print(BOT_INVITE_URL)

model_config = cfg.model_cfg
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:0')

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor)
print('Initialization Finished')


def stateful_single_answer(image, message):
    chat_state = CONV_VISION.copy()
    img_list = []
    llm_message = chat.upload_img(image, chat_state, img_list) # shou
    assert llm_message == "Received."
    chat.ask(message, chat_state)
    #chatbot = chatbot + [[user_message, None]]
    num_beams,temperature = 5,1.0
    llm_message = chat.answer(conv=chat_state, img_list=img_list, max_new_tokens=1000, num_beams=num_beams, temperature=temperature)[0]
    #chatbot[-1][1] = llm_message
    return llm_message

#OWNER_ID = environ['OWNER_ID']
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s", level=logging.INFO
)

import discord
intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)
tree = discord.app_commands.CommandTree(client)

def should_block(guild: Optional[discord.Guild]) -> bool:
    if guild is None: # dm's not supported
        logger.info(f"DM not supported")
        return True

    if guild.id and guild.id not in ALLOWED_SERVER_IDS: # not allowed in this server
        logger.info(f"Guild {guild} not allowed")
        return True
    return False

import asyncio
import concurrent.futures
from PIL import Image
from io import BytesIO
queue = asyncio.Queue()

@client.event
async def on_ready():
    logger.info(f"We have logged in as {client.user}. Invite URL: {BOT_INVITE_URL}")
    #asyncio.get_running_loop().create_task(background_task())
    await tree.sync()

def create_embed(prompt, user):
    embed = discord.Embed(title='MiniGPT-4', description=prompt, color=0x00ff00)
    embed.set_author(name=user.name, url="https://todo.com", icon_url=user.display_avatar.url)
    embed.set_thumbnail(url="attachment://thumb.webp")
    return embed

async def send_interaction(interaction, embed, pil_im, result):
    thumb = pil_im.copy()
    thumb.thumbnail((256, 256))
    with BytesIO() as bio:
        thumb.save(bio, format="webp")
        bio.seek(0)
        file = discord.File(bio, filename='thumb.webp')
        #embedVar.add_field(name="Query", value=req.query.text, inline=False)
        embed.add_field(name="Response", value=result[:1000], inline=False)
        for i in range(len(result)//1000):
            I = i+1
            embed.add_field(name="\u200b"*I, value=result[1000*I:1000*(I+1)], inline=False)

        await interaction.followup.send(embed=embed, file=file)

import functools
async def stateful_single_answer_async(image, message):
    result = await asyncio.get_running_loop().run_in_executor(
        None, functools.partial(stateful_single_answer, image, message)
    )
    return result

# /mgpt message:
@tree.command(name="mgpt4", description="Send an instruction to MiniGPT-4")
@discord.app_commands.checks.has_permissions(send_messages=True)
@discord.app_commands.checks.has_permissions(view_channel=True)
@discord.app_commands.checks.bot_has_permissions(send_messages=True)
@discord.app_commands.checks.bot_has_permissions(view_channel=True)
async def chat_command(interaction: discord.Interaction, image: discord.Attachment, prompt: str="Write a poem about this image."):
    try:
        # only use text channels
        if not isinstance(interaction.channel, discord.TextChannel):
            return
        # block servers not in allow list
        if should_block(guild=interaction.guild):
            return

        logger.info(f"Chat command by {interaction.user} {prompt[:20]}")

        image_bytes = await image.read()
        await interaction.response.defer()
        try: 
            pil_im = Image.open(BytesIO(image_bytes)).convert("RGB")
            embed = create_embed(prompt, interaction.user)
            llm_message = await stateful_single_answer_async(pil_im, prompt)
            await send_interaction(interaction, embed, pil_im, llm_message)
            #
        except Exception as e:
            await interaction.followup.send("got an error, sorry")
            raise e
    except Exception as e:
        logger.error(f"Error in chat_command: {e}")
        return


client.run(BOT_TOKEN)


