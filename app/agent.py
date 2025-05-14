from typing import AsyncGenerator
from langchain_openai import ChatOpenAI
import os
from .callbacks import (
    on_step_end, 
    on_step_start
)

from browser_use import Agent
from browser_use.browser.context import BrowserContext
from .controllers import get_controler
from .models import browser_use_custom_models
from .utils import get_system_prompt, repair_json_no_except, refine_chat_history
import logging
import openai
import json
from .signals import UnauthorizedAccess


logger = logging.getLogger()


async def get_agent(task: str, ctx: BrowserContext) -> Agent:
    return Agent(
        task=task,
        llm=ChatOpenAI(
            model=os.getenv("LLM_MODEL_ID", 'local-llm'),
            openai_api_base=os.getenv("LLM_BASE_URL", 'http://localhost:65534/v1'),
            openai_api_key=os.getenv("LLM_API_KEY", 'no-need'),
        ),
        page_extraction_llm=None,
        planner_llm=None,
        browser_context=ctx,
        controller=get_controler(),
        extend_system_message=get_system_prompt(),
        is_planner_reasoning=False,
        use_vision=True,
        use_vision_for_planner=True,
        enable_memory=False
    )

async def browse(task_query: str, ctx: BrowserContext, **_) -> AsyncGenerator[str, None]:
    current_agent = await get_agent(task=task_query, ctx=ctx) 

    try:
        res = await current_agent.run(
            max_steps=40,
            on_step_start=on_step_start, 
            on_step_end=on_step_end
        )

    except UnauthorizedAccess as err:
        logger.info(f"Unauthorized access: {err}")
        yield "Let's sign in to your account first, then notify me once you finish.\n\n"
        return 

    final_result = res.final_result()

    if final_result is not None:
        try:
            parsed: browser_use_custom_models.FinalAgentResult \
                = browser_use_custom_models.FinalAgentResult.model_validate_json(
                    repair_json_no_except(final_result)
                )

            if parsed.status == "pending":
                logger.info(f"Completed task in status {parsed.status}")

            yield parsed.message + '\n\n'
        except Exception as err:
            logger.info(f"Exception raised while parsing final answer: {err}")
            yield f"Exception raised: {err}\n\n"
    else:
        logger.info("Final result is None")
        yield "I've done my task!"


async def prompt(messages: list[dict[str, str]], browser_context: BrowserContext, **_) -> AsyncGenerator[str, None]:
    functions = [
        {
            "type": "function",
            "function": {
                "name": "xbrowser",
                "description": "Ask the XBrowser to complete the browsing task, as you cannot!",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "Task description to do in browser. It should be rich information. For instance, the user want to go shopping online, the task definition should be: what to buy, where and, or how to retrieve the needed information, etc."
                        }    
                    },
                    "required": ["task"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }
    ]

    llm = openai.AsyncClient(
        base_url=os.getenv("LLM_BASE_URL", "http://localhost:65534/v1"),
        api_key=os.getenv("LLM_API_KEY", "no-need")
    )

    completion = await llm.chat.completions.create(
        model=os.getenv("LLM_MODEL_ID", 'local-llm'),
        messages=await refine_chat_history(messages, get_system_prompt()),
        tools=functions,
        tool_choice="auto",
        max_tokens=256
    )

    if completion.choices[0].message.content:
        yield completion.choices[0].message.content

    if (
        completion.choices[0].message.tool_calls is not None \
        and len(completion.choices[0].message.tool_calls) > 0
    ):
        compose_task = ""

        for call in completion.choices[0].message.tool_calls:
            _args: dict = json.loads(call.function.arguments)

            if desc := _args.get("task"):
                compose_task += desc + '\n'

        if compose_task:
            async for msg in browse(task_query=compose_task, ctx=browser_context):
                yield msg
            
            