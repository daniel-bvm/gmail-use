from typing import AsyncGenerator
from langchain_openai import ChatOpenAI
import os
from .callbacks import (
    on_step_end, 
    on_step_start
)
import json
from browser_use import Agent
from browser_use.agent.memory import MemoryConfig
from browser_use.browser.context import BrowserContext
from .controllers import get_controler, check_authorization, Controller
from .models import browser_use_custom_models, oai_compatible_models
from .utils import (
    get_system_prompt,
    repair_json_no_except,
    refine_chat_history,
    to_chunk_data,
    refine_assistant_message,
    wrap_chunk,
    random_uuid
)
import logging
import openai
from .signals import UnauthorizedAccess
import openai
from typing import Optional, Any
import httpx
from datetime import datetime, timedelta

logger = logging.getLogger()

async def get_agent(task: str, ctx: BrowserContext, controller: Controller=None) -> Agent:
    controller = controller or get_controler()

    ellm = ChatOpenAI(
        model=os.getenv("LLM_MODEL_ID", 'local-llm'),
        openai_api_base=os.getenv("LLM_BASE_URL", 'http://localhost:65534/v1'),
        openai_api_key=os.getenv("LLM_API_KEY", 'no-need'),
        temperature=0.1
    )
        
    planner_extend_system_message = get_system_prompt() + 'The actor has accessibility to the following tools:\n'

    for i, (k, v) in enumerate(controller.registry.registry.actions.items()):
        planner_extend_system_message += f"{i + 1}. {k}: {v.description}\n"

    planner_extend_system_message += '\n'
    planner_extend_system_message += 'Read the descriptions carefully first then guide the actor to choose the right for execution.'

    return Agent(
        task=task,

        llm=ellm,
        planner_llm=ellm,
        page_extraction_llm=ellm,

        browser_context=ctx,
        controller=controller,
        planner_interval=1,

        extend_system_message=get_system_prompt(),
        extend_planner_system_message=planner_extend_system_message,

        is_planner_reasoning=False,
        use_vision=True,
        use_vision_for_planner=True,

        tool_calling_method='function_calling',
        max_actions_per_step=2,

        # @TODO: fix this bug
        enable_memory=False, # bug here
        # memory_config=MemoryConfig(
        #     agent_id="gmail-use-agent",
        #     memory_interval=5,
        #     embedder_provider='openai',
        #     embedder_model=os.getenv("EMBEDDING_MODEL_ID", 'text-embedding-3-small'),
        #     embedder_dims=int(os.getenv("EMBEDDING_DIMS", 256))
        # )        
    )

async def browse(task_query: str, ctx: BrowserContext, max_steps:int = 40, **_) -> AsyncGenerator[str, None]:
    current_agent = await get_agent(task=task_query, ctx=ctx) 

    res = await current_agent.run(
        max_steps=max_steps,
        on_step_start=on_step_start, 
        on_step_end=on_step_end
    )

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

from fnmatch import fnmatch

class IncrementID(object):
    def __init__(self, start: int = 1):
        self._it = start - 1
    
    def __call__(self, *args, **kwds):
        self._it += 1
        return self._it

async def get_emails(
    ctx: BrowserContext,
    from_date: Optional[str]=None, 
    to_date: Optional[str]=None,
    sender: Optional[str]=None, 
    recipient: Optional[str]=None,
    include_words: Optional[str]="", 
    has_attachment: Optional[bool]=False,
) -> AsyncGenerator[str, None]:
    gen_id = IncrementID()

    task = 'Strictly follow the steps below to complete the task:\n\n'
    
    page = await ctx.get_current_page()
    url = page.url

    if not fnmatch(url, 'https://mail.google.com/*'):
        task += f'{gen_id()}. navigate to mail.google.com or use open_mail_box to open the mail box\n'

    if from_date or to_date or sender or include_words or has_attachment or recipient:
        query_str = f'{include_words}'

        if from_date:
            query_str += f' after:{from_date}'

        if to_date:
            if to_date == from_date:
                to_date_obj = datetime.strptime(to_date, '%Y/%m/%d')
                correct_to_date = to_date_obj + timedelta(days=1)
                to_date = correct_to_date.strftime('%Y/%m/%d')

            query_str += f' before:{to_date}'

        if sender:
            query_str += f' from:{sender}'

        if recipient:
            query_str += f' to:{recipient}'

        if has_attachment:
            query_str += ' has:attachment'

        task += f"{gen_id()}. Perform searching by executing search_email with the the query: {query_str!r}\n"

    task += f"{gen_id()}. Extract emails from the current page including name of the senders, subject and time the email was sent.\n"

    async for msg in browse(task, ctx):
        yield msg

async def send_email(
    ctx: BrowserContext, 
    recipient: str, 
    subject: str, 
    body: str,
    required_user_confirmation: Optional[bool]=True,
) -> AsyncGenerator[str, None]:
    gen_id = IncrementID()
    
    dest = 'prepare an email' if required_user_confirmation else 'send an email' 
    task = f'Strictly follow the steps below to {dest}:\n\n'

    page = await ctx.get_current_page()
    url = page.url

    if not fnmatch(url, 'https://mail.google.com/*'):
        task += f'{gen_id()}. Navigate to mail.google.com\n'

    form_data = {
        "recipient": recipient,
        "subject": subject,
        "body": body
    }

    task += f"{gen_id()}. Click the Compose button to open the new-email form\n"
    task += f"{gen_id()}. use fill_email_form with the following arguments:\n" 
    task += f"```json\n{json.dumps(form_data, indent=2)}\n```\n"

    if not required_user_confirmation:
        task += f"{gen_id()}. Click Send\n"

    else:
        task += f"{gen_id()}. Do not click the send button. Let the user do it manually\n"

    task += f"\nImportant: Do not try to fill the form manually yourself"

    async for msg in browse(task, ctx):
        yield msg

async def sign_out(
    ctx: BrowserContext
) -> AsyncGenerator[str, None]:
    gen_id = IncrementID()
    task = ''

    task += f"{gen_id()}. Use sign_out tool to sign out the current account (MUST USE)\n"

    try:
        async for msg in browse(task, ctx):
            yield msg

    except UnauthorizedAccess as e:
        yield "Account is successfully signed out!"

async def execute_openai_compatible_toolcall(
    ctx: BrowserContext,
    name: str,
    args: dict[str, str]
) -> AsyncGenerator[str, None]:
    logger.info(f"Executing tool call: {name} with args: {args}")
    
    if name == "get_emails":
        async for msg in get_emails(ctx, **args):
            yield msg
            
        return

    if name == "send_email":
        async for msg in send_email(ctx, **args):
            yield msg
            
        return
    
    if name == "sign_out":
        async for msg in sign_out(ctx):
            yield msg
            
        return
    
    if name == "xbrowse":
        task = args.get("task", "")

        if not task:
            yield "No task provided for xbrowse tool call."
            return

        async for msg in browse(task, ctx):
            yield msg
            
        return

    yield f"Unknown tool call: {name}; Available tools are: get_emails, send_email"

async def prompt(messages: list[dict[str, str]], browser_context: BrowserContext, **_) -> AsyncGenerator[str, None]:
    functions = [
        {
            "type": "function",
            "function": {
                "name": "get_emails",
                "description": "Get emails from Gmail with the specified criteria.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "from_date": {
                            "type": ["string", "null"],
                            "description": "The starting date to filter emails by, in format (yyyy/mm/dd)."
                        },
                        "to_date": {
                            "type": ["string", "null"],
                            "description": "The ending date to filter emails by, in format (yyyy/mm/dd). If not provided, it defaults to the current date."
                        },
                        "sender": {
                            "type": ["string", "null"],
                            "description": "Filter emails by sender's email address."
                        },
                        "recipient": {
                            "type": ["string", "null"],
                            "description": "Filter emails by recipient's email address."
                        },
                        "include_words": {
                            "type": ["string", "null"],
                            "description": "Include words to filter emails."
                        },
                        "has_attachment": {
                            "type": ["boolean", "null"],
                            "description": "Whether to filter emails that have attachments."
                        }
                    },
                    "required": [],
                    "additionalProperties": False
                },
                "strict": False
            }
        },
        {
            "type": "function",
            "function": {
                "name": "send_email",
                "description": "Send an email using Gmail.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "recipient": {
                            "type": "string",
                            "description": "The email address of the recipient."
                        },
                        "subject": {
                            "type": "string",
                            "description": "The subject of the email."
                        },
                        "body": {
                            "type": "string",
                            "description": "The body content of the email."
                        },
                        "required_user_confirmation": {
                            "type": ["string", "null"],
                            "description": "let the user confirm and send it manually. Default to 'true'"
                        }
                    },
                    "required": ["recipient", "subject", "body"],
                    "additionalProperties": False
                },
                "strict": False
            }
        },
        {
            "type": "function",
            "function": {
                "name": "xbrowse",
                "description": "Ask xbrowser to do a task in the browser like replying an email or anything that there is no tools to execute, etc.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "Task description to execute in browser. It should be as much detail as possible, step-by-step to achieve the task."
                        }    
                    },
                    "required": ["task"],
                    "additionalProperties": False
                },
                "strict": True
            }
        },
        {
            "type": "function",
            "function": {
                "name": "sign_out",
                "description": "Sign out.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False
                },
                "strict": False
            }
        },
        
    ]

    if await check_authorization(browser_context):
        functions = functions[:-1] # remove sign_out function
    
    llm = openai.AsyncClient(
        base_url=os.getenv("LLM_BASE_URL", "http://localhost:65534/v1"),
        api_key=os.getenv("LLM_API_KEY", "no-need")
    )

    response_uuid = random_uuid()

    error_details = ''
    error_message = ''
    calls = 0
    
    messages = await refine_chat_history(messages, get_system_prompt())

    try:
        completion = await llm.chat.completions.create(
            model=os.getenv("LLM_MODEL_ID", 'local-llm'),
            messages=messages,
            tools=functions,
            tool_choice="auto",
            max_tokens=512
        )

        if completion.choices[0].message.content:
            yield completion.choices[0].message.content

        messages.append(await refine_assistant_message(completion.choices[0].message.model_dump()))

        while completion.choices[0].message.tool_calls is not None and len(completion.choices[0].message.tool_calls) > 0:
            calls += len(completion.choices[0].message.tool_calls)

            for call in completion.choices[0].message.tool_calls:
                _id, _name = call.id, call.function.name
                _args = json.loads(call.function.arguments)

                yield await to_chunk_data(
                    await wrap_chunk(
                        response_uuid,
                        f"**Calling**: {_name}...\n",
                        role="tool",
                    )
                )
                
                result, unauthorized = '', False

                try:

                    async for msg in execute_openai_compatible_toolcall(
                        ctx=browser_context, 
                        name=_name,
                        args=_args
                    ):
                        yield await to_chunk_data(
                            await wrap_chunk(
                                response_uuid,
                                msg,
                                role="tool"
                            )
                        )

                        result += msg + '\n'

                except UnauthorizedAccess as e:
                    logger.warning(f"{e}")

                    yield await to_chunk_data(
                        await wrap_chunk(
                            response_uuid,
                            f"Required log-in first. Pausing...\n",
                            role="tool"
                        )
                    )
   
                    result = f"Unauthorized access: {str(e)}\nNow, halt the current task and wait for the user to sign in manually. After then, Re-execute {_name} with these arguments: {_args}" 
                    unauthorized = True

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": _id,
                        "content": result
                    }
                )

                if unauthorized:
                    break

            need_toolcalls = calls < 10 and not unauthorized

            completion = await llm.chat.completions.create(
                messages=messages,
                model=os.getenv("LLM_MODEL_ID", 'local-llm'),
                tools=functions if need_toolcalls else openai._types.NOT_GIVEN,  # type: ignore
                tool_choice="auto" if need_toolcalls else openai._types.NOT_GIVEN,  # type: ignore
                max_tokens=512
            )

            logger.info(f"Assistant: {completion.choices[0].message.content!r}")

            if completion.choices[0].message.content:
                yield completion.choices[0].message.content

            messages.append(await refine_assistant_message(completion.choices[0].message.model_dump()))

    except openai.APIConnectionError as e:
        error_message=f"Failed to connect to language model: {e}"
        import traceback
        error_details = traceback.format_exc(limit=-6)

    except openai.RateLimitError as e:
        error_message=f"Rate limit error: {e}"

    except openai.APIError as e:
        error_message=f"Language model returned an API Error: {e}"

    except httpx.HTTPStatusError as e:
        error_message=f"HTTP status error: {e}"
        
    except Exception as e:
        error_message=f"Unhandled error: {e}"
        error_details = traceback.format_exc(limit=-6)
        
    finally:
        if error_message:

            logger.error(f"Error occurred: {error_message}")
            logger.error(f"Error details: {error_details}")

            yield await to_chunk_data(
                oai_compatible_models.PromptErrorResponse(
                    message=error_message, 
                    details=error_details
                )
            )
