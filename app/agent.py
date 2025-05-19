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
import traceback

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

        extend_system_message=get_system_prompt(),
        extend_planner_system_message=planner_extend_system_message,

        is_planner_reasoning=False,
        use_vision=False,
        use_vision_for_planner=False,

        tool_calling_method='function_calling',

        # @TODO: fix this bug
        enable_memory=False, # bug here
        memory_config=MemoryConfig(
            agent_id="gmail-use-agent",
            memory_interval=5,
            embedder_provider='openai',
            embedder_model=os.getenv("EMBEDDING_MODEL_ID", 'text-embedding-3-small'),
            embedder_dims=int(os.getenv("EMBEDDING_DIMS", 256))
        )        
    )

async def browse(task_query: str, ctx: BrowserContext, **_) -> AsyncGenerator[str, None]:
    current_agent = await get_agent(task=task_query, ctx=ctx) 

    res = await current_agent.run(
        max_steps=40,
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
    date: Optional[str]=None, 
    date_within: Optional[str]=None,
    sender: Optional[str]=None, 
    include_words: Optional[str]="", 
    has_attachment: Optional[bool]=False,
    additional_message: Optional[str]=None
) -> AsyncGenerator[str, None]:
    gen_id = IncrementID()

    task = ''
    
    page = await ctx.get_current_page()
    url = page.url

    if not fnmatch(url, 'https://mail.google.com/*'):
        task += f'{gen_id()}. navigate to mail.google.com or use open_mail_box to open the mail box\n'

    if date or date_within or sender or include_words or has_attachment:
        task += f"{gen_id()}. Open the advance search panel in Gmail\n"

        if date:
            date_within = date_within or '1 day'

            task += f"{gen_id()}. Set the date to {date} in format (yyyy/mm/dd)\n"
            task += f"{gen_id()}. In the 'Date within' dropdown list, select {date_within} or a similar option that has close meaning to {date_within}\n"
  
        if sender:
            task += f"{gen_id()}. fill 'From' field with {sender}\n"

        if include_words:
            task += f"{gen_id()}. fill the 'Has the words' field with {include_words}\n"

        if has_attachment:
            task += f"{gen_id()}. check the 'Has attachment' checkbox\n"
            
        task += f"{gen_id()}. Click the search button to perform the search and wait until the filtering is done\n"

    if not additional_message:
        task += f"{gen_id()}. Extract the results in the center of the current page, summarize and response back to the user, including senders, subject, time received. No need to get details of each e-mail\n"

    else:
        task += f"{gen_id()}. Extract the results in the center of the current page, summarize and response back to the user, including senders, subject, time received. Do the additional request by the user: {additional_message!r}\n"

    async for msg in browse(task, ctx):
        yield msg

async def send_email(
    ctx: BrowserContext, 
    recipient: str, 
    subject: str, 
    body: str,
    sign: Optional[str] = None
) -> AsyncGenerator[str, None]:
    gen_id = IncrementID()
    task = ''

    page = await ctx.get_current_page()
    url = page.url

    if not fnmatch(url, 'https://mail.google.com/*'):
        task += f'{gen_id()}. navigate to mail.google.com\n'

    task += f"{gen_id()}. Click the 'Compose' button to open a new email window\n"
    task += f"{gen_id()}. Fill the 'To' field with {recipient!r}\n"
    task += f"{gen_id()}. Fill the 'Subject' field with {subject!r}\n"
    task += f"{gen_id()}. Fill the email body with {body!r}\n"
    
    if sign:
        task += f"{gen_id()}. Use the following signature: {sign!r}\n"

    else:
        task += f"{gen_id()}. Re-check all information above, then halt and wait for user manual execution\n"

    async for msg in browse(task, ctx):
        yield msg
        
async def sign_out(
    ctx: BrowserContext
) -> AsyncGenerator[str, None]:
    gen_id = IncrementID()
    task = ''

    task += f"{gen_id()}. Use sign_out tool to sign out the current account (MUST USE)\n"

    async for msg in browse(task, ctx):
        yield msg
        
async def call_llm(messages: list[dict[str, str]], tools: list[dict[str, Any]], max_tokens: int):
    llm = openai.AsyncClient(
        base_url=os.getenv("LLM_BASE_URL", "http://localhost:65534/v1"),
        api_key=os.getenv("LLM_API_KEY", "no-need")
    )

    model_id = os.getenv("LLM_MODEL_ID", 'local-llm')

    completion = await llm.chat.completions.create(
        model=model_id,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_tokens=max_tokens
    )

    return completion

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
                        "date": {
                            "type": ["string", "null"],
                            "description": "The date to filter emails by, in format (yyyy/mm/dd)."
                        },
                        "date_within": {
                            "type": ["string", "null"],
                            "description": "The time period to filter emails by, e.g., '1 day', '1 week'."
                        },
                        "sender": {
                            "type": ["string", "null"],
                            "description": "Filter emails by sender's email address."
                        },
                        "include_words": {
                            "type": ["string", "null"],
                            "description": "Include words to filter emails."
                        },
                        "has_attachment": {
                            "type": ["boolean", "null"],
                            "description": "Whether to filter emails that have attachments."
                        },
                        "additional_message": {
                            "type": ["string", "null"],
                            "description": "Optional additional message to include in the response."
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
                        "sign": {
                            "type": ["string", "null"],
                            "description": "Optional signature to append to the email body."
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
                "name": "sign_out",
                "description": "Sign out to the current account.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False
                },
                "strict": False
            }
        }
    ]

    if not await check_authorization(browser_context):
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
