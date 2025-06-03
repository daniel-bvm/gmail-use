from typing import AsyncGenerator
import os
import json
from browser_use.browser.context import BrowserContext
from .models import (
    oai_compatible_models,
)
from .utils import (
    get_system_prompt,
    refine_chat_history,
    to_chunk_data,
    refine_assistant_message,
    random_uuid,
    wrap_toolcall_request,
    wrap_toolcall_response, 
    refine_mcp_response
)
import logging
import openai
from .signals import UnauthorizedAccess, RequireUserConfirmation
import openai
import httpx
import traceback
from .toolcalls import (
    get_context_aware_available_toolcalls,
    execute_toolcall,
    get_current_user_identity
)

logger = logging.getLogger()

async def prompt(messages: list[dict[str, str]], browser_context: BrowserContext, **_) -> AsyncGenerator[str, None]:
    llm = openai.AsyncClient(
        base_url=os.getenv("LLM_BASE_URL", "http://localhost:65534/v1"),
        api_key=os.getenv("LLM_API_KEY", "no-need")
    )

    response_uuid = random_uuid()

    error_details = ''
    error_message = ''
    calls = 0
    
    system_prompt = get_system_prompt()
    system_prompt.strip(" \n")

    try:
        user_identity_req = await get_current_user_identity(browser_context)
        if user_identity_req.success:
            system_prompt += "\n- User identity to use as mail signature: " + user_identity_req.result
    except Exception as e:
        pass

    messages = refine_chat_history(messages, system_prompt)
    toolcalls = await get_context_aware_available_toolcalls(browser_context)

    try:
        completion = await llm.chat.completions.create(
            model=os.getenv("LLM_MODEL_ID", 'local-llm'),
            messages=messages,
            tools=toolcalls,
            tool_choice="auto"
        )

        if completion.choices[0].message.content:
            yield completion.choices[0].message.content

        messages.append(refine_assistant_message(completion.choices[0].message.model_dump()))

        while completion.choices[0].message.tool_calls is not None and len(completion.choices[0].message.tool_calls) > 0:
            calls += len(completion.choices[0].message.tool_calls)
            has_user_interaction_requested = False

            for call in completion.choices[0].message.tool_calls:
                _id, _name = call.id, call.function.name    
                _args = json.loads(call.function.arguments)
                result  = ''

                yield to_chunk_data(wrap_toolcall_request(
                    uuid=response_uuid,
                        fn_name=_name,
                        args=_args
                    )
                )

                try:
                    res = await execute_toolcall(
                        ctx=browser_context,
                        tool_name=_name,
                        args=_args
                    )
                    
                    logger.info(f"Tool call result: {res}")

                    if res.success:
                        yield to_chunk_data(
                            wrap_toolcall_response(
                                uuid=response_uuid,
                                fn_name=_name,
                                args=_args,
                                result=res.result
                            )
                        )

                        result = json.dumps(refine_mcp_response(res.result))

                    else:
                        result = f"Tool call failed: {res.error}"

                except UnauthorizedAccess as e:
                    result = f"Waiting for the user to sign in manually." 
                    has_user_interaction_requested = True

                except RequireUserConfirmation as e:
                    result = str(e)
                    has_user_interaction_requested = True

                except Exception as e:
                    result = f"Exception raised while  executing tool call: {e}"
                    logger.error(f"Exception raised while executing tool call: {e}", stack_info=True)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": _id,
                        "content": result
                    }
                )

                if has_user_interaction_requested:
                    break

            need_toolcalls = calls < 10 and not has_user_interaction_requested
            
            with open('messages.json', 'w') as f:
                json.dump(messages, f, indent=2)

            completion = await llm.chat.completions.create(
                messages=messages,
                model=os.getenv("LLM_MODEL_ID", 'local-llm'),
                tools=toolcalls if need_toolcalls else openai._types.NOT_GIVEN,  # type: ignore
                tool_choice="auto" if need_toolcalls else openai._types.NOT_GIVEN,  # type: ignore
            )

            logger.info(f"Assistant: {completion.choices[0].message.content!r}")

            if completion.choices[0].message.content:
                yield completion.choices[0].message.content

            messages.append(refine_assistant_message(completion.choices[0].message.model_dump()))

    except openai.APIConnectionError as e:
        error_message=f"Failed to connect to language model: {e}"
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

            yield to_chunk_data(
                oai_compatible_models.PromptErrorResponse(
                    message=error_message, 
                    details=error_details
                )
            )