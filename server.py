import logging

# print to file
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
)

from browser_use.browser.chrome import CHROME_ARGS

CHROME_ARGS += [
    "--start-maximized"
]

import fastapi
import uvicorn
import asyncio
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
import os
from contextlib import asynccontextmanager
import sys
from typing import Union
from app.models.oai_compatible_models import (
    ChatCompletionStreamResponse, 
    PromptErrorResponse
)
from app import prompt
from typing import AsyncGenerator
import time
import uuid
import openai
from browser_use import Browser, BrowserConfig
from browser_use.browser.context import BrowserContextConfig

logger = logging.getLogger(__name__)

if not load_dotenv():
    logger.warning("hehe, .env not found")

_GLOBALS = {}

@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    processes: list[asyncio.subprocess.Process] = []

    BROWSER_WINDOW_SIZE_WIDTH = int(os.getenv("BROWSER_WINDOW_SIZE_WIDTH", 1280)) 
    BROWSER_WINDOW_SIZE_HEIGHT = int(os.getenv("BROWSER_WINDOW_SIZE_HEIGHT", 768)) 
    SCREEN_COLOR_DEPTH_BITS = int(os.getenv("SCREEN_COLOR_DEPTH_BITS", 24))
    DISPLAY = os.getenv("DISPLAY", ":99")
    NO_VNC_PORT = os.getenv("NO_VNC_PORT", 6080)
    CHROME_DEBUG_PORT = os.getenv("CHROME_DEBUG_PORT", 9222)
    os.environ['CDP_URL'] = f"http://localhost:{CHROME_DEBUG_PORT}"

    commands = [
        'Xvfb {d} -screen 0 {w}x{h}x{b}'.format(
            w=BROWSER_WINDOW_SIZE_WIDTH,
            h=BROWSER_WINDOW_SIZE_HEIGHT,
            b=SCREEN_COLOR_DEPTH_BITS,
            d=DISPLAY
        ),
        'fluxbox',
        'x11vnc -display {d} -nopw -forever -shared'.format(
            d=DISPLAY
        ),
        '/opt/novnc/utils/novnc_proxy --vnc localhost:5900 --listen {no_vnc_port}'.format(
            no_vnc_port=NO_VNC_PORT
        )
    ]

    try:
        for command in commands:
            logger.info(f"Executing {command}")
            p = await asyncio.create_subprocess_shell(
                command,
                stdout=sys.stdout,
                stderr=sys.stderr,
                shell=True,
                executable="/bin/bash"
            )

            processes.append(p)

        browser = Browser(
            config=BrowserConfig(
                headless=False,
                disable_security=False,
                new_context_config=BrowserContextConfig(
                    allowed_domains=["*google.com*"],
                    cookies_file=None,
                    maximum_wait_page_load_time=5,
                    disable_security=False,
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
                    viewport=dict(
                        width=BROWSER_WINDOW_SIZE_WIDTH,
                        height=BROWSER_WINDOW_SIZE_HEIGHT
                    )
                )
            )
        )

        ctx = await browser.new_context() 
        
        _GLOBALS['browser'] = browser
        _GLOBALS['browser_context'] = ctx

        await _GLOBALS['browser_context'].__aenter__()

        page = await ctx.get_current_page()
        await page.goto(url='https://google.com', wait_until='commit')

        yield

    except Exception as err:
        logger.error(f"Exception raised {err}", stack_info=True)
        import traceback
        logger.error(traceback.format_exc())

    finally:

        if _GLOBALS.get('browser_context'):
            try:
                await _GLOBALS['browser_context'].__aexit__(None, None, None)
            except Exception as err:
                logger.error(f"Exception raised while closing browser context: {err}", stack_info=True)

        for process in processes:
            try:
                process.kill()
            except Exception as err: 
                logger.error(f"Failed to kill process {process}: {err}", stack_info=True)


async def stream_reader(s: AsyncGenerator[Union[str, bytes], None]):
    error_message = None
    response_uuid = str(uuid.uuid4())

    try:
        async for chunk in s:
            if chunk is None:
                continue

            if isinstance(chunk, str):
                chunk_model = ChatCompletionStreamResponse(
                    id=response_uuid,
                    object='chat.completion.chunk',
                    created=int(time.time()),
                    model='unspecified',
                    choices=[
                        dict(
                            index=0,
                            delta=dict(
                                content=chunk,
                                role='assistant'
                            )
                        )
                    ]
                )

                yield (f'data: {chunk_model.model_dump_json()}\n\n').encode('utf-8')
            else:
                yield chunk

    except openai.APIConnectionError as e:
        error_message=f"Failed to connect to language model: {e}"

    except openai.RateLimitError as e:
        error_message=f"Rate limit error: {e}"

    except openai.APIError as e:
        error_message=f"Language model returned an API Error: {e}"

    except Exception as err:
        error_message = "Unhandled error: " + str(err)

        import traceback
        logger.error(traceback.format_exc())

    finally:
        if error_message:
            yield (f'data: {PromptErrorResponse(message=error_message).model_dump_json()}\n\n').encode('utf-8')

        yield b'data: [DONE]\n\n'

def main():
    api_app = fastapi.FastAPI(
        lifespan=lifespan
    )
    
    @api_app.get("/processing-url")
    async def get_processing_url():
        http_display_url = os.getenv("HTTP_DISPLAY_URL", "http://localhost:6080/vnc.html?host=localhost&port=6080&autoconnect=true")

        if http_display_url:
            return JSONResponse(
                content={
                    "url": http_display_url,
                    "status": "ready"
                },
                status_code=200
            )

        return JSONResponse(
            content={
                "status": "not ready"
            },
            status_code=404
        )
        
    @api_app.post("/prompt", response_model=None)
    async def post_prompt(body: dict) -> Union[StreamingResponse, PlainTextResponse, JSONResponse]:
        if body.get('ping'):
            return PlainTextResponse("online")

        messages: list[dict[str, str]] = body.pop('messages', [])

        if len(messages) == 0:
            return JSONResponse(
                content=PromptErrorResponse(
                    message="Received empty messages"
                ).model_dump(),
                status_code=400
            )

        if isinstance(messages[-1], str):
            messages[-1] = {
                "role": "user",
                "content": messages[-1]
            }

        messages[-1].setdefault('role', 'user')

        try:
            stream = prompt(
                messages, 
                browser_context=_GLOBALS["browser_context"], 
                **body
            )

            return StreamingResponse(
                stream_reader(stream),
                media_type="text/event-stream"
            )
        except Exception as err:
            error_message = "Unexpected Error: " + str(err)

            import traceback
            logger.error(traceback.format_exc())

            return JSONResponse(
                content=PromptErrorResponse(
                    message=error_message
                ).model_dump(),
                status_code=500
            )

    api_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # pre-setup
    event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(event_loop)

    config = uvicorn.Config(
        api_app,
        loop=event_loop,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "80")),
        log_level="info",
        timeout_keep_alive=300,
    )

    server = uvicorn.Server(config)
    event_loop.run_until_complete(server.serve())

if __name__ == '__main__':
    main()