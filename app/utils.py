from io import StringIO
import sys
from json_repair import repair_json
import logging
import datetime
import os
import base64
from .models.oai_compatible_models import ChatCompletionStreamResponse
import json
import time
from typing import Any
from pydantic import BaseModel
import uuid

logger = logging.getLogger()

class CustomStream(StringIO):
    pass
    
class STDOutCapture(object):
    def __init__(self, buffer: StringIO):
        self.orig = sys.stdout
        self.buffer = buffer 
 
    def __enter__(self):
        sys.stdout = self.buffer

    def __exit__(self, *_):
        sys.stdout = self.orig
        
def get_system_prompt() -> str:
    import os

    if os.path.exists('system_prompt.txt'):
        with open('system_prompt.txt', 'r') as fp:
            return fp.read()

    return ''

def repair_json_no_except(json_str: str) -> str:
    try:
        return repair_json(json_str)
    except:
        logger.info(f"failed to repair json string {json_str}")
        return json_str



def preserve_upload_file(file_data_uri: str, file_name: str) -> str:
    os.makedirs(os.path.join(os.getcwd(), 'uploads'), exist_ok=True)

    file_data_base64 = file_data_uri.split(',')[-1]
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    try:
        file_data = base64.b64decode(file_data_base64)
        file_path = os.path.join(os.getcwd(), 'uploads', f"{timestamp}_{file_name}")
        
        with open(file_path, 'wb') as f:
            f.write(file_data)
        
        return file_path
    except Exception as e:
        logger.error(f"Failed to preserve upload file: {e}")
        return None

import re
def strip_toolcall_noti(content: str) -> str:
    cleaned = re.sub(r"<details\b[^>]*>.*?</details>", "", content, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()
 

def refine_chat_history(messages: list[dict[str, str]], system_prompt: str) -> list[dict[str, str]]:
    refined_messages = []
    has_system_prompt = False

    for message in messages:
        message: dict[str, str]

        if isinstance(message, dict) and message.get('role', 'undefined') == 'system':
            message['content'] += f'\n{system_prompt}'
            has_system_prompt = True
            refined_messages.append(message)
            continue
    
        if isinstance(message, dict) \
            and message.get('role', 'undefined') == 'user' \
            and isinstance(message.get('content'), list):

            content = message['content']
            text_input = ''
            attachments = []

            for item in content:
                if item.get('type', 'undefined') == 'text':
                    text_input += item.get('text') or ''

                elif item.get('type', 'undefined') == 'file':
                    file_item = item.get('file', {})
                    if 'file_data' in file_item and 'filename' in file_item:
                        file_path = preserve_upload_file(
                            file_item.get('file_data', ''),
                            file_item.get('filename', '')
                        )

                        if file_path:
                            attachments.append(file_path)

            if attachments:
                text_input += '\nAttachments:\n'

                for attachment in attachments:
                    text_input += f'- {attachment}\n'

            refined_messages.append({
                "role": "user",
                "content": text_input
            })

        else:
            _message = {
                "role": message.get('role', 'user'),
                "content": strip_toolcall_noti(message.get('content', ''))
            }

            refined_messages.append(_message)

    if not has_system_prompt and system_prompt != "":
        refined_messages.insert(0, {
            "role": "system",
            "content": system_prompt
        })

    if isinstance(refined_messages[-1], str):
        refined_messages[-1] = {
            "role": "user",
            "content": refined_messages[-1]
        }

    current_time_utc_str = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    refined_messages[-1]['content'] += f'\nCurrent time is {current_time_utc_str} UTC'

    return refined_messages

def to_chunk_data(chunk: ChatCompletionStreamResponse) -> bytes:
    return ("data: " + json.dumps(chunk.model_dump()) + "\n\n").encode()

def done_token() -> bytes:
    return "data: [DONE]\n\n".encode()

def refine_mcp_response(something: Any) -> str:
    if isinstance(something, dict):
        return {
            k: refine_mcp_response(v)
            for k, v in something.items()
        }

    elif isinstance(something, (list, tuple)):
        return [
            refine_mcp_response(v)
            for v in something
        ]

    elif isinstance(something, BaseModel):
        return something.model_dump()

    return something

def wrap_chunk(uuid: str, raw: str, role="assistant") -> ChatCompletionStreamResponse:
    return ChatCompletionStreamResponse(
        id=uuid,
        object='chat.completion.chunk',
        created=int(time.time()),
        model='unspecified',
        choices=[
            dict(
                index=0,
                delta=dict(
                    content=raw,
                    role=role
                )
            )
        ]
    )
    
def refine_assistant_message(
    assistant_message: dict[str, str]
) -> dict[str, str]:

    if 'content' in assistant_message:
        assistant_message['content'] = assistant_message['content'] or ""

    return assistant_message
    
def random_uuid() -> str:
    return str(uuid.uuid4())

def wrap_toolcall_request(uuid: str, fn_name: str, args: dict[str, Any]) -> ChatCompletionStreamResponse:
    args_str = json.dumps(args, indent=2)

    template = f'''
<action>Executing <b>{fn_name}</b></action>

<details>
<summary>
Arguments:
</summary>

```json
{args_str}
```

</details>
'''

    return ChatCompletionStreamResponse(
        id=uuid,
        model='unspecified',
        choices=[
            dict(
                index=0,
                delta=dict(
                    content=template,
                    role='tool'
                ),
            )
        ]
    )
    

def wrap_toolcall_response(
    uuid: str,
    fn_name: str,
    args: dict[str, Any],
    result: dict[str, Any]
) -> ChatCompletionStreamResponse:

    data = refine_mcp_response(result)

    try:
        data = json.dumps(data, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Failed to JOSN serialize tool call response: {e}")
        data = str(data)


    result = f'''
<details>
<summary>
Response:
</summary>

{data}

</details>
<br>

'''

    return ChatCompletionStreamResponse(
        id=uuid,
        model='unspecified',
        choices=[
            dict(
                index=0,
                delta=dict(
                    content=result,
                    role='tool'
                ),
            )
        ]
    )
    