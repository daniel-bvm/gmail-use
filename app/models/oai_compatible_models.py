# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/lm-sys/FastChat/blob/168ccc29d3f7edc50823016105c024fe2282732a/fastchat/protocol/openai_api_protocol.py
import time
from typing import Annotated, Any, ClassVar, Literal, Optional, Union, Iterable
from pydantic import (BaseModel, ConfigDict, Field, TypeAdapter,
                      ValidationInfo, field_validator, model_validator)
from openai.types.chat import ChatCompletionContentPartParam
from openai.types.chat import (
    ChatCompletionMessageParam as OpenAIChatCompletionMessageParam)
from openai.types.chat import (ChatCompletionMessageToolCallParam,)
from typing_extensions import TypedDict, Required
import logging 
from dataclasses import dataclass
logger = logging.getLogger(__name__)

import uuid

def random_uuid() -> str:
    """Generate a random UUID string."""
    return str(uuid.uuid4())

class OpenAIBaseModel(BaseModel):
    # OpenAI API does allow extra fields
    model_config = ConfigDict(extra="allow")

    # Cache class field names
    field_names: ClassVar[Optional[set[str]]] = None

    @model_validator(mode="wrap")
    @classmethod
    def __log_extra_fields__(cls, data, handler):
        result = handler(data)
        if not isinstance(data, dict):
            return result
        field_names = cls.field_names
        if field_names is None:
            # Get all class field names and their potential aliases
            field_names = set()
            for field_name, field in cls.model_fields.items():
                field_names.add(field_name)
                if alias := getattr(field, "alias", None):
                    field_names.add(alias)
            cls.field_names = field_names

        # # Compare against both field names and aliases
        # if any(k not in field_names for k in data):
        #     logger.warning(
        #         "The following fields were present in the request "
        #         "but ignored: %s",
        #         data.keys() - field_names,
        #     )
        return result


class ErrorResponse(OpenAIBaseModel):
    object: str = "error"
    message: str
    type: str
    param: Optional[str] = None
    code: int


class ModelPermission(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"modelperm-{random_uuid()}")
    object: str = "model_permission"
    created: int = Field(default_factory=lambda: int(time.time()))
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: Optional[str] = None
    is_blocking: bool = False


class ModelCard(OpenAIBaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "vllm"
    root: Optional[str] = None
    parent: Optional[str] = None
    max_model_len: Optional[int] = None
    permission: list[ModelPermission] = Field(default_factory=list)


class ModelList(OpenAIBaseModel):
    object: str = "list"
    data: list[ModelCard] = Field(default_factory=list)


class PromptTokenUsageInfo(OpenAIBaseModel):
    cached_tokens: Optional[int] = None


class UsageInfo(OpenAIBaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0
    prompt_tokens_details: Optional[PromptTokenUsageInfo] = None


class RequestResponseMetadata(BaseModel):
    request_id: str
    final_usage_info: Optional[UsageInfo] = None


class JsonSchemaResponseFormat(OpenAIBaseModel):
    name: str
    description: Optional[str] = None
    # schema is the field in openai but that causes conflicts with pydantic so
    # instead use json_schema with an alias
    json_schema: Optional[dict[str, Any]] = Field(default=None, alias='schema')
    strict: Optional[bool] = None


class ResponseFormat(OpenAIBaseModel):
    # type must be "json_schema", "json_object" or "text"
    type: Literal["text", "json_object", "json_schema"]
    json_schema: Optional[JsonSchemaResponseFormat] = None


class StreamOptions(OpenAIBaseModel):
    include_usage: Optional[bool] = True
    continuous_usage_stats: Optional[bool] = False


class FunctionDefinition(OpenAIBaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[dict[str, Any]] = None


class ChatCompletionToolsParam(OpenAIBaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


class ChatCompletionNamedFunction(OpenAIBaseModel):
    name: str


class ChatCompletionNamedToolChoiceParam(OpenAIBaseModel):
    function: ChatCompletionNamedFunction
    type: Literal["function"] = "function"


class LogitsProcessorConstructor(BaseModel):
    qualname: str
    args: Optional[list[Any]] = None
    kwargs: Optional[dict[str, Any]] = None


LogitsProcessors = list[Union[str, LogitsProcessorConstructor]]

class CustomChatCompletionMessageParam(TypedDict, total=False):
    """Enables custom roles in the Chat Completion API."""
    role: Required[str]
    """The role of the message's author."""

    content: Union[str, list[ChatCompletionContentPartParam]]
    """The contents of the message."""

    name: str
    """An optional name for the participant.

    Provides the model information to differentiate between participants of the
    same role.
    """

    tool_call_id: Optional[str]
    """Tool call that this message is responding to."""

    tool_calls: Optional[Iterable[ChatCompletionMessageToolCallParam]]
    """The tool calls generated by the model, such as function calls."""
    
ChatCompletionMessageParam = CustomChatCompletionMessageParam

class ChatCompletionRequest(OpenAIBaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/chat/create
    messages: list[ChatCompletionMessageParam]
    model: Optional[str] = None
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[dict[str, float]] = None
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = 0
    # TODO(#9845): remove max_tokens when field is removed from OpenAI API
    max_tokens: Optional[int] = Field(
        default=None,
        deprecated=
        'max_tokens is deprecated in favor of the max_completion_tokens field')
    max_completion_tokens: Optional[int] = None
    n: Optional[int] = 1
    presence_penalty: Optional[float] = 0.0
    response_format: Optional[ResponseFormat] = None
    seed: Optional[int] = Field(None, ge=-1e9, le=1e9)
    stop: Optional[Union[str, list[str]]] = Field(default_factory=list)
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    tools: Optional[list[ChatCompletionToolsParam]] = None
    tool_choice: Optional[Union[
        Literal["none"],
        Literal["auto"],
        Literal["required"],
        ChatCompletionNamedToolChoiceParam,
    ]] = "none"

    # NOTE this will be ignored by vLLM -- the model determines the behavior
    parallel_tool_calls: Optional[bool] = False
    user: Optional[str] = None

    # doc: begin-chat-completion-sampling-params
    best_of: Optional[int] = None
    use_beam_search: bool = False
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    length_penalty: float = 1.0
    stop_token_ids: Optional[list[int]] = Field(default_factory=list)
    include_stop_str_in_output: bool = False
    ignore_eos: bool = False
    min_tokens: int = 0
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None
    prompt_logprobs: Optional[int] = None
    # doc: end-chat-completion-sampling-params

    # doc: begin-chat-completion-extra-params
    echo: bool = Field(
        default=False,
        description=(
            "If true, the new message will be prepended with the last message "
            "if they belong to the same role."),
    )
    add_generation_prompt: bool = Field(
        default=True,
        description=
        ("If true, the generation prompt will be added to the chat template. "
         "This is a parameter used by chat template in tokenizer config of the "
         "model."),
    )
    continue_final_message: bool = Field(
        default=False,
        description=
        ("If this is set, the chat will be formatted so that the final "
         "message in the chat is open-ended, without any EOS tokens. The "
         "model will continue this message rather than starting a new one. "
         "This allows you to \"prefill\" part of the model's response for it. "
         "Cannot be used at the same time as `add_generation_prompt`."),
    )
    add_special_tokens: bool = Field(
        default=False,
        description=(
            "If true, special tokens (e.g. BOS) will be added to the prompt "
            "on top of what is added by the chat template. "
            "For most models, the chat template takes care of adding the "
            "special tokens so this should be set to false (as is the "
            "default)."),
    )
    documents: Optional[list[dict[str, str]]] = Field(
        default=None,
        description=
        ("A list of dicts representing documents that will be accessible to "
         "the model if it is performing RAG (retrieval-augmented generation)."
         " If the template does not support RAG, this argument will have no "
         "effect. We recommend that each document should be a dict containing "
         "\"title\" and \"text\" keys."),
    )
    chat_template: Optional[str] = Field(
        default=None,
        description=(
            "A Jinja template to use for this conversion. "
            "As of transformers v4.44, default chat template is no longer "
            "allowed, so you must provide a chat template if the tokenizer "
            "does not define one."),
    )
    chat_template_kwargs: Optional[dict[str, Any]] = Field(
        default=None,
        description=("Additional kwargs to pass to the template renderer. "
                     "Will be accessible by the chat template."),
    )
    mm_processor_kwargs: Optional[dict[str, Any]] = Field(
        default=None,
        description=("Additional kwargs to pass to the HF processor."),
    )
    guided_json: Optional[Union[str, dict, BaseModel]] = Field(
        default=None,
        description=("If specified, the output will follow the JSON schema."),
    )
    guided_regex: Optional[str] = Field(
        default=None,
        description=(
            "If specified, the output will follow the regex pattern."),
    )
    guided_choice: Optional[list[str]] = Field(
        default=None,
        description=(
            "If specified, the output will be exactly one of the choices."),
    )
    guided_grammar: Optional[str] = Field(
        default=None,
        description=(
            "If specified, the output will follow the context free grammar."),
    )
    guided_decoding_backend: Optional[str] = Field(
        default=None,
        description=(
            "If specified, will override the default guided decoding backend "
            "of the server for this specific request. If set, must be either "
            "'outlines' / 'lm-format-enforcer'"),
    )
    guided_whitespace_pattern: Optional[str] = Field(
        default=None,
        description=(
            "If specified, will override the default whitespace pattern "
            "for guided json decoding."),
    )
    priority: int = Field(
        default=0,
        description=(
            "The priority of the request (lower means earlier handling; "
            "default: 0). Any priority other than 0 will raise an error "
            "if the served model does not use priority scheduling."),
    )
    request_id: str = Field(
        default_factory=lambda: f"{random_uuid()}",
        description=(
            "The request_id related to this request. If the caller does "
            "not set it, a random_uuid will be generated. This id is used "
            "through out the inference process and return in response."),
    )
    logits_processors: Optional[LogitsProcessors] = Field(
        default=None,
        description=(
            "A list of either qualified names of logits processors, or "
            "constructor objects, to apply when sampling. A constructor is "
            "a JSON object with a required 'qualname' field specifying the "
            "qualified name of the processor class/factory, and optional "
            "'args' and 'kwargs' fields containing positional and keyword "
            "arguments. For example: {'qualname': "
            "'my_module.MyLogitsProcessor', 'args': [1, 2], 'kwargs': "
            "{'param': 'value'}}."))
    return_tokens_as_token_ids: Optional[bool] = Field(
        default=None,
        description=(
            "If specified with 'logprobs', tokens are represented "
            " as strings of the form 'token_id:{token_id}' so that tokens "
            "that are not JSON-encodable can be identified."))

    # doc: end-chat-completion-extra-params

    # Default sampling parameters for chat completion requests
    _DEFAULT_SAMPLING_PARAMS: dict = {
        "repetition_penalty": 1.0,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
    }


    def _get_guided_json_from_tool(
            self) -> Optional[Union[str, dict, BaseModel]]:
        # user has chosen to not use any tool
        if self.tool_choice == "none" or self.tools is None:
            return None

        # user has chosen to use a named tool
        if type(self.tool_choice) is ChatCompletionNamedToolChoiceParam:
            tool_name = self.tool_choice.function.name
            tools = {tool.function.name: tool.function for tool in self.tools}
            if tool_name not in tools:
                raise ValueError(
                    f"Tool '{tool_name}' has not been passed in `tools`.")
            tool = tools[tool_name]
            return tool.parameters

        if self.tool_choice == "required":
            # Pydantic schema generation cannot be used since the JSON schema
            # has to be constructed for a specific instantiation of a tool list
            # so that parameters of a function are correctly generated
            # based on the chosen function name
            def get_tool_schema(tool: ChatCompletionToolsParam) -> dict:
                return {
                    "properties": {
                        "name": {
                            "type": "string",
                            "enum": [tool.function.name]
                        },
                        # parameters are always generated as '{}' in the final
                        # output if they are missing from the request
                        # (i.e. are None or '{}') so the schema is
                        # updated to produce an empty object in that case
                        "parameters": tool.function.parameters
                        if tool.function.parameters else {
                            "type": "object",
                            "properties": {}
                        }
                    },
                    "required": ["name", "parameters"]
                }

            json_schema = {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "anyOf": [get_tool_schema(tool) for tool in self.tools]
                }
            }
            return json_schema

        return None

    @model_validator(mode="before")
    @classmethod
    def validate_stream_options(cls, data):
        if data.get("stream_options") and not data.get("stream"):
            raise ValueError(
                "Stream options can only be defined when `stream=True`.")

        return data

    @model_validator(mode="before")
    @classmethod
    def check_logprobs(cls, data):
        if (prompt_logprobs := data.get("prompt_logprobs")) is not None:
            if data.get("stream") and prompt_logprobs > 0:
                raise ValueError(
                    "`prompt_logprobs` are not available when `stream=True`.")

            if prompt_logprobs < 0:
                raise ValueError("`prompt_logprobs` must be a positive value.")

        if (top_logprobs := data.get("top_logprobs")) is not None:
            if top_logprobs < 0:
                raise ValueError("`top_logprobs` must be a positive value.")

            if top_logprobs > 0 and not data.get("logprobs"):
                raise ValueError(
                    "when using `top_logprobs`, `logprobs` must be set to true."
                )

        return data

    @model_validator(mode="before")
    @classmethod
    def check_guided_decoding_count(cls, data):
        if isinstance(data, ValueError):
            raise data

        guide_count = sum([
            "guided_json" in data and data["guided_json"] is not None,
            "guided_regex" in data and data["guided_regex"] is not None,
            "guided_choice" in data and data["guided_choice"] is not None
        ])
        # you can only use one kind of guided decoding
        if guide_count > 1:
            raise ValueError(
                "You can only use one kind of guided decoding "
                "('guided_json', 'guided_regex' or 'guided_choice').")
        # you can only either use guided decoding or tools, not both
        if guide_count > 1 and data.get("tool_choice", "none") not in (
                "none",
                "auto",
                "required",
        ):
            raise ValueError(
                "You can only either use guided decoding or tools, not both.")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_tool_usage(cls, data):

        # if "tool_choice" is not specified but tools are provided,
        # default to "auto" tool_choice
        if "tool_choice" not in data and data.get("tools"):
            data["tool_choice"] = "auto"

        # if "tool_choice" is "none" -- ignore tools if present
        if "tool_choice" in data and data["tool_choice"] == "none":
            # ensure that no tools are present
            data.pop("tools", None)
            return data

        # if "tool_choice" is specified -- validation
        if "tool_choice" in data:

            # ensure that if "tool choice" is specified, tools are present
            if "tools" not in data or data["tools"] is None:
                raise ValueError(
                    "When using `tool_choice`, `tools` must be set.")

            # make sure that tool choice is either a named tool
            # OR that it's set to "auto" or "required"
            if data["tool_choice"] not in [
                    "auto", "required"
            ] and not isinstance(data["tool_choice"], dict):
                raise NotImplementedError(
                    f'Invalid value for `tool_choice`: {data["tool_choice"]}! '\
                    'Only named tools, "none", "auto" or "required" '\
                    'are supported.'
                )

            # ensure that if "tool_choice" is specified as an object,
            # it matches a valid tool
            if isinstance(data["tool_choice"], dict):
                valid_tool = False
                specified_function = data["tool_choice"].get("function")
                if not specified_function:
                    raise ValueError(
                        "Expected field `function` in `tool_choice`."
                        " Correct usage: `{\"type\": \"function\","
                        " \"function\": {\"name\": \"my_function\"}}`")
                specified_function_name = specified_function.get("name")
                if not specified_function_name:
                    raise ValueError(
                        "Expected field `name` in `function` in `tool_choice`."
                        "Correct usage: `{\"type\": \"function\", "
                        "\"function\": {\"name\": \"my_function\"}}`")
                for tool in data["tools"]:
                    if tool["function"]["name"] == specified_function_name:
                        valid_tool = True
                        break
                if not valid_tool:
                    raise ValueError(
                        "The tool specified in `tool_choice` does not match any"
                        " of the specified `tools`")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_generation_prompt(cls, data):
        if data.get("continue_final_message") and data.get(
                "add_generation_prompt"):
            raise ValueError("Cannot set both `continue_final_message` and "
                             "`add_generation_prompt` to True.")
        return data


class CompletionRequest(OpenAIBaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/completions/create
    model: Optional[str] = None
    prompt: Union[list[int], list[list[int]], str, list[str]]
    best_of: Optional[int] = None
    echo: Optional[bool] = False
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[dict[str, float]] = None
    logprobs: Optional[int] = None
    max_tokens: Optional[int] = 16
    n: int = 1
    presence_penalty: Optional[float] = 0.0
    seed: Optional[int] = Field(None, ge=-1e9, le=1e9)
    stop: Optional[Union[str, list[str]]] = Field(default_factory=list)
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    suffix: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    user: Optional[str] = None

    # doc: begin-completion-sampling-params
    use_beam_search: bool = False
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    length_penalty: float = 1.0
    stop_token_ids: Optional[list[int]] = Field(default_factory=list)
    include_stop_str_in_output: bool = False
    ignore_eos: bool = False
    min_tokens: int = 0
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None
    allowed_token_ids: Optional[list[int]] = None
    prompt_logprobs: Optional[int] = None
    # doc: end-completion-sampling-params

    # doc: begin-completion-extra-params
    add_special_tokens: bool = Field(
        default=True,
        description=(
            "If true (the default), special tokens (e.g. BOS) will be added to "
            "the prompt."),
    )
    response_format: Optional[ResponseFormat] = Field(
        default=None,
        description=
        ("Similar to chat completion, this parameter specifies the format of "
         "output. Only {'type': 'json_object'}, {'type': 'json_schema'} or "
         "{'type': 'text' } is supported."),
    )
    guided_json: Optional[Union[str, dict, BaseModel]] = Field(
        default=None,
        description="If specified, the output will follow the JSON schema.",
    )
    guided_regex: Optional[str] = Field(
        default=None,
        description=(
            "If specified, the output will follow the regex pattern."),
    )
    guided_choice: Optional[list[str]] = Field(
        default=None,
        description=(
            "If specified, the output will be exactly one of the choices."),
    )
    guided_grammar: Optional[str] = Field(
        default=None,
        description=(
            "If specified, the output will follow the context free grammar."),
    )
    guided_decoding_backend: Optional[str] = Field(
        default=None,
        description=(
            "If specified, will override the default guided decoding backend "
            "of the server for this specific request. If set, must be one of "
            "'outlines' / 'lm-format-enforcer'"),
    )
    guided_whitespace_pattern: Optional[str] = Field(
        default=None,
        description=(
            "If specified, will override the default whitespace pattern "
            "for guided json decoding."),
    )
    priority: int = Field(
        default=0,
        description=(
            "The priority of the request (lower means earlier handling; "
            "default: 0). Any priority other than 0 will raise an error "
            "if the served model does not use priority scheduling."),
    )
    logits_processors: Optional[LogitsProcessors] = Field(
        default=None,
        description=(
            "A list of either qualified names of logits processors, or "
            "constructor objects, to apply when sampling. A constructor is "
            "a JSON object with a required 'qualname' field specifying the "
            "qualified name of the processor class/factory, and optional "
            "'args' and 'kwargs' fields containing positional and keyword "
            "arguments. For example: {'qualname': "
            "'my_module.MyLogitsProcessor', 'args': [1, 2], 'kwargs': "
            "{'param': 'value'}}."))

    return_tokens_as_token_ids: Optional[bool] = Field(
        default=None,
        description=(
            "If specified with 'logprobs', tokens are represented "
            " as strings of the form 'token_id:{token_id}' so that tokens "
            "that are not JSON-encodable can be identified."))

    # doc: end-completion-extra-params

    # Default sampling parameters for completion requests
    _DEFAULT_SAMPLING_PARAMS: dict = {
        "repetition_penalty": 1.0,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
    }

    @model_validator(mode="before")
    @classmethod
    def check_guided_decoding_count(cls, data):
        guide_count = sum([
            "guided_json" in data and data["guided_json"] is not None,
            "guided_regex" in data and data["guided_regex"] is not None,
            "guided_choice" in data and data["guided_choice"] is not None
        ])
        if guide_count > 1:
            raise ValueError(
                "You can only use one kind of guided decoding "
                "('guided_json', 'guided_regex' or 'guided_choice').")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_logprobs(cls, data):
        if (prompt_logprobs := data.get("prompt_logprobs")) is not None:
            if data.get("stream") and prompt_logprobs > 0:
                raise ValueError(
                    "`prompt_logprobs` are not available when `stream=True`.")

            if prompt_logprobs < 0:
                raise ValueError("`prompt_logprobs` must be a positive value.")

        if (logprobs := data.get("logprobs")) is not None and logprobs < 0:
            raise ValueError("`logprobs` must be a positive value.")

        return data

    @model_validator(mode="before")
    @classmethod
    def validate_stream_options(cls, data):
        if data.get("stream_options") and not data.get("stream"):
            raise ValueError(
                "Stream options can only be defined when `stream=True`.")

        return data


class EmbeddingCompletionRequest(OpenAIBaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/embeddings
    model: Optional[str] = None
    input: Union[list[int], list[list[int]], str, list[str]]
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None

    # doc: begin-embedding-pooling-params
    additional_data: Optional[Any] = None
    # doc: end-embedding-pooling-params

    # doc: begin-embedding-extra-params
    add_special_tokens: bool = Field(
        default=True,
        description=(
            "If true (the default), special tokens (e.g. BOS) will be added to "
            "the prompt."),
    )
    priority: int = Field(
        default=0,
        description=(
            "The priority of the request (lower means earlier handling; "
            "default: 0). Any priority other than 0 will raise an error "
            "if the served model does not use priority scheduling."),
    )


class EmbeddingChatRequest(OpenAIBaseModel):
    model: Optional[str] = None
    messages: list[ChatCompletionMessageParam]

    encoding_format: Literal["float", "base64"] = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None

    # doc: begin-chat-embedding-pooling-params
    additional_data: Optional[Any] = None
    # doc: end-chat-embedding-pooling-params

    # doc: begin-chat-embedding-extra-params
    add_special_tokens: bool = Field(
        default=False,
        description=(
            "If true, special tokens (e.g. BOS) will be added to the prompt "
            "on top of what is added by the chat template. "
            "For most models, the chat template takes care of adding the "
            "special tokens so this should be set to false (as is the "
            "default)."),
    )
    chat_template: Optional[str] = Field(
        default=None,
        description=(
            "A Jinja template to use for this conversion. "
            "As of transformers v4.44, default chat template is no longer "
            "allowed, so you must provide a chat template if the tokenizer "
            "does not define one."),
    )
    chat_template_kwargs: Optional[dict[str, Any]] = Field(
        default=None,
        description=("Additional kwargs to pass to the template renderer. "
                     "Will be accessible by the chat template."),
    )
    mm_processor_kwargs: Optional[dict[str, Any]] = Field(
        default=None,
        description=("Additional kwargs to pass to the HF processor."),
    )
    priority: int = Field(
        default=0,
        description=(
            "The priority of the request (lower means earlier handling; "
            "default: 0). Any priority other than 0 will raise an error "
            "if the served model does not use priority scheduling."),
    )
    # doc: end-chat-embedding-extra-params

    @model_validator(mode="before")
    @classmethod
    def check_generation_prompt(cls, data):
        if data.get("continue_final_message") and data.get(
                "add_generation_prompt"):
            raise ValueError("Cannot set both `continue_final_message` and "
                             "`add_generation_prompt` to True.")
        return data


EmbeddingRequest = Union[EmbeddingCompletionRequest, EmbeddingChatRequest]

PoolingCompletionRequest = EmbeddingCompletionRequest
PoolingChatRequest = EmbeddingChatRequest
PoolingRequest = Union[PoolingCompletionRequest, PoolingChatRequest]


class ScoreRequest(OpenAIBaseModel):
    model: Optional[str] = None
    text_1: Union[list[str], str]
    text_2: Union[list[str], str]
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None

    # doc: begin-score-pooling-params
    additional_data: Optional[Any] = None
    # doc: end-score-pooling-params

    # doc: begin-score-extra-params
    priority: int = Field(
        default=0,
        description=(
            "The priority of the request (lower means earlier handling; "
            "default: 0). Any priority other than 0 will raise an error "
            "if the served model does not use priority scheduling."),
    )


class RerankRequest(OpenAIBaseModel):
    model: Optional[str] = None
    query: str
    documents: list[str]
    top_n: int = Field(default_factory=lambda: 0)
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None

    # doc: begin-rerank-pooling-params
    additional_data: Optional[Any] = None
    # doc: end-rerank-pooling-params

    # doc: begin-rerank-extra-params
    priority: int = Field(
        default=0,
        description=(
            "The priority of the request (lower means earlier handling; "
            "default: 0). Any priority other than 0 will raise an error "
            "if the served model does not use priority scheduling."),
    )


class RerankDocument(BaseModel):
    text: str


class RerankResult(BaseModel):
    index: int
    document: RerankDocument
    relevance_score: float


class RerankUsage(BaseModel):
    total_tokens: int


class RerankResponse(OpenAIBaseModel):
    id: str
    model: str
    usage: RerankUsage
    results: list[RerankResult]


class CompletionLogProbs(OpenAIBaseModel):
    text_offset: list[int] = Field(default_factory=list)
    token_logprobs: list[Optional[float]] = Field(default_factory=list)
    tokens: list[str] = Field(default_factory=list)
    top_logprobs: list[Optional[dict[str,
                                     float]]] = Field(default_factory=list)

@dataclass
class Logprob:
    """Infos for supporting OpenAI compatible logprobs and token ranks.

    Attributes:
        logprob: The logprob of chosen token
        rank: The vocab rank of chosen token (>=1)
        decoded_token: The decoded chosen token index
    """
    logprob: float
    rank: Optional[int] = None
    decoded_token: Optional[str] = None

class CompletionResponseChoice(OpenAIBaseModel):
    index: int
    text: str
    logprobs: Optional[CompletionLogProbs] = None
    finish_reason: Optional[str] = None
    stop_reason: Optional[Union[int, str]] = Field(
        default=None,
        description=(
            "The stop string or token id that caused the completion "
            "to stop, None if the completion finished for some other reason "
            "including encountering the EOS token"),
    )
    prompt_logprobs: Optional[list[Optional[dict[int, Logprob]]]] = None


class CompletionResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{random_uuid()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionResponseChoice]
    usage: UsageInfo


class CompletionResponseStreamChoice(OpenAIBaseModel):
    index: int
    text: str
    logprobs: Optional[CompletionLogProbs] = None
    finish_reason: Optional[str] = None
    stop_reason: Optional[Union[int, str]] = Field(
        default=None,
        description=(
            "The stop string or token id that caused the completion "
            "to stop, None if the completion finished for some other reason "
            "including encountering the EOS token"),
    )


class CompletionStreamResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{random_uuid()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(default=None)


class EmbeddingResponseData(OpenAIBaseModel):
    index: int
    object: str = "embedding"
    embedding: Union[list[float], str]


class EmbeddingResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"embd-{random_uuid()}")
    object: str = "list"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    data: list[EmbeddingResponseData]
    usage: UsageInfo


class PoolingResponseData(OpenAIBaseModel):
    index: int
    object: str = "pooling"
    data: Union[list[list[float]], list[float], str]


class PoolingResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"pool-{random_uuid()}")
    object: str = "list"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    data: list[PoolingResponseData]
    usage: UsageInfo


class ScoreResponseData(OpenAIBaseModel):
    index: int
    object: str = "score"
    score: float


class ScoreResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"embd-{random_uuid()}")
    object: str = "list"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    data: list[ScoreResponseData]
    usage: UsageInfo


class FunctionCall(OpenAIBaseModel):
    name: str
    arguments: str


class ToolCall(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-tool-{random_uuid()}")
    type: Literal["function"] = "function"
    function: FunctionCall


class DeltaFunctionCall(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None


# a tool call delta where everything is optional
class DeltaToolCall(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-tool-{random_uuid()}")
    type: Literal["function"] = "function"
    index: int = Field(default_factory=lambda: -1)
    function: Optional[DeltaFunctionCall] = None


class ExtractedToolCallInformation(BaseModel):
    # indicate if tools were called
    tools_called: bool

    # extracted tool calls
    tool_calls: list[ToolCall]

    # content - per OpenAI spec, content AND tool calls can be returned rarely
    # But some models will do this intentionally
    content: Optional[str] = None


class ChatMessage(OpenAIBaseModel):
    role: str
    reasoning_content: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = Field(default_factory=list)


class ChatCompletionLogProb(OpenAIBaseModel):
    token: str
    logprob: float = -9999.0
    bytes: Optional[list[int]] = None


class ChatCompletionLogProbsContent(ChatCompletionLogProb):
    # Workaround: redefine fields name cache so that it's not
    # shared with the super class.
    field_names: ClassVar[Optional[set[str]]] = None
    top_logprobs: list[ChatCompletionLogProb] = Field(default_factory=list)


class ChatCompletionLogProbs(OpenAIBaseModel):
    content: Optional[list[ChatCompletionLogProbsContent]] = None


class ChatCompletionResponseChoice(OpenAIBaseModel):
    index: int
    message: ChatMessage
    logprobs: Optional[ChatCompletionLogProbs] = None
    # per OpenAI spec this is the default
    finish_reason: Optional[str] = "stop"
    # not part of the OpenAI spec but included in vLLM for legacy reasons
    stop_reason: Optional[Union[int, str]] = None


class ChatCompletionResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: UsageInfo
    prompt_logprobs: Optional[list[Optional[dict[int, Logprob]]]] = None


class DeltaMessage(OpenAIBaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: list[DeltaToolCall] = Field(default_factory=list)


class ChatCompletionResponseStreamChoice(OpenAIBaseModel):
    index: int
    delta: DeltaMessage
    logprobs: Optional[ChatCompletionLogProbs] = None
    finish_reason: Optional[str] = None
    stop_reason: Optional[Union[int, str]] = None


class ChatCompletionStreamResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(default=None)

class PromptErrorResponse(OpenAIBaseModel):
    type: Literal["error"] = "error"
    message: str
    details: Optional[str] = None

class BatchRequestInput(OpenAIBaseModel):
    """
    The per-line object of the batch input file.

    NOTE: Currently only the `/v1/chat/completions` endpoint is supported.
    """

    # A developer-provided per-request id that will be used to match outputs to
    # inputs. Must be unique for each request in a batch.
    custom_id: str

    # The HTTP method to be used for the request. Currently only POST is
    # supported.
    method: str

    # The OpenAI API relative URL to be used for the request. Currently
    # /v1/chat/completions is supported.
    url: str

    # The parameters of the request.
    body: Union[ChatCompletionRequest, EmbeddingRequest, ScoreRequest]

    @field_validator('body', mode='plain')
    @classmethod
    def check_type_for_url(cls, value: Any, info: ValidationInfo):
        # Use url to disambiguate models
        url = info.data['url']
        if url == "/v1/chat/completions":
            return ChatCompletionRequest.model_validate(value)
        if url == "/v1/embeddings":
            return TypeAdapter(EmbeddingRequest).validate_python(value)
        if url == "/v1/score":
            return ScoreRequest.model_validate(value)
        return TypeAdapter(Union[ChatCompletionRequest, EmbeddingRequest,
                                 ScoreRequest]).validate_python(value)


class BatchResponseData(OpenAIBaseModel):
    # HTTP status code of the response.
    status_code: int = 200

    # An unique identifier for the API request.
    request_id: str

    # The body of the response.
    body: Optional[Union[ChatCompletionResponse, EmbeddingResponse,
                         ScoreResponse]] = None


class BatchRequestOutput(OpenAIBaseModel):
    """
    The per-line object of the batch output and error files
    """

    id: str

    # A developer-provided per-request id that will be used to match outputs to
    # inputs.
    custom_id: str

    response: Optional[BatchResponseData]

    # For requests that failed with a non-HTTP error, this will contain more
    # information on the cause of the failure.
    error: Optional[Any]