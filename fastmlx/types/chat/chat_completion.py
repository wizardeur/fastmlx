from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: FunctionCall


class FunctionParameters(BaseModel):
    type: str
    required: List[str]
    properties: Dict[str, Any]


class FunctionDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: FunctionParameters


class Function(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]


class FunctionTool(BaseModel):
    type: Literal["function"]
    function: FunctionDefinition


class ChatCompletionContentPartParam(BaseModel):
    type: Literal["text", "image_url"]
    text: str = None
    image_url: dict = None


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[ChatCompletionContentPartParam]]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    image: Optional[str] = Field(default=None)
    max_tokens: Optional[int] = Field(default=1024)
    stream: Optional[bool] = Field(default=False)
    temperature: Optional[float] = Field(default=0.2)
    tools: Optional[List[FunctionTool]] = Field(default=None)
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(default=None)
    parallel_tool_calls: Optional[bool] = Field(default=False)


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]
    tool_calls: Optional[List[ToolCall]] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[Dict[str, Any]]
