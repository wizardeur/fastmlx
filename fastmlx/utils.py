import json
import os
import re
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Generator, List, Union

from jinja2 import Environment, FileSystemLoader

from .types.chat.chat_completion import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    FunctionCall,
    ToolCall,
)

# MLX Imports
try:
    import mlx.core as mx
    from mlx_lm import load as lm_load
    from mlx_lm import models as lm_models
    from mlx_lm.tokenizer_utils import TokenizerWrapper
    from mlx_lm.utils import generate_step
    from mlx_lm.utils import stream_generate as lm_stream_generate
    from mlx_vlm import load as vlm_load
    from mlx_vlm import models as vlm_models
    from mlx_vlm.utils import load_image_processor
    from mlx_vlm.utils import stream_generate as vlm_stream_generate
except ImportError:
    print("Warning: mlx or mlx_lm not available. Some functionality will be limited.")


TOOLS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "tools"))


def get_model_type_list(models, type="vlm"):

    # Get the directory path of the models package
    models_dir = os.path.dirname(models.__file__)

    # List all items in the models directory
    all_items = os.listdir(models_dir)

    if type == "vlm":
        submodules = [
            item
            for item in all_items
            if os.path.isdir(os.path.join(models_dir, item))
            and not item.startswith(".")
            and item != "__pycache__"
        ]
        return submodules
    else:

        return [item for item in all_items if not item.startswith("__")]


MODELS = {
    "vlm": get_model_type_list(vlm_models),
    "lm": get_model_type_list(lm_models, "lm"),
}
MODEL_REMAPPING = {"llava-qwen2": "llava_bunny", "bunny-llama": "llava_bunny"}


@contextmanager
def working_directory(path):
    """A context manager to change the working directory temporarily."""
    current_path = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(current_path)


def load_tools_config():
    with working_directory(TOOLS_PATH):
        with open("config.json", "r") as file:
            return json.load(file)


def get_model_type(model_name, available_models):
    # Convert model name to lowercase for case-insensitive matching
    model_name_lower = model_name.lower().replace(".", "_")

    # Check if any of the available model types are in the model name
    for model_type in available_models:
        if model_type != "default" and model_type in model_name_lower:
            return model_type

    # If no match is found, return 'default'
    return "default"


def get_tool_prompt(model_name, tools, prompt, tool_choice, parallel_tool_calls):
    tool_config = load_tools_config()
    available_models = tool_config["models"].keys()
    model_type = get_model_type(model_name, available_models)
    model_config = tool_config["models"].get(
        model_type, tool_config["models"]["default"]
    )
    env = Environment(loader=FileSystemLoader(TOOLS_PATH))
    template = env.get_template(model_config["prompt_template"])

    tool_definitions = json.dumps([tool['function'] for tool in tools], indent=2)
    if model_config.get("query", False):
        rendered_prompt=template.render(
                tools=tool_definitions,
                parallel_tool_calling=model_config.get("parallel_tool_calling", False),
                current_date=datetime.now().strftime("%d %b %Y"),
                query=prompt,
                tool_choice=tool_choice
            )
        return (
            template.render(
                tools=tool_definitions,
                parallel_tool_calling=model_config.get("parallel_tool_calling", False),
                current_date=datetime.now().strftime("%d %b %Y"),
                query=prompt,
                tool_choice=tool_choice
            ),
            True,
        )
    else:
        rendered_prompt=template.render(
                tools=tool_definitions,
                parallel_tool_calling=model_config.get("parallel_tool_calling", False),
                current_date=datetime.now().strftime("%d %b %Y"),
                tool_choice=tool_choice
            )
        return (
            template.render(
                tools=tool_definitions,
                parallel_tool_calling=model_config.get("parallel_tool_calling", False),
                current_date=datetime.now().strftime("%d %b %Y"),
                tool_choice=tool_choice
            ),
            False,
        )


def get_eom_token(model_name):
    tool_config = load_tools_config()
    available_models = tool_config["models"].keys()
    model_type = get_model_type(model_name, available_models)
    model_config = tool_config["models"].get(
        model_type, tool_config["models"]["default"]
    )
    eom_token = model_config.get("eom_token", None)
    return eom_token


def apply_lm_chat_template(
    tokenizer: Any, chat_messages: List[Dict], request: ChatCompletionRequest
) -> str:
    if tokenizer.chat_template is not None and hasattr(
        tokenizer, "apply_chat_template"
    ):
        if "firefunction-v2" in request.model:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return tokenizer.apply_chat_template(
                chat_messages,
                functions=json.dumps(
                    [tool.model_dump() for tool in request.tools], indent=4
                ),
                datetime=now,
                tokenize=False,
            )
        else:
            return tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    else:
        return request.messages[-1].content


def handle_function_calls(output: str, request):
    tool_calls = []

    # Check for JSON format tool calls
    json_match = re.search(r'\{.*"tool_calls":\s*\[.*\].*\}', output, re.DOTALL)
    json_match_name = re.search(r'\{.*"name":\s*".*",\s*"parameters":\s*\{.*\}\s*\}', output, re.DOTALL) 
    if json_match:
        try:
            json_data = json.loads(json_match.group())
            for call in json_data.get("tool_calls", []):
                tool_calls.append(
                    ToolCall(
                        id=f"call_{os.urandom(4).hex()}",
                        function=FunctionCall(
                            name=call["name"], arguments=json.dumps(call["arguments"])
                        ),
                    )
                )
            # Remove the JSON from the output
            output = re.sub(
                r'\{.*"tool_calls":\s*\[.*\].*\}', "", output, flags=re.DOTALL
            ).strip()
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON tool calls: {e}")

    # Parse JSON-based calls as per the llama-3.1 spec (works for llama-3.2 too)
    elif json_match_name:
        try:
            tool_call = json.loads(json_match_name.group())
            tool_calls.append(
                ToolCall(
                    id=f"call_{os.urandom(4).hex()}",
                    function=FunctionCall(
                        name=tool_call["name"],
                        arguments=json.dumps(tool_call["parameters"])
                    ),
                )
            )
            output = re.sub(r'\{.*"name":\s*".*",\s*"parameters":\s*\{.*\}\s*\}', "", output, flags=re.DOTALL).strip()
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON tool call: {e}")

    # Check for XML-style function calls
    # Check for function calls in both old and new XML formats
    elif "<function_calls>" in output.lower():
        try:
            # Try parsing old format
            function_calls = re.findall(r"<function=(\w+)>\s*({[^<>]+})", output)
            for function_name, args_str in function_calls:
                args = json.loads(args_str)
                tool_calls.append(
                    ToolCall(
                        id=f"call_{os.urandom(4).hex()}",
                        function=FunctionCall(
                            name=function_name, arguments=json.dumps(args)
                        ),
                    )
                )

            # Try parsing new XML format
            invoke_blocks = re.findall(
                r"<invoke>(.*?)</invoke>", output, re.DOTALL | re.IGNORECASE
            )
            for block in invoke_blocks:
                tool_name = re.search(
                    r"<tool_name>(.*?)</tool_name>", block, re.IGNORECASE
                )
                parameters = re.findall(r"<(\w+)>(.*?)</\1>", block, re.IGNORECASE)

                if tool_name:
                    args = {
                        param[0].lower(): param[1]
                        for param in parameters
                        if param[0].lower() != "tool_name"
                    }
                    tool_calls.append(
                        ToolCall(
                            id=f"call_{os.urandom(4).hex()}",
                            function=FunctionCall(
                                name=tool_name.group(1), arguments=json.dumps(args)
                            ),
                        )
                    )

            # Remove the function calls from the output
            output = re.sub(
                r"<function_calls>.*</function_calls>",
                "",
                output,
                flags=re.DOTALL | re.IGNORECASE,
            ).strip()
        except Exception as e:
            print(f"Error parsing function call: {e}")

    elif "functools[" in output:
        try:
            functools_match = re.search(r"functools\[(.*?)\]", output, re.DOTALL)
            if functools_match:
                functools_data = json.loads(f"[{functools_match.group(1)}]")
                for call in functools_data:
                    tool_calls.append(
                        ToolCall(
                            id=f"call_{os.urandom(4).hex()}",
                            function=FunctionCall(
                                name=call["name"],
                                arguments=json.dumps(call["arguments"]),
                            ),
                        )
                    )
                # Remove the functools call from the output
                output = re.sub(
                    r"functools\[.*?\]", "", output, flags=re.DOTALL
                ).strip()
        except Exception as e:
            print(f"Error parsing functools call: {e}")

    # Handle tool_choice
    if hasattr(request, 'tool_choice'):
        if request.tool_choice == "none":
            tool_calls = []
        elif request.tool_choice == "auto":
            # Keep existing tool_calls
            pass
        elif isinstance(request.tool_choice, dict) and request.tool_choice.get("type") == "function":
            function_name = request.tool_choice["function"]["name"]
            # Find the matching tool call or create a new one
            matching_call = next((call for call in tool_calls if call.function.name == function_name), None)
            if matching_call:
                tool_calls = [matching_call]
            else:
                tool_calls = [
                    ToolCall(
                        id=f"call_{os.urandom(4).hex()}",
                        function=FunctionCall(
                            name=function_name,
                            arguments=json.dumps({}),  # You may need to extract arguments from the output
                        ),
                    )
                ]

    # Prepare the response
    response = ChatCompletionResponse(
        id=f"chatcmpl-{os.urandom(4).hex()}",
        created=int(time.time()),
        model=request.model,
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": output, "tool_calls": tool_calls},
                "finish_reason": "stop" if not tool_calls else "tool_call",
            }
        ],
        tool_calls=tool_calls,
    )

    return response


# Model Loading and Generation Functions
def load_vlm_model(model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    model, processor = vlm_load(model_name, {"trust_remote_code": True})
    image_processor = load_image_processor(model_name)
    return {
        "model": model,
        "processor": processor,
        "image_processor": image_processor,
        "config": config,
    }


def load_lm_model(model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    model, tokenizer = lm_load(model_name)
    return {"model": model, "tokenizer": tokenizer, "config": config}


def vlm_stream_generator(
    model,
    model_name,
    processor,
    image,
    prompt,
    image_processor,
    max_tokens,
    temperature,
):
    for token in vlm_stream_generate(
        model,
        processor,
        image,
        prompt,
        image_processor,
        max_tokens=max_tokens,
        temp=temperature,
    ):
        chunk = ChatCompletionChunk(
            id=f"chatcmpl-{os.urandom(4).hex()}",
            created=int(time.time()),
            model=model_name,
            choices=[
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": token},
                    "finish_reason": None,
                }
            ],
        )
        yield f"data: {json.dumps(chunk.model_dump())}\n\n"
    yield "data: [DONE]\n\n"


def lm_generate(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 100,
    **kwargs,
) -> Union[str, Generator[str, None, None]]:
    """
    Generate a complete response from the model.

    Args:
       model (nn.Module): The language model.
       tokenizer (PreTrainedTokenizer): The tokenizer.
       prompt (str): The string prompt.
       max_tokens (int): The maximum number of tokens. Default: ``100``.
       verbose (bool): If ``True``, print tokens and timing information.
           Default: ``False``.
       formatter (Optional[Callable]): A function which takes a token and a
           probability and displays it.
       kwargs: The remaining options get passed to :func:`generate_step`.
          See :func:`generate_step` for more details.
    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    stop_words = kwargs.pop("stop_words", [])

    stop_words_id = (
        tokenizer._tokenizer(stop_words)["input_ids"][0] if stop_words else None
    )

    prompt_tokens = mx.array(tokenizer.encode(prompt))
    detokenizer = tokenizer.detokenizer

    detokenizer.reset()

    for (token, logprobs), n in zip(
        generate_step(prompt_tokens, model, **kwargs),
        range(max_tokens),
    ):
        if token == tokenizer.eos_token_id or (
            stop_words_id and token in stop_words_id
        ):
            break

        detokenizer.add_token(token)

    detokenizer.finalize()
    return detokenizer.text


def lm_stream_generator(
    model, model_name, tokenizer, prompt, max_tokens, temperature, **kwargs
):
    stop_words = kwargs.pop("stop_words", [])

    for token in lm_stream_generate(
        model, tokenizer, prompt, max_tokens=max_tokens, temp=temperature
    ):
        if stop_words and token in stop_words:
            break

        chunk = ChatCompletionChunk(
            id=f"chatcmpl-{os.urandom(4).hex()}",
            created=int(time.time()),
            model=model_name,
            choices=[
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": token},
                    "finish_reason": None,
                }
            ],
        )
        yield f"data: {json.dumps(chunk.model_dump())}\n\n"

    yield "data: [DONE]\n\n"
