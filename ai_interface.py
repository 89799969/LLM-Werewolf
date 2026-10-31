# ai_interface.py (LangChain 版 - 基于 LangChain 框架统一调用在线/离线大模型 + 颜色日志)
#
# 本模块是游戏与大模型之间的核心通信层，基于 LangChain 框架实现：
# - 通过 langchain_openai.ChatOpenAI 统一接入任何 OpenAI 兼容 API：
#   * 在线大模型: OpenAI GPT 系列、阿里云百炼/DashScope Qwen 系列、DeepSeek 等
#   * 离线大模型: Ollama (http://localhost:11434/v1)、LM Studio、vLLM 等本地部署服务
# - 针对 Qwen 开启 enable_thinking 的深度思考流式输出，
#   通过 LangChain 流式接口自动分离思考内容 (reasoning_content) 与正式回答 (content)，
#   且不打印思考过程。
# - 其余类型依赖 parse_ai_response 进行 <think> 标签清理与 ===THOUGHT===/===ACTION=== 分离。

import traceback
from typing import List, Dict, Any, Optional, Tuple

# --- LangChain 核心依赖 ---
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

# 假设 terminal_colors.py 在项目根目录或者Python可以找到的路径下
try:
    from terminal_colors import (
        colorize, log_level_color, Colors,
        red, green, yellow, blue, magenta, cyan, grey, bold
    )
except ImportError:
    # Fallback if terminal_colors is not found
    def colorize(text: str, _color_code: str) -> str: return text
    def log_level_color(_level: str) -> str: return ""
    class Colors: RESET = ""; BOLD = ""; RED = ""; GREEN = ""; YELLOW = ""; BLUE = ""; MAGENTA = ""; CYAN = ""; BRIGHT_BLACK = ""; BRIGHT_MAGENTA = ""; BRIGHT_YELLOW = ""; BRIGHT_RED = ""
    def red(text: str) -> str: return text
    def green(text: str) -> str: return text
    def yellow(text: str) -> str: return text
    def blue(text: str) -> str: return text
    def magenta(text: str) -> str: return text
    def cyan(text: str) -> str: return text
    def grey(text: str) -> str: return text
    def bold(text: str) -> str: return text

from game_config import DEFAULT_API_ENDPOINT, DEFAULT_API_KEY, DEFAULT_MODEL_NAME
from response_parser import parse_ai_response # 用于处理<think>标签与===THOUGHT===/===ACTION===分离

MODULE_COLOR = Colors.BLUE # AIComms 用蓝色

def _log_ai_comms(message: str, level: str = "INFO", player_config_name: Optional[str] = None):
    """AI通信模块的日志记录器。"""
    level_colored = colorize(level, log_level_color(level))
    prefix_module = colorize("[AIComms:", MODULE_COLOR)

    prefix = f"{prefix_module}{level_colored}]"
    if player_config_name:
        p_name_colored = colorize(player_config_name, Colors.BRIGHT_MAGENTA) # AI玩家名用亮品红
        prefix += f" ({p_name_colored})"

    print(f"{prefix} {message}")

# ---------------------------------------------------------------------------
# LangChain 辅助函数
# ---------------------------------------------------------------------------

def _endpoint_to_base_url(endpoint: str) -> str:
    """
    将完整的 chat/completions 端点 URL 转换为 LangChain ChatOpenAI 所需的 base_url。
    例如:
      "http://localhost:11434/v1/chat/completions" -> "http://localhost:11434/v1"
      "https://api.openai.com/v1"                  -> "https://api.openai.com/v1" (原样保留)
    """
    url = (endpoint or "").strip().rstrip("/")
    if url.endswith("/chat/completions"):
        url = url[: -len("/chat/completions")]
    return url


def _convert_messages_to_langchain(messages: List[Dict[str, str]]) -> List[Any]:
    """将游戏内部 {role, content} 消息列表转换为 LangChain 消息对象列表。"""
    lc_messages: List[Any] = []
    for msg in messages:
        role = str(msg.get("role", "user")).lower()
        content = str(msg.get("content", ""))
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
        else:
            lc_messages.append(HumanMessage(content=content))
    return lc_messages


def _build_chat_model(
    api_endpoint: Optional[str],
    api_key: Optional[str],
    model_name: Optional[str],
    timeout_seconds: int,
    extra_body: Optional[Dict[str, Any]] = None
) -> ChatOpenAI:
    """构建 LangChain ChatOpenAI 模型实例 (兼容在线与离线 OpenAI 兼容服务)。"""
    base_url = _endpoint_to_base_url(api_endpoint or DEFAULT_API_ENDPOINT)
    key_to_use = api_key if api_key is not None else DEFAULT_API_KEY
    if not key_to_use or not str(key_to_use).strip():
        key_to_use = "EMPTY" # 本地服务 (如 Ollama) 不需要真实密钥

    return ChatOpenAI(
        model=model_name or DEFAULT_MODEL_NAME,
        base_url=base_url,
        api_key=str(key_to_use),
        temperature=0.7,
        top_p=0.95,
        max_tokens=4096,
        timeout=timeout_seconds,
        max_retries=0, # 重试逻辑由上层 (player_interaction) 统一控制
        extra_body=extra_body, # 额外顶层请求参数，如 {"enable_thinking": True}
    )


def _extract_text(content: Any) -> str:
    """从 LangChain 消息的 content 中提取纯文本 (兼容字符串与多模态块列表)。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if text:
                    parts.append(str(text))
        return "".join(parts)
    return str(content) if content else ""


def _stream_qwen_with_thinking(
    llm: ChatOpenAI,
    lc_messages: List[Any],
    player_config_name: Optional[str]
) -> str:
    """
    通过 LangChain 流式接口处理 Qwen 深度思考输出：
    收集正式回答内容 (content)，记录但不打印思考内容 (reasoning_content)。
    """
    reasoning_parts: List[str] = []
    answer_parts: List[str] = []
    is_answering_started = False

    _log_ai_comms(colorize("开始接收Qwen深度思考流 (LangChain streaming)...", Colors.GREEN), "DEBUG", player_config_name)

    for chunk in llm.stream(lc_messages):
        additional = getattr(chunk, "additional_kwargs", None) or {}
        reasoning_chunk = additional.get("reasoning_content")
        if reasoning_chunk:
            reasoning_parts.append(str(reasoning_chunk))

        text_chunk = _extract_text(chunk.content)
        if text_chunk:
            if not is_answering_started and text_chunk.strip():
                _log_ai_comms(colorize("Qwen开始正式回复...", Colors.CYAN), "DEBUG", player_config_name)
                is_answering_started = True
            answer_parts.append(text_chunk)

    final_reasoning_text = "".join(reasoning_parts)
    final_answer_text = "".join(answer_parts)
    _log_ai_comms(
        f"Qwen流解析完毕. 思考内容长度: {bold(str(len(final_reasoning_text)))}. 回复内容长度: {bold(str(len(final_answer_text)))}",
        "DEBUG", player_config_name
    )
    return final_answer_text.strip()

# ---------------------------------------------------------------------------
# 主接口 (保持原有函数签名，调用方无需修改)
# ---------------------------------------------------------------------------

def make_api_call_to_ai(
    player_config_name: str,
    messages: List[Dict[str, str]],
    api_endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    response_handler_type: str = "standard",
    player_display_name_for_parser: str = "AI玩家",
    timeout_seconds: int = 180
) -> Tuple[Optional[str], Optional[str]]:
    """
    基于 LangChain 框架向指定的AI模型发送请求，并根据handler_type处理响应。
    返回值: (最终AI输出文本, 错误信息) —— 成功时错误信息为 None。
    """
    endpoint_to_use = api_endpoint or DEFAULT_API_ENDPOINT
    model_to_use = model_name or DEFAULT_MODEL_NAME

    is_qwen_deep_think_stream = response_handler_type == "qwen_stream_with_thinking"

    _log_ai_comms(
        f"向模型 '{colorize(model_to_use, Colors.CYAN)}' @ '{colorize(_endpoint_to_base_url(endpoint_to_use), Colors.BLUE)}' 发送请求 (LangChain). "
        f"Handler: '{colorize(response_handler_type, Colors.YELLOW)}'. Streaming: {colorize(str(is_qwen_deep_think_stream), Colors.BOLD)}",
        "DEBUG", player_config_name
    )
    if messages:
        _log_ai_comms(f"最后消息预览 (user prompt): {grey(messages[-1]['content'][:150])}{grey('...') if len(messages[-1]['content']) > 150 else ''}", "TRACE", player_config_name)

    try:
        # Qwen深度思考模式: 通过 extra_body 传递顶层参数 enable_thinking 开启思考输出
        model_kwargs: Dict[str, Any] = {"enable_thinking": True} if is_qwen_deep_think_stream else {}
        llm = _build_chat_model(endpoint_to_use, api_key, model_name, timeout_seconds, model_kwargs)
        lc_messages = _convert_messages_to_langchain(messages)

        final_ai_output_text: Optional[str] = None

        if is_qwen_deep_think_stream:
            final_answer_text = _stream_qwen_with_thinking(llm, lc_messages, player_config_name)
            if not final_answer_text:
                return None, "Qwen AI流式回复内容为空"
            final_ai_output_text = final_answer_text
        else:
            ai_message = llm.invoke(lc_messages)
            raw_text = _extract_text(ai_message.content)
            # 将 LangChain 返回的文本包装为 parse_ai_response 期望的 OpenAI 格式，
            # 复用现有解析逻辑 (<think>标签清理 / ===THOUGHT===与===ACTION===分离)。
            pseudo_response = {"choices": [{"message": {"content": raw_text}}]}
            final_ai_output_text = parse_ai_response(
                response_data=pseudo_response,
                handler_type=response_handler_type,
                model_name_for_logging=model_to_use,
                player_display_name=player_display_name_for_parser
            )

        if not final_ai_output_text or \
           (isinstance(final_ai_output_text, str) and (
               final_ai_output_text.strip().startswith("（AI未能按预期格式回应）") or
               final_ai_output_text.strip().startswith("（AI内容解析后非文本）")
           )):
            error_detail = f"最终AI输出内容为空或为解析器错误提示: '{colorize(final_ai_output_text if final_ai_output_text else '空响应', Colors.YELLOW)}'"
            _log_ai_comms(error_detail, "WARN", player_config_name)
            return None, f"AI响应处理失败: {final_ai_output_text if final_ai_output_text else '空响应'}"

        return str(final_ai_output_text).strip(), None

    except Exception as e:
        err_text = str(e)
        if "timed out" in err_text.lower() or "timeout" in err_text.lower():
            _log_ai_comms(colorize(f"API调用超时 ({timeout_seconds}s)。", Colors.RED), "ERROR", player_config_name)
            return None, f"API调用超时({timeout_seconds}s)"
        error_msg = colorize(f"LangChain API调用时发生错误: {e}", Colors.RED)
        _log_ai_comms(error_msg, "ERROR", player_config_name)
        tb_str = traceback.format_exc()
        _log_ai_comms(colorize(tb_str, Colors.BRIGHT_RED), "CRITICAL", player_config_name)
        return None, f"LangChain API调用错误: {err_text}"
