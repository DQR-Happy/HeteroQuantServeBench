"""Input formatting without device imports or synthesized chat templates."""


class ContextLimitExceeded(ValueError):
    """Input admission failed before input transfer or a model forward."""

    def __init__(self, input_tokens, output_tokens, context_limit):
        self.budget = {"input_tokens": input_tokens, "context_limit": context_limit,
                       "available_output_tokens": max(0, context_limit - input_tokens)}
        super().__init__(f"输入 {input_tokens} tokens ＋输出预算 {output_tokens} tokens 超过总窗口 {context_limit} tokens；"
                         f"请将输出预算降至 {self.budget['available_output_tokens']} tokens 以内，或缩短输入。模型仍可继续使用。")

def prepare_prompt(tokenizer, request, model_type):
    if request.get("input_mode") == "completion" or request.get("prompt") is not None:
        if request.get("messages") or not isinstance(request.get("prompt"), str) or not request["prompt"].strip():
            raise ValueError("COMPLETION_PROMPT_REQUIRED")
        # Preserve publisher tokenization, including a native BOS when required.
        return request["prompt"], True
    return tokenizer.apply_chat_template(
        request["messages"], tokenize=False, add_generation_prompt=True,
        **({"enable_thinking": False} if model_type == "qwen3" else {}),
    ), False
