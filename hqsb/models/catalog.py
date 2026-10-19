"""Curated ModelScope text-model artifacts, independent of torch and the UI.

These checkpoints are candidates for per-device validation. Only Qwen3 1.7B
matches the separately source/shape-pinned HQSB custom-kernel adapter.
"""

MODELS = {
    "qwen3-0.6b": ("Qwen/Qwen3-0.6B", "Qwen3 · 0.6B", "qwen3"),
    "qwen3-4b": ("Qwen/Qwen3-4B", "Qwen3 · 4B", "qwen3"),
    "qwen3-8b": ("Qwen/Qwen3-8B", "Qwen3 · 8B", "qwen3"),
    "llama3.2-1b": ("LLM-Research/Llama-3.2-1B-Instruct", "Llama 3.2 · 1B Instruct", "llama"),
    "llama3.2-3b": ("LLM-Research/Llama-3.2-3B-Instruct", "Llama 3.2 · 3B Instruct", "llama"),
    "gemma3-1b": ("google/gemma-3-1b-it", "Gemma 3 · 1B IT", "gemma3_text"),
    "deepseek-r1-1.5b": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "DeepSeek R1 · Distill Qwen 1.5B", "qwen2"),
}


def runtime_files(entries):
    """Select one root-level safetensors representation and tokenizer assets.

Reject unsafe paths and missing publisher hashes before downloading. Remote
Python code, optimizer state and alternate weight formats are never selected.
"""
    import re

    names = {"config.json", "generation_config.json", "tokenizer.json",
             "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
             "tokenizer.model", "chat_template.jinja", "vocab.json", "merges.txt",
             "model.safetensors.index.json", "README.md", "LICENSE", "LICENSE.txt"}
    selected = []
    for item in entries:
        name = item["Path"]
        if item.get("Type") != "blob" or "/" in name or "\\" in name or name in {"", ".", ".."}:
            continue
        if name not in names and not re.fullmatch(r"model(?:-\d{5}-of-\d{5})?\.safetensors", name):
            continue
        if not re.fullmatch(r"[0-9a-fA-F]{64}", item.get("Sha256", "")):
            raise ValueError("MODELSCOPE_PUBLISHER_HASH_MISSING:" + name)
        if type(item.get("Size")) is not int or item["Size"] < 0:
            raise ValueError("MODELSCOPE_FILE_SIZE_INVALID:" + name)
        selected.append(item)
    paths = [item["Path"] for item in selected]
    if len(set(paths)) != len(paths):
        raise ValueError("MODELSCOPE_DUPLICATE_FILE")
    if not {"config.json", "tokenizer.json", "tokenizer_config.json"}.issubset(paths):
        raise ValueError("MODELSCOPE_RUNTIME_FILES_MISSING")
    if not any(name.endswith(".safetensors") for name in paths):
        raise ValueError("MODELSCOPE_WEIGHTS_MISSING")
    return sorted(selected, key=lambda item: item["Path"])
