"""ModelScope 1.29 metadata and byte transport; never imports an inference runtime."""
from __future__ import annotations

import hashlib
import json
import re
import ast
from functools import lru_cache
from importlib.metadata import distribution
from urllib.parse import quote

from hqsb.models.catalog import runtime_files

# Explicit provenance policy; an arbitrary uploader is not an original publisher.
PUBLISHERS = {
    "Qwen": ("Qwen · 阿里", "publisher"),
    "google": ("Google · Gemma", "publisher"),
    "deepseek-ai": ("DeepSeek", "publisher"),
    "microsoft": ("Microsoft · Phi", "publisher"),
    "ZhipuAI": ("智谱", "publisher"),
    "Shanghai_AI_Laboratory": ("InternLM · 上海人工智能实验室", "publisher"),
    "01ai": ("零一万物 · Yi", "publisher"),
    "LLM-Research": ("Llama 等 · 平台镜像（非原厂组织）", "mirror"),
}
REPO = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
UNSUPPORTED_FORMAT = re.compile(r"mlx|gguf|gptq|awq|fp[48]|int[48]|[2468]bit", re.IGNORECASE)
@lru_cache(maxsize=1)
def native_causal_types():
    """Read installed Transformers' data table without importing torch/CUDA.

    This is structural engine compatibility, not a model execution certificate.
    Only literal mapping entries are accepted; no installed source is executed.
    """
    path = distribution("transformers").locate_file("transformers/models/auto/modeling_auto.py")
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "MODEL_FOR_CAUSAL_LM_MAPPING_NAMES" for target in node.targets):
            if isinstance(node.value, ast.Call) and node.value.args:
                return frozenset(key for key, _ in ast.literal_eval(node.value.args[0]))
    raise ValueError("TRANSFORMERS_CAUSAL_MAPPING_UNAVAILABLE")


def validate_repo(repo):
    if not isinstance(repo, str) or not REPO.fullmatch(repo) or repo.split("/")[0] not in PUBLISHERS:
        raise ValueError("MODEL_PUBLISHER_NOT_ALLOWED:请选择已登记发布方的仓库")
    if any(part in {".", ".."} for part in repo.split("/")):
        raise ValueError("MODEL_REPOSITORY_INVALID")
    return repo


def file_url(repo, entry):
    validate_repo(repo)
    # Use the publisher's per-file commit, not a mutable branch during transfer.
    revision = entry.get("Revision", "")
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", revision):
        raise ValueError("MODEL_FILE_REVISION_MISSING:" + entry["Path"])
    return (f"https://modelscope.cn/api/v1/models/{repo}/repo?Revision={revision}"
            f"&FilePath={quote(entry['Path'], safe='')}")


def summary(owner, raw):
    repo = validate_repo(owner + "/" + raw["Name"])
    info = (raw.get("ModelInfos") or {}).get("safetensor") or {}
    params = info.get("model_size")
    files = info.get("files") or []
    weight_bytes = sum(item["size"] for item in files if str(item.get("name", "")).endswith(".safetensors") and type(item.get("size")) is int and item["size"] >= 0)
    return {"model_id": repo, "name": raw["Name"], "publisher": owner,
            "provenance": PUBLISHERS[owner][1], "architectures": raw.get("Architectures") or [],
            "parameters": params if type(params) is int and params > 0 else None,
            "weight_bytes": weight_bytes or None, "license": raw.get("LicenseName") or raw.get("License") or "未声明",
            "url": "https://modelscope.cn/models/" + repo}


class ModelScopeSource:
    def __init__(self):
        # Optional dependency stays at the explicit hub boundary, not app import.
        from modelscope.hub.api import HubApi
        self.api = HubApi(endpoint="https://modelscope.cn", timeout=15, max_retries=0)

    def close(self):
        self.api.session.close()

    def page(self, owner, page, page_size=50):
        if owner not in PUBLISHERS:
            raise ValueError("MODEL_PUBLISHER_NOT_ALLOWED")
        if not 1 <= page_size <= 10000:
            raise ValueError("MODEL_CATALOG_PAGE_SIZE")
        result = self.api.list_models(owner, page_number=page, page_size=page_size)
        return [summary(owner, row) for row in result.get("Models") or []], int(result.get("TotalCount") or 0)

    def revisions(self, repo):
        validate_repo(repo)
        branches, tags = self.api.get_model_branches_and_tags(repo)
        return {"branches": branches, "tags": tags}

    def stream(self, repo, entry, offset=0):
        headers = {"Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        response = self.api.session.get(file_url(repo, entry), headers=headers, stream=True, timeout=(10, 20))
        try:
            response.raise_for_status()
        except Exception:
            response.close()
            raise
        if offset and (response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
            response.close()
            raise ValueError("MODEL_RESUME_RANGE_REJECTED:服务未确认续传范围")
        return response

    def small_json(self, repo, entry):
        if entry["Size"] > 8 * 1024**2:
            raise ValueError("MODEL_METADATA_TOO_LARGE")
        with self.stream(repo, entry) as response:
            content = bytearray()
            for chunk in response.iter_content(64 * 1024):
                content.extend(chunk)
                if len(content) > entry["Size"]:
                    raise ValueError("MODEL_METADATA_SIZE_CHANGED")
        if len(content) != entry["Size"] or hashlib.sha256(content).hexdigest() != entry["Sha256"].lower():
            raise ValueError("MODEL_METADATA_HASH_CHANGED")
        return json.loads(content)

    def inspect(self, repo, revision):
        validate_repo(repo)
        revisions = self.revisions(repo)
        if revision not in revisions["branches"] + revisions["tags"]:
            raise ValueError("MODEL_REVISION_UNKNOWN")
        entries = runtime_files(self.api.get_model_files(repo, revision=revision, recursive=True))
        if len(entries) > 256:
            raise ValueError("MODEL_FILE_LIMIT")
        for entry in entries:
            file_url(repo, entry)
        by_name = {entry["Path"]: entry for entry in entries}
        config = self.small_json(repo, by_name["config.json"])
        tokenizer = self.small_json(repo, by_name["tokenizer_config.json"])
        if "model.safetensors.index.json" in by_name:
            index = self.small_json(repo, by_name["model.safetensors.index.json"])
            shards = set(index.get("weight_map", {}).values())
            if not shards or not shards.issubset(by_name):
                raise ValueError("MODEL_SHARDS_MISSING")
        reasons = []
        model_type = str(config.get("model_type") or "")
        if model_type not in native_causal_types():
            reasons.append("架构尚未接入当前纯文本 FP16 Provider：" + str(config.get("model_type")))
        if config.get("auto_map") or tokenizer.get("auto_map"):
            reasons.append("需要远程自定义代码，当前服务不执行该代码")
        if config.get("quantization_config") or config.get("quantization") or UNSUPPORTED_FORMAT.search(repo.split("/")[1]):
            reasons.append("量化权重尚未接入当前 FP16 加载契约")
        if (config.get("vision_config") or config.get("num_local_experts") or config.get("num_experts")
                or config.get("n_routed_experts") or "moe" in model_type or model_type in {"qwen3_next", "qwen3_5", "qwen3_5_moe"}):
            reasons.append("多模态或 MoE 配置尚未通过本下载部署通路适配")
        if not tokenizer.get("chat_template") and "chat_template.jinja" not in by_name:
            reasons.append("缺少仓库自带聊天模板，交互推理不兼容")
        return {"model_id": repo, "revision": revision, "revisions": revisions,
                "files": [{key: item[key] for key in ("Path", "Size", "Sha256", "Revision")} for item in entries],
                "config": {key: config.get(key) for key in ("model_type", "hidden_size", "num_hidden_layers", "num_key_value_heads", "num_attention_heads", "head_dim", "max_position_embeddings")},
                "download_bytes": sum(item["Size"] for item in entries),
                "weight_bytes": sum(item["Size"] for item in entries if item["Path"].endswith(".safetensors")),
                "runtime_reasons": reasons, "license_url": "https://modelscope.cn/models/" + repo,
                "provenance": PUBLISHERS[repo.split("/")[0]][1],
                "revision_scope": "selected_branch_or_tag_with_publisher_per_file_commit_and_sha256"}
