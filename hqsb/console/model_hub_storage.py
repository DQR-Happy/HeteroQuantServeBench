"""Paths owned by a ModelScope acquisition plan and their removal."""
import re
import shutil
from pathlib import Path

from hqsb.core.optimization import fingerprint


def artifact_paths(root: Path, item: dict) -> list[Path]:
    digest = item["plan_sha256"]
    if not re.fullmatch(r"[0-9a-f]{64}", digest) or fingerprint(item["plan"]) != digest:
        raise ValueError("下载计划身份校验失败，请检查该下载记录")
    paths = [root / digest, root / ("partial-" + digest)]
    if item.get("snapshot_path") and Path(item["snapshot_path"]).absolute() != paths[0]:
        raise ValueError("制品目录与下载计划记录不一致，请检查该下载记录")
    for path in paths:
        if path.is_symlink() or path.resolve() != path or (path.exists() and not path.is_dir()):
            raise ValueError("制品目录类型已变化，请检查节点上的目录")
    quarantine = root / "quarantine"
    if quarantine.is_symlink():
        raise ValueError("隔离目录类型已变化，请检查节点上的目录")
    identity = item["id"]
    if not re.fullmatch(r"download-[0-9a-f]{32}", identity):
        raise ValueError("下载记录身份校验失败，请检查该下载记录")
    for path in quarantine.glob(identity + "-*"):
        if re.fullmatch(re.escape(identity) + r"-[0-9a-f]{32}", path.name):
            if path.is_symlink() or path.resolve() != path or not path.is_file():
                raise ValueError("隔离文件类型已变化，请检查节点上的目录")
            paths.append(path)
    return paths


def remove_artifacts(paths: list[Path]):
    for path in paths:
        if path.is_symlink() or path.resolve() != path:
            raise ValueError("制品目录类型已变化，请检查节点上的目录")
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
