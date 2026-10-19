"""Filesystem measurements for node-local model inventory; never opens weights."""
from pathlib import Path
import shutil
import stat

from hqsb.core.optimization import fingerprint

MARKERS = {"config.json", "l4-manifest.json", "l5-manifest.json"}


def measure(path: Path, budget=50000):
    files, errors, seen, signature = [], [], set(), []
    stack = [path]
    while stack and len(signature) < budget:
        current = stack.pop()
        try:
            info = current.lstat()
            signature.append((str(current.relative_to(path)), info.st_dev, info.st_ino,
                              info.st_size, info.st_mtime_ns))
            if stat.S_ISLNK(info.st_mode):
                errors.append(f"符号链接单列，目标空间由目标目录统计：{current}")
            elif stat.S_ISDIR(info.st_mode):
                stack.extend(sorted(current.iterdir(), reverse=True))
            elif stat.S_ISREG(info.st_mode) and (info.st_dev, info.st_ino) not in seen:
                seen.add((info.st_dev, info.st_ino))
                files.append((info.st_dev, info.st_ino, info.st_size, info.st_blocks * 512))
        except FileNotFoundError:
            if current != path:
                errors.append(f"扫描期间文件发生变化：{current}")
        except OSError as exc:
            errors.append(f"{current}: {exc.strerror}")
    if stack:
        errors.append("目录达到本轮文件扫描上限，请缩小扫描目录")
    return {"exists": path.exists(), "logical_bytes": sum(f[2] for f in files),
            "allocated_bytes": sum(f[3] for f in files), "file_count": len(files),
            "complete": not errors, "issues": errors[:12], "revision": fingerprint(signature),
            "inodes": files, "entries_scanned": len(signature)}


def discover(roots: list[Path], budget=5000):
    found, issues, visited = [], [], set()
    stack = [(root, 0) for root in reversed(roots) if root.is_dir()]
    while stack and len(visited) < budget:
        path, depth = stack.pop()
        if path in visited:
            continue
        visited.add(path)
        try:
            if path.is_symlink():
                continue
            entries = list(path.iterdir())
            if any((p.name in MARKERS or p.suffix in {".safetensors", ".gguf"} or p.name == "pytorch_model.bin") and p.is_file() for p in entries):
                found.append(path)
                continue
            if depth < 6:
                stack.extend((p, depth + 1) for p in entries if p.is_dir() and not p.is_symlink() and p.name not in {".git", "node_modules", "__pycache__"})
        except OSError as exc:
            issues.append(f"{path}: {exc.strerror}")
    if stack:
        issues.append("本轮扫描达到目录上限，请添加更具体的模型目录")
    return found, issues


def filesystem(path: Path):
    ancestor = path
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    info = ancestor.stat()
    usage = shutil.disk_usage(ancestor)
    mount = ancestor
    while mount != mount.parent and mount.parent.stat().st_dev == info.st_dev:
        mount = mount.parent
    return {"device": str(info.st_dev), "path": str(mount),
            "total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free}
