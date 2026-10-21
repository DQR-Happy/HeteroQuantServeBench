"""Locate native NVIDIA CLIs and report installation independently of permissions."""
from pathlib import Path
import shutil
import subprocess


def resolve_executable(tool, configured=None):
    if tool not in {"nsys", "ncu"}:
        raise ValueError("TOOL_UNSUPPORTED")
    path = configured or shutil.which(tool)
    if not path:
        folder = "nsight-compute" if tool == "ncu" else "nsight-systems*"
        path = next((str(p) for p in sorted(Path("/opt/nvidia").glob(folder + "/**/" + tool), reverse=True) if p.is_file()), None)
    return path


def discover(tool, configured=None):
    path = resolve_executable(tool, configured)
    result = {"tool": tool, "installed": False, "executable": path,
              "collection_permission": "checked_when_collecting"}
    if path:
        try:
            probe = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=8, check=True)
            result.update(installed=True, version=(probe.stdout + probe.stderr).strip()[:2000])
        except (OSError, subprocess.SubprocessError) as exc:
            result["reason"] = str(exc)[:500]
    else:
        result["reason"] = "请在执行节点安装工具，并在服务配置 diagnostic_tools 中设置路径。"
    return result
