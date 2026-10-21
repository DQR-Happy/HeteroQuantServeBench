"""Compile the private Neutrino memory-event probe for one selected PTX entry.

Executed only inside the isolated CUDA runtime's compiler subprocess. Original
instructions execute once; each enabled thread records a bounded event sequence.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import shutil
import importlib.util


def compile_memory(workdir, name, pruned, functions):
    from hqsb.observation.dmat_instrument import instrument_memory_events

    root = Path(workdir)
    if functions.strip():
        raise ValueError("DMAT_CALLED_FUNCTIONS_REQUIRE_INSTRUMENTATION")
    # A selected entry's explicit global loads/stores are supported. Other
    # global/generic memory operations require their own operand semantics.
    code = re.sub(r"/\*.*?\*/|//[^\n]*", "", pruned, flags=re.S)
    unsupported = re.findall(r"\b(?:cp\.async[\w.]*|(?:atom|red)[\w.]*|(?:ld|st)\.(?:global|generic)[\w.]*|(?:ld|st)\.[busf]\d+)\b", code)
    unsupported = sorted({op for op in unsupported if not re.match(r"^(?:ld|st)\.global(?:\.|$)|^cp\.async\.(?:ca|cg)\.shared\.global(?:\.|$)|^cp\.async\.(?:commit_group|wait_group|wait_all)$", op)})
    if unsupported:
        raise ValueError("DMAT_MEMORY_INSTRUCTIONS_REQUIRE_ADAPTER:" + ",".join(unsupported))
    capacity = int(os.environ["HQSB_DMAT_CAPACITY"])
    entry = pruned.index(".entry")
    param_start = pruned.index("(", entry)
    param_end = pruned.index(")", param_start)
    # PTX parameter declarations need not occupy separate lines. Each .param
    # is one kernelParams element, including aligned byte-array parameters.
    params = re.findall(r"\.param\b", pruned[param_start:param_end])
    probed, sites = instrument_memory_events(pruned, capacity)
    # GPU-wide timer gives comparable timestamps across SMs. Its granularity is
    # device-dependent; ns units do not imply 1 ns measurement resolution.
    (root / "pruned.ptx").write_text(pruned)
    (root / "probed.ptx").write_text(probed)
    assemblers = [shutil.which("ptxas") or "ptxas"]
    triton = importlib.util.find_spec("triton")
    if triton and triton.origin:
        bundled = Path(triton.origin).parent / "backends/nvidia/bin/ptxas"
        if bundled.is_file():
            assemblers.append(str(bundled))
    assembler = assemblers[0]
    for stem in ("pruned", "probed"):
        choices = assemblers if stem == "pruned" else [assembler]
        for assembler in choices:
            result = subprocess.run([assembler, "-arch=sm_" + os.environ["HQSB_ERE_ARCH"], "-m64",
                                 str(root / (stem + ".ptx")), "-o", str(root / (stem + ".bin"))],
                                text=True, capture_output=True)
            with (root / (stem + "-ptxas.log")).open("a") as log:
                log.write(assembler + "\n" + result.stdout + result.stderr)
            if result.returncode == 0 or "Unsupported .version" not in result.stderr:
                break
        result.check_returncode()
    (root / "kernel.info").write_text(f"{name}\n{len(params)}\n2\n0,{capacity * 48}\n0,8\n")
    for site in sites:
        site["operation"] = "read" if site["opcode"].startswith(("ld.", "cp.async")) else "write"
    (root / "memory-sites.json").write_text(json.dumps({
        "schema_version": "hqsb-dmat-probe-2", "event_bytes": 48, "kernel": name, "sites": sites,
        "capacity": capacity, "clock_domain": "GPU_globaltimer", "time_unit": "ns",
        "clock_domains": ["thread_entry_clock64", "GPU_globaltimer"],
        "normalized_clock_origin": "thread_entry_before_probe_buffer_calculation",
        "sample_semantics": "every_enabled_thread_at_supported_global_instruction_before_execution",
        "ptx_sha256": hashlib.sha256(pruned.encode()).hexdigest(),
        "probed_ptx_sha256": hashlib.sha256(probed.encode()).hexdigest(),
        "assembler": assembler,
        "assembler_version": subprocess.check_output([assembler, "--version"], text=True),
    }, indent=2) + "\n")
