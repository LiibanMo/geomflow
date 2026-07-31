"""Write reproducible CUDA runner metadata as JSON."""

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import torch


def command(*args: str) -> str:
    return subprocess.run(args, check=True, text=True, capture_output=True).stdout.strip()


def cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def memory_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        return None


devices = []
for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    devices.append(
        {
            "index": index,
            "name": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability(index)),
            "memory_bytes": properties.total_memory,
        }
    )

try:
    commit = os.environ["GITHUB_SHA"]
except KeyError:
    try:
        commit = command("git", "rev-parse", "HEAD")
    except subprocess.CalledProcessError:
        commit = "unavailable"

metadata = {
    "commit": commit,
    "os": platform.platform(),
    "python": sys.version,
    "pytorch": torch.__version__,
    "pytorch_cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "driver": command("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
    "gpu_uuids": command("nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader").splitlines(),
    "devices": devices,
    "cpu_model": cpu_model(),
    "cpu_count": os.cpu_count(),
    "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
    "memory_bytes": memory_bytes(),
    "torch_num_threads": torch.get_num_threads(),
    "torch_num_interop_threads": torch.get_num_interop_threads(),
    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    "cudnn_benchmark": torch.backends.cudnn.benchmark,
}
output = Path(os.getenv("GEOMFLOW_ENV_OUTPUT", "gpu-environment.json"))
output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
print(output)
