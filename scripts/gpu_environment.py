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
    "devices": devices,
}
output = Path(os.getenv("GEOMFLOW_ENV_OUTPUT", "gpu-environment.json"))
output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
print(output)
