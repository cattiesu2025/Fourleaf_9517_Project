"""Runtime helpers shared by scratch and transfer-learning pipelines."""

from __future__ import annotations

import os
import platform
import random
import socket
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    """Resolve an explicit or automatic PyTorch device selection."""

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return device


def unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor, Any]:
    """Normalize dict-style and tuple-style dataloader batches."""

    if isinstance(batch, dict):
        return batch["image"], batch["class_idx"], batch["image_id"]
    return batch


def hardware_info(device: torch.device) -> dict[str, Any]:
    """Collect the hardware fields used by every neural-model runtime file."""

    gpu_name = None
    gpu_memory_gb = None
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        gpu_memory_gb = round(
            torch.cuda.get_device_properties(device).total_memory / (1024**3),
            2,
        )
    elif device.type == "mps":
        gpu_name = "Apple MPS"

    ram_gb = None
    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            ram_gb = round(pages * page_size / (1024**3), 2)
        except (ValueError, OSError):
            pass

    return {
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "pbs_jobid": os.environ.get("PBS_JOBID"),
        "cpu": platform.processor() or platform.machine(),
        "gpu": gpu_name,
        "gpu_memory_gb": gpu_memory_gb,
        "ram_gb": ram_gb,
    }


def software_info() -> dict[str, str]:
    """Collect the shared neural-model software versions."""

    import torchvision

    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "torchvision": torchvision.__version__,
    }
