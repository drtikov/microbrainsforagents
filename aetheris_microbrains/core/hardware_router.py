import logging
import os
import platform
import shutil
import subprocess
from typing import Any, Dict

log = logging.getLogger("aetheris_hardware_router")


def get_best_execution_provider(batch_size: int = 1) -> str:
    """Return the best execution provider for ONNX/Tensor workloads (e.g. CPU, CUDA, DirectML)."""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        for p in ["DmlExecutionProvider", "CUDAExecutionProvider", "OpenVINOExecutionProvider"]:
            if p in providers:
                return p
        if "CPUExecutionProvider" in providers:
            return "CPUExecutionProvider"
    except ImportError:
        pass

    return "CPUExecutionProvider"


def recommended_microbrain_dispatch(batch_size: int = 1) -> str:
    """Arbitrate between CPU SIMD (fast single reflex) and OpenCL GPU (parallel colony)."""
    try:
        from .opencl_microbrain_accelerator import get_opencl_accelerator
        acc = get_opencl_accelerator()
        if acc.gpu_available and int(batch_size) >= acc.GPU_BATCH_THRESHOLD:
            return "OpenCL_GPU"
    except Exception:
        pass
    return "CPU"


def get_hardware_report() -> Dict[str, Any]:
    """Inspect local hardware (CPU, NVIDIA CUDA/OpenCL, ONNX providers) and return capabilities report."""
    report: Dict[str, Any] = {
        "os": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "npu_detected": False,
        "npu_type": None,
        "gpu_detected": False,
        "gpu_type": None,
        "gpu_name": None,
        "opencl_available": False,
        "opencl_compute_units": 0,
        "opencl_vram_mb": 0,
        "best_provider": "CPU",
    }

    # Check GPU availability via nvidia-smi if safe and available
    try:
        smi_path = shutil.which("nvidia-smi")
        if smi_path:
            res = subprocess.run(
                [smi_path, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if res.returncode == 0:
                report["gpu_detected"] = True
                report["gpu_type"] = "NVIDIA"
                line = (res.stdout or "").strip().splitlines()
                if line:
                    report["gpu_name"] = line[0].split(",")[0].strip()
                report["cuda_driver_present"] = True
    except Exception:
        pass

    # Check OpenCL GPU co-processor (supports legacy Kepler/Maxwell CUDA cards like GT 710)
    try:
        from .opencl_microbrain_accelerator import get_opencl_accelerator
        acc = get_opencl_accelerator()
        if acc.gpu_available:
            report["gpu_detected"] = True
            report["opencl_available"] = True
            report["gpu_name"] = report["gpu_name"] or acc.device_name
            report["gpu_type"] = f"OpenCL ({acc.platform_name}: {acc.device_name})"
            report["opencl_compute_units"] = acc.compute_units
            report["opencl_vram_mb"] = acc.global_mem_mb
            report["best_provider"] = "OpenCL_GPU"
    except Exception:
        pass

    best_provider = get_best_execution_provider(batch_size=1)
    if best_provider == "DmlExecutionProvider":
        report["gpu_detected"] = True
        report["gpu_type"] = "DirectML Compatible"
        report["best_provider"] = "DirectML"
    elif best_provider == "OpenVINOExecutionProvider":
        report["npu_detected"] = True
        report["npu_type"] = "Intel NPU/CPU (OpenVINO)"
        report["best_provider"] = "OpenVINO"
    elif best_provider == "CUDAExecutionProvider":
        report["gpu_detected"] = True
        report["gpu_type"] = "NVIDIA CUDA"
        report["best_provider"] = "CUDA"

    return report
