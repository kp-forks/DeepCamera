"""
env_config.py — Shared hardware environment detection and model optimization.

Provides a single entry point for any DeepCamera skill to:
  1. Detect available compute hardware (NVIDIA, AMD, Apple, Intel, CPU)
  2. Auto-export models to the optimal inference format
  3. Load cached optimized models with PyTorch fallback

Usage:
    from lib.env_config import HardwareEnv

    env = HardwareEnv.detect()
    model, fmt = env.load_optimized("yolo26n")
"""

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _log(msg: str):
    """Log to stderr."""
    print(f"[env_config] {msg}", file=sys.stderr, flush=True)


# ─── Backend definitions ────────────────────────────────────────────────────

@dataclass
class BackendSpec:
    """Specification for a compute backend's optimized export."""
    name: str               # "cuda", "rocm", "mps", "intel", "cpu"
    export_format: str      # ultralytics export format string
    model_suffix: str       # file extension/dir to look for cached model
    half: bool = True       # use FP16
    extra_export_args: dict = field(default_factory=dict)


BACKEND_SPECS = {
    "cuda": BackendSpec(
        name="cuda",
        export_format="engine",
        model_suffix=".engine",
        half=True,
    ),
    "rocm": BackendSpec(
        name="rocm",
        export_format="onnx",
        model_suffix=".onnx",
        half=False,  # ONNX Runtime ROCm handles precision internally
    ),
    "mps": BackendSpec(
        name="mps",
        export_format="coreml",
        model_suffix=".mlpackage",
        half=True,
        extra_export_args={"nms": False},
    ),
    "intel": BackendSpec(
        name="intel",
        export_format="openvino",
        model_suffix="_openvino_model",
        half=True,
    ),
    "cpu": BackendSpec(
        name="cpu",
        export_format="onnx",
        model_suffix=".onnx",
        half=False,
    ),
}


# ─── Hardware detection ──────────────────────────────────────────────────────

@dataclass
class HardwareEnv:
    """Detected hardware environment with model optimization capabilities."""

    backend: str = "cpu"              # "cuda" | "rocm" | "mps" | "intel" | "cpu"
    device: str = "cpu"               # torch device string
    export_format: str = "onnx"       # optimal export format
    gpu_name: str = ""                # human-readable GPU name
    gpu_memory_mb: int = 0            # GPU memory in MB
    driver_version: str = ""          # GPU driver version
    framework_ok: bool = False        # True if optimized runtime is importable
    detection_details: dict = field(default_factory=dict)  # raw detection info

    # Timing (populated by export/load)
    export_ms: float = 0.0
    load_ms: float = 0.0

    @staticmethod
    def detect() -> "HardwareEnv":
        """Probe the system and return a populated HardwareEnv."""
        env = HardwareEnv()

        # Try each backend in priority order
        if env._try_cuda():
            pass
        elif env._try_rocm():
            pass
        elif env._try_mps():
            pass
        elif env._try_intel():
            pass
        else:
            env._fallback_cpu()

        # Set export format from backend spec
        spec = BACKEND_SPECS.get(env.backend, BACKEND_SPECS["cpu"])
        env.export_format = spec.export_format

        # Check if optimized runtime is available
        env.framework_ok = env._check_framework()

        _log(f"Detected: backend={env.backend}, device={env.device}, "
             f"gpu={env.gpu_name or 'none'}, "
             f"format={env.export_format}, "
             f"framework_ok={env.framework_ok}")

        return env

    def _try_cuda(self) -> bool:
        """Detect NVIDIA GPU via nvidia-smi and torch."""
        if not shutil.which("nvidia-smi"):
            return False
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return False

            line = result.stdout.strip().split("\n")[0]
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                self.backend = "cuda"
                self.device = "cuda"
                self.gpu_name = parts[0]
                self.gpu_memory_mb = int(float(parts[1]))
                self.driver_version = parts[2]
                self.detection_details["nvidia_smi"] = line
                _log(f"NVIDIA GPU: {self.gpu_name} ({self.gpu_memory_mb}MB, driver {self.driver_version})")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError) as e:
            _log(f"nvidia-smi probe failed: {e}")
        return False

    def _try_rocm(self) -> bool:
        """Detect AMD GPU via amd-smi (preferred) or rocm-smi."""
        has_amd_smi = shutil.which("amd-smi") is not None
        has_rocm_smi = shutil.which("rocm-smi") is not None
        has_rocm_dir = Path("/opt/rocm").is_dir()

        if not (has_amd_smi or has_rocm_smi or has_rocm_dir):
            return False

        self.backend = "rocm"
        self.device = "cuda"  # ROCm exposes as CUDA in PyTorch

        # Strategy 1: amd-smi static --json (ROCm 6.3+/7.x, richest output)
        if has_amd_smi:
            try:
                result = subprocess.run(
                    ["amd-smi", "static", "--json"],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0:
                    import json as _json
                    data = _json.loads(result.stdout)
                    # amd-smi may return {"gpu_data": [...]} or a bare list
                    gpu_list = data.get("gpu_data", data) if isinstance(data, dict) else data
                    if isinstance(gpu_list, list) and len(gpu_list) > 0:
                        # Pick GPU with most VRAM (discrete > iGPU)
                        def _vram_mb(g):
                            vram = g.get("vram", {}).get("size", {})
                            if isinstance(vram, dict):
                                return int(vram.get("value", 0))
                            return 0

                        best_gpu = max(gpu_list, key=_vram_mb)
                        best_idx = gpu_list.index(best_gpu)
                        asic = best_gpu.get("asic", {})
                        vram = best_gpu.get("vram", {}).get("size", {})

                        self.gpu_name = asic.get("market_name", "AMD GPU")
                        self.gpu_memory_mb = int(vram.get("value", 0)) if isinstance(vram, dict) else 0
                        self.detection_details["amd_smi"] = {
                            "gpu_index": best_idx,
                            "gfx_version": asic.get("target_graphics_version", ""),
                            "total_gpus": len(gpu_list),
                        }

                        # Pin to discrete GPU if multiple GPUs present
                        if len(gpu_list) > 1:
                            os.environ["HIP_VISIBLE_DEVICES"] = str(best_idx)
                            os.environ["ROCR_VISIBLE_DEVICES"] = str(best_idx)
                            _log(f"Multi-GPU: pinned to GPU {best_idx} ({self.gpu_name})")
            except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, Exception) as e:
                _log(f"amd-smi probe failed: {e}")

        # Strategy 2: rocm-smi fallback (legacy ROCm <6.3)
        if not self.gpu_name and has_rocm_smi:
            try:
                result = subprocess.run(
                    ["rocm-smi", "--showproductname", "--csv"],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split("\n")
                    if len(lines) > 1:
                        self.gpu_name = lines[1].split(",")[0].strip()
                self.detection_details["rocm_smi"] = result.stdout.strip()
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

            try:
                result = subprocess.run(
                    ["rocm-smi", "--showmeminfo", "vram", "--csv"],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().split("\n")[1:]:
                        parts = line.split(",")
                        if len(parts) >= 2:
                            try:
                                self.gpu_memory_mb = int(float(parts[0].strip()) / (1024 * 1024))
                            except ValueError:
                                pass
                            break
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

        _log(f"AMD ROCm GPU: {self.gpu_name or 'detected'} ({self.gpu_memory_mb}MB)")
        return True

    def _try_mps(self) -> bool:
        """Detect Apple Silicon via uname + sysctl."""
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            return False

        self.backend = "mps"
        self.device = "mps"

        # Get chip name
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                self.gpu_name = result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.gpu_name = "Apple Silicon"

        # Get total memory (shared with GPU on Apple Silicon)
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                self.gpu_memory_mb = int(int(result.stdout.strip()) / (1024 * 1024))
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass

        _log(f"Apple Silicon: {self.gpu_name} ({self.gpu_memory_mb}MB unified)")
        return True

    def _try_intel(self) -> bool:
        """Detect Intel OpenVINO-capable hardware."""
        # Check for OpenVINO installation
        has_openvino = False
        try:
            import openvino  # noqa: F401
            has_openvino = True
        except ImportError:
            # Check for system install
            has_openvino = Path("/opt/intel/openvino").is_dir()

        if not has_openvino:
            # Check CPU flags for Intel-specific features (AVX-512, AMX)
            try:
                if platform.system() == "Linux":
                    with open("/proc/cpuinfo") as f:
                        cpuinfo = f.read()
                    if "GenuineIntel" in cpuinfo:
                        self.backend = "intel"
                        self.device = "cpu"
                        self.gpu_name = "Intel CPU"
                        _log("Intel CPU detected (no OpenVINO installed)")
                        return True
            except FileNotFoundError:
                pass
            return False

        self.backend = "intel"
        self.device = "cpu"  # OpenVINO handles device selection internally
        self.gpu_name = "Intel (OpenVINO)"

        # Check for Intel GPU / NPU
        try:
            from openvino.runtime import Core
            core = Core()
            devices = core.available_devices
            self.detection_details["openvino_devices"] = devices
            if "GPU" in devices:
                self.gpu_name = "Intel GPU (OpenVINO)"
            if "NPU" in devices:
                self.gpu_name = "Intel NPU (OpenVINO)"
            _log(f"OpenVINO devices: {devices}")
        except Exception:
            pass

        _log(f"Intel: {self.gpu_name}")
        return True

    def _fallback_cpu(self):
        """CPU-only fallback."""
        self.backend = "cpu"
        self.device = "cpu"
        self.gpu_name = ""

        # Report CPU info
        try:
            self.detection_details["cpu"] = platform.processor() or "unknown"
        except Exception:
            pass

        _log("No GPU detected, using CPU backend")

    def _check_rocm_runtime(self):
        """Verify onnxruntime has ROCm provider, not just CPU."""
        import onnxruntime
        providers = onnxruntime.get_available_providers()
        if "ROCmExecutionProvider" in providers or "MIGraphXExecutionProvider" in providers:
            _log(f"onnxruntime ROCm providers: {providers}")
            return True
        _log(f"onnxruntime providers: {providers} — ROCmExecutionProvider not found")
        _log("Fix: pip uninstall onnxruntime && pip install onnxruntime-rocm")
        raise ImportError("ROCmExecutionProvider not available")

    def _check_framework(self) -> bool:
        """Check if the optimized inference runtime is importable."""
        checks = {
            "cuda": lambda: __import__("tensorrt"),
            "rocm": lambda: self._check_rocm_runtime(),
            "mps": lambda: __import__("coremltools"),
            "intel": lambda: __import__("openvino"),
            "cpu": lambda: __import__("onnxruntime"),
        }

        check = checks.get(self.backend)
        if not check:
            return False
        try:
            check()
            return True
        except ImportError:
            _log(f"Optimized runtime not installed for {self.backend}, "
                 f"will use PyTorch fallback")
            return False

    # ─── Model export & loading ──────────────────────────────────────────

    def get_optimized_path(self, model_name: str) -> Path:
        """Get the expected path for the optimized model."""
        spec = BACKEND_SPECS.get(self.backend, BACKEND_SPECS["cpu"])
        return Path(f"{model_name}{spec.model_suffix}")

    def export_model(self, model, model_name: str) -> Optional[Path]:
        """Export PyTorch model to optimal format. Returns path or None."""
        if not self.framework_ok:
            _log(f"Skipping export — {self.backend} runtime not available")
            return None

        spec = BACKEND_SPECS.get(self.backend, BACKEND_SPECS["cpu"])
        optimized_path = self.get_optimized_path(model_name)

        # Already exported
        if optimized_path.exists():
            _log(f"Cached model found: {optimized_path}")
            return optimized_path

        # Guard: numpy 2.x breaks coremltools PyTorch→MIL converter
        # (TypeError: only 0-dimensional arrays can be converted to Python scalars)
        if spec.export_format == "coreml":
            try:
                import numpy as np
                np_major = int(np.__version__.split('.')[0])
                if np_major >= 2:
                    _log(f"numpy {np.__version__} detected — CoreML export "
                         f"requires numpy<2.0.0 (coremltools incompatibility)")
                    _log("Fix: pip install 'numpy>=1.24,<2.0'")
                    return None
            except Exception:
                pass  # If numpy check fails, try export anyway

        try:
            _log(f"Exporting {model_name}.pt → {spec.export_format} "
                 f"(one-time, may take 30-120s)...")
            t0 = time.perf_counter()

            export_kwargs = {
                "format": spec.export_format,
                "half": spec.half,
            }
            export_kwargs.update(spec.extra_export_args)

            exported = model.export(**export_kwargs)
            self.export_ms = (time.perf_counter() - t0) * 1000

            exported_path = Path(exported)
            if exported_path.exists():
                _log(f"Export complete: {exported_path} ({self.export_ms:.0f}ms)")
                return exported_path

            _log(f"Export returned {exported} but path not found")
        except Exception as e:
            _log(f"Export failed ({spec.export_format}): {e}")

        return None

    def load_optimized(self, model_name: str, use_optimized: bool = True):
        """
        Load the best available model for this hardware.

        Returns:
            (model, format_str) — the YOLO model and its format name
        """
        from ultralytics import YOLO

        t0 = time.perf_counter()

        if use_optimized and self.framework_ok:
            # Try loading from cache first (no export needed)
            optimized_path = self.get_optimized_path(model_name)
            if optimized_path.exists():
                try:
                    model = YOLO(str(optimized_path))
                    self.load_ms = (time.perf_counter() - t0) * 1000
                    _log(f"Loaded {self.export_format} model ({self.load_ms:.0f}ms)")
                    return model, self.export_format
                except Exception as e:
                    _log(f"Failed to load cached model: {e}")

            # Try exporting then loading
            pt_model = YOLO(f"{model_name}.pt")
            exported = self.export_model(pt_model, model_name)
            if exported:
                try:
                    model = YOLO(str(exported))
                    self.load_ms = (time.perf_counter() - t0) * 1000
                    _log(f"Loaded freshly exported {self.export_format} model ({self.load_ms:.0f}ms)")
                    return model, self.export_format
                except Exception as e:
                    _log(f"Failed to load exported model: {e}")

            # Fallback: use the PT model we already loaded
            _log("Falling back to PyTorch model")
            pt_model.to(self.device)
            self.load_ms = (time.perf_counter() - t0) * 1000
            return pt_model, "pytorch"

        # No optimization requested or framework missing
        model = YOLO(f"{model_name}.pt")
        model.to(self.device)
        self.load_ms = (time.perf_counter() - t0) * 1000
        return model, "pytorch"

    def to_dict(self) -> dict:
        """Serialize environment info for JSON output."""
        return {
            "backend": self.backend,
            "device": self.device,
            "export_format": self.export_format,
            "gpu_name": self.gpu_name,
            "gpu_memory_mb": self.gpu_memory_mb,
            "driver_version": self.driver_version,
            "framework_ok": self.framework_ok,
            "export_ms": round(self.export_ms, 1),
            "load_ms": round(self.load_ms, 1),
        }


# ─── CLI: run standalone for diagnostics ─────────────────────────────────────

if __name__ == "__main__":
    env = HardwareEnv.detect()
    print(json.dumps(env.to_dict(), indent=2))
