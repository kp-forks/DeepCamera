#!/usr/bin/env python3
"""
Coral TPU Device Probe — enumerates connected Edge TPU devices.

Outputs JSON to stdout for Aegis skill deployment verification.

Usage:
  python scripts/tpu_probe.py
  docker run --device /dev/bus/usb aegis-coral-tpu python3 scripts/tpu_probe.py
"""

import json
import sys


def probe_tpus():
    """Enumerate Edge TPU devices and return info dict."""
    result = {
        "event": "tpu_probe",
        "available": False,
        "count": 0,
        "devices": [],
        "runtime": None,
        "error": None,
    }

    # Check pycoral availability
    try:
        from pycoral.utils.edgetpu import list_edge_tpus
        result["runtime"] = "pycoral"
    except ImportError:
        result["error"] = "pycoral not installed"
        # Try tflite-runtime as fallback
        try:
            import tflite_runtime.interpreter as tflite
            result["runtime"] = "tflite-runtime (no Edge TPU delegate)"
            # Can't enumerate TPUs without pycoral
            result["error"] = "pycoral required for TPU enumeration"
        except ImportError:
            result["runtime"] = None
            result["error"] = "Neither pycoral nor tflite-runtime installed"
        return result

    # Enumerate TPUs
    try:
        tpus = list_edge_tpus()
        result["count"] = len(tpus)
        result["available"] = len(tpus) > 0

        for i, tpu in enumerate(tpus):
            device_info = {
                "index": i,
                "type": tpu.get("type", "unknown") if isinstance(tpu, dict) else str(tpu),
            }
            result["devices"].append(device_info)

    except Exception as e:
        result["error"] = f"Failed to enumerate TPUs: {str(e)}"

    # Check USB devices for additional context
    try:
        import subprocess
        lsusb = subprocess.run(
            ["lsusb"], capture_output=True, text=True, timeout=5
        )
        coral_lines = [
            line.strip() for line in lsusb.stdout.splitlines()
            if "1a6e" in line.lower() or "18d1" in line.lower()  # Global Unichip / Google
            or "coral" in line.lower() or "edge tpu" in line.lower()
        ]
        if coral_lines:
            result["usb_devices"] = coral_lines
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass  # lsusb not available

    return result


def main():
    result = probe_tpus()
    print(json.dumps(result, indent=2))

    # Human-readable summary to stderr
    if result["available"]:
        sys.stderr.write(f"✓ Found {result['count']} Edge TPU device(s)\n")
        for dev in result["devices"]:
            sys.stderr.write(f"  [{dev['index']}] {dev['type']}\n")
    else:
        sys.stderr.write(f"✗ No Edge TPU detected\n")
        if result["error"]:
            sys.stderr.write(f"  Error: {result['error']}\n")

    # Exit code: 0 if TPU found, 1 if not
    sys.exit(0 if result["available"] else 1)


if __name__ == "__main__":
    main()
