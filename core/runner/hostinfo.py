"""Collect host and toolchain facts for run-manifest.json. Nothing here identifies the person or machine name."""
import platform
import re
import shutil
import subprocess
import sys


def _first_line(command):
    try:
        out = subprocess.run([str(c) for c in command], capture_output=True, text=True, timeout=30)
        text = (out.stdout or out.stderr).strip()
        return text.splitlines()[0] if text else None
    except (OSError, subprocess.SubprocessError):
        return None


def _command_text(command):
    try:
        out = subprocess.run([str(c) for c in command], capture_output=True, text=True, timeout=30)
        return (out.stdout or out.stderr).strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _windows_cpu_and_memory():
    import ctypes
    import winreg

    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
        cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()

    class Status(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    status = Status()
    status.dwLength = ctypes.sizeof(Status)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return cpu, int(status.ullTotalPhys)


def _mac_cpu_and_memory():
    cpu = _first_line(["sysctl", "-n", "machdep.cpu.brand_string"])
    memory = _first_line(["sysctl", "-n", "hw.memsize"])
    return cpu, int(memory) if memory else None


def _linux_cpu_and_memory():
    cpu = memory = None
    try:
        for line in open("/proc/cpuinfo", encoding="utf-8"):
            if line.lower().startswith(("model name", "hardware", "cpu model")):
                cpu = line.split(":", 1)[1].strip()
                break
        for line in open("/proc/meminfo", encoding="utf-8"):
            if line.startswith("MemTotal"):
                memory = int(re.findall(r"\d+", line)[0]) * 1024
                break
    except OSError:
        pass
    return cpu, memory


def collect(os_label, compilers, cuda_compiler=None):
    """compilers: {"c": path or None, "cxx": path or None}; CUDA details are best-effort."""
    reader = {"win32": _windows_cpu_and_memory, "darwin": _mac_cpu_and_memory}.get(sys.platform, _linux_cpu_and_memory)
    try:
        cpu, memory = reader()
    except Exception:  # host details are best-effort; the run must not fail on them
        cpu, memory = None, None
    import os
    from common import tool

    def used(name):
        try:
            return tool(name)
        except RuntimeError:
            return shutil.which(name) or name
    result = {
        "os_label": os_label,
        "os": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu or platform.processor() or "unknown",
        "logical_cpus": os.cpu_count(),
        "memory_bytes": memory,
        "python": platform.python_version(),
        "tools": {
            "git": _first_line(["git", "--version"]),
            "cmake": _first_line([used("cmake"), "--version"]),
            "ninja": _first_line([used("ninja"), "--version"]),
            "graphviz_dot": _first_line(["dot", "-V"]),
            "c_compiler": compilers.get("c") and {"path": str(compilers["c"]), "version": _first_line([compilers["c"], "--version"])},
            "cxx_compiler": compilers.get("cxx") and {"path": str(compilers["cxx"]), "version": _first_line([compilers["cxx"], "--version"])},
        },
    }
    if cuda_compiler:
        nvcc = _command_text([cuda_compiler, "--version"])
        release = next((line.strip() for line in (nvcc or "").splitlines() if "release " in line), None)
        result["tools"]["cuda_compiler"] = {"path": str(cuda_compiler), "version": release}
        gpu = _first_line(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
                           "--format=csv,noheader,nounits"])
        if gpu:
            fields = [field.strip() for field in gpu.split(",")]
            if len(fields) == 4:
                result["cuda_device"] = {"name": fields[0], "driver_version": fields[1],
                                         "memory_total_mib": float(fields[2]), "compute_capability": fields[3]}
    return result
