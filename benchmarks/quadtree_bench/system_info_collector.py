import contextlib
import json
import platform
from typing import Any, Dict


def safe_import(name: str):
    with contextlib.suppress(ImportError):
        return __import__(name)


def collect_system_info() -> Dict[str, Any]:
    psutil = safe_import("psutil")
    cpuinfo = safe_import("cpuinfo")
    distro = safe_import("distro")
    gputil = safe_import("GPUtil")

    info = {
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "cpu": {},
        "memory_gb": None,
        "gpus": [],
    }

    # CPU and RAM via psutil
    if psutil:
        with contextlib.suppress(Exception):
            info["cpu"]["physical_cores"] = psutil.cpu_count(logical=False)
            info["cpu"]["logical_cores"] = psutil.cpu_count(logical=True)
            freq = psutil.cpu_freq()
            if freq:
                info["cpu"]["max_freq_mhz"] = getattr(freq, "max", None)
                info["cpu"]["current_freq_mhz"] = getattr(freq, "current", None)
            info["memory_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)

    # CPU model via py-cpuinfo
    if cpuinfo:
        with contextlib.suppress(Exception):
            ci = cpuinfo.get_cpu_info()
            info["cpu"]["model"] = ci.get("brand_raw") or ci.get("brand")
            if not info["cpu"].get("max_freq_mhz"):
                hz = ci.get("hz_advertised_friendly") or ci.get("hz_advertised")
                info["cpu"]["advertised_freq"] = hz

    # Linux distro pretty name
    if distro and info["os"]["system"] == "Linux":
        with contextlib.suppress(Exception):
            info["os"]["distro"] = distro.name(pretty=True)

    # GPUs via GPUtil
    if gputil:
        with contextlib.suppress(Exception):
            for gpu in gputil.getGPUs():
                info["gpus"].append(
                    {
                        "name": gpu.name,
                        "memory_total_mb": int(gpu.memoryTotal),
                        "driver": getattr(gpu, "driver", None),
                        "uuid": getattr(gpu, "uuid", None),
                    }
                )

    return info


def _fmt_num(x, nd=2):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:  # noqa: BLE001
        return str(x)


def format_system_info_markdown(info: dict) -> str:
    osd = info.get("os", {})
    pyd = info.get("python", {})
    cpu = info.get("cpu", {})
    gpus = info.get("gpus", []) or []
    mem = info.get("memory_gb")

    os_name = osd.get("distro") or osd.get("system")
    kernel = osd.get("release")
    machine = osd.get("machine")
    py_impl = pyd.get("implementation")
    py_ver = pyd.get("version")

    lines = []
    # Summary bullets
    lines.append("#### System Info")
    lines.append(f"- **OS**: {os_name} (Linux {kernel}) on {machine}")
    lines.append(f"- **Python**: {py_impl} {py_ver}")
    model = cpu.get("model") or "Unknown CPU"
    pcores = cpu.get("physical_cores")
    lcores = cpu.get("logical_cores")
    maxmhz = cpu.get("max_freq_mhz")
    curmhz = cpu.get("current_freq_mhz")
    lines.append(f"- **CPU**: {model}")
    parts = []
    if pcores is not None:
        parts.append(f"physical cores: {pcores}")
    if lcores is not None:
        parts.append(f"logical cores: {lcores}")
    if maxmhz is not None:
        parts.append(f"max freq: {_fmt_num(maxmhz)} MHz")
    if curmhz is not None:
        parts.append(f"current freq: {_fmt_num(curmhz)} MHz")
    if parts:
        lines.append("  • " + "  \n  • ".join(parts))
    if mem is not None:
        lines.append(f"- **Memory**: {_fmt_num(mem, 1)} GB")

    # GPU section
    if gpus:
        lines.append("- **GPUs**:")
        lines.append("")
        lines.append("| # | Name | Memory (MB) | Driver | UUID |")
        lines.append("|-:|------|------------:|:------|:-----|")
        for i, g in enumerate(gpus):
            lines.append(
                f"| {i} | {g.get('name','')} | {g.get('memory_total_mb','')} | "
                f"{g.get('driver','')} | {g.get('uuid','')} |"
            )
    else:
        lines.append("- **GPUs**: none detected")

    return "\n".join(lines)


# Example pretty print
if __name__ == "__main__":
    print(json.dumps(collect_system_info(), indent=2))
    print(format_system_info_markdown(collect_system_info()))
