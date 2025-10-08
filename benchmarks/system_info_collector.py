import contextlib
import json
import platform
import shutil
import subprocess
from typing import Any, Dict

import cpuinfo
import distro
import psutil


def _collect_gpus() -> list[dict]:
    gpus = []
    # 2) NVIDIA via nvidia-smi
    if shutil.which("nvidia-smi"):
        with contextlib.suppress(Exception):
            q = "name,memory.total,driver_version,uuid"
            out = subprocess.check_output(
                ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                text=True,
            )
            for i, line in enumerate(
                gpu_info_line.strip()
                for gpu_info_line in out.splitlines()
                if gpu_info_line.strip()
            ):
                name, mem, driver, uuid = (x.strip() for x in line.split(","))
                gpus.append(
                    {
                        "vendor": "NVIDIA",
                        "index": i,
                        "name": name,
                        "memory_total_mb": int(mem),
                        "driver": driver,
                        "uuid": uuid,
                    }
                )
            if gpus:
                return gpus

    # 3) AMD via rocm-smi (optional)
    if shutil.which("rocm-smi"):
        with contextlib.suppress(Exception):
            # Newer rocm-smi supports --json
            out = subprocess.check_output(
                ["rocm-smi", "--showproductname", "--showvram", "--json"], text=True
            )
            data = json.loads(out)
            for i, dev in enumerate(data.get("card", [])):
                name = dev.get("Card series") or dev.get("Card model")
                vram = dev.get("VRAM Total Memory (B)")
                mem_mb = int(vram) // (1024 * 1024) if vram else None
                gpus.append(
                    {
                        "vendor": "AMD",
                        "index": i,
                        "name": name,
                        "memory_total_mb": mem_mb,
                        "driver": None,
                        "uuid": None,
                    }
                )
            if gpus:
                return gpus

    return gpus


def collect_system_info() -> Dict[str, Any]:
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

    # GPUs
    gpus = _collect_gpus()
    info["gpus"] = gpus

    return info


def _fmt_num(x, nd=2):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:  # noqa: BLE001
        return str(x)


def _os_string(osd: dict) -> str:
    """Compact OS string without assuming Linux."""
    system = osd.get("system") or "UnknownOS"
    release = osd.get("release") or ""
    distro_name = osd.get("distro")
    machine = osd.get("machine") or ""
    # Prefer pretty distro on Linux, otherwise use system name
    base = distro_name or system
    parts = [base]
    if release:
        parts.append(release)
    if machine:
        parts.append(machine)
    return " ".join(parts)


def _gpu_briefs(gpus: list[dict]) -> list[str]:
    briefs = []
    for g in gpus or []:
        name = g.get("name") or "GPU"
        mem_mb = g.get("memory_total_mb")
        mem_gb = f"{mem_mb/1024:.1f} GB" if isinstance(mem_mb, (int, float)) else "?"
        briefs.append(f"{name} ({mem_gb})")
    return briefs


def format_system_info_markdown(info: dict) -> str:
    osd = info.get("os", {})
    pyd = info.get("python", {})
    cpu = info.get("cpu", {})
    gpus = info.get("gpus", []) or []
    mem = info.get("memory_gb")

    py_impl = pyd.get("implementation")
    py_ver = pyd.get("version")

    lines = []
    # Summary bullets
    lines.append("#### System Info")
    lines.append(f"- **OS**: {_os_string(osd)}")
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


def format_system_info_markdown_lite(info: dict) -> str:
    """Very short markdown context for benchmark sections."""
    os_str = _os_string(info.get("os", {}))
    py = info.get("python", {})
    cpu = info.get("cpu", {})
    mem = info.get("memory_gb")
    gpus = info.get("gpus", [])

    py_str = f"{py.get('implementation','Python')} {py.get('version','?')}"
    cpu_model = cpu.get("model") or "Unknown CPU"
    lcores = cpu.get("logical_cores")
    core_str = f" ({lcores} threads)" if lcores is not None else ""
    mem_str = f"{mem:.1f} GB" if isinstance(mem, (int, float)) else "?"

    gpu_list = _gpu_briefs(gpus)
    gpu_str = ", ".join(gpu_list) if gpu_list else "none"

    lines = []
    lines.append("## System Info")
    lines.append(f"- **OS**: {os_str}")
    lines.append(f"- **Python**: {py_str}")
    lines.append(f"- **CPU**: {cpu_model}{core_str}")
    lines.append(f"- **Memory**: {mem_str}")
    lines.append(f"- **GPU**: {gpu_str}")
    return "\n".join(lines)


def format_system_info_oneliner(info: dict) -> str:
    """Single line summary suitable for inline captions."""
    os_str = _os_string(info.get("os", {}))
    py = info.get("python", {})
    cpu = info.get("cpu", {})
    mem = info.get("memory_gb")
    gpus = info.get("gpus", [])

    py_str = f"{py.get('implementation','Python')} {py.get('version','?')}"
    cpu_model = cpu.get("model") or "Unknown CPU"
    lcores = cpu.get("logical_cores")
    mem_str = f"{mem:.1f} GB" if isinstance(mem, (int, float)) else "?"

    gpu_list = _gpu_briefs(gpus)
    gpu_str = "; ".join(gpu_list) if gpu_list else "no GPU"

    core_str = f", {lcores} threads" if lcores is not None else ""
    return (
        f"{os_str} | {py_str} | {cpu_model}{core_str} | RAM {mem_str} | GPU {gpu_str}"
    )


# Example pretty print
if __name__ == "__main__":
    print(json.dumps(collect_system_info(), indent=2))
    print(format_system_info_markdown(collect_system_info()))
    print(format_system_info_markdown_lite(collect_system_info()))
    print(format_system_info_oneliner(collect_system_info()))
    print()
