"""Fortran TopoSRT engine: write inputs, run the executable, parse outputs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from ..params import SRTParams

ROOT = Path(__file__).resolve().parents[2]
MINGW_BIN = Path("D:/tools/msys64/mingw64/bin")
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


class CancelledError(RuntimeError):
    pass


def fortran_enabled() -> bool:
    """Public deployments can disable the Windows Fortran binary."""
    return os.environ.get("SRT_ALLOW_FORTRAN", "1").lower() not in {"0", "false", "no", "off"}


def resource_path(relative: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", ROOT))
    return base / relative


def find_exe() -> Path:
    if not fortran_enabled():
        raise FileNotFoundError("Fortran 引擎已禁用（SRT_ALLOW_FORTRAN=0）")
    env_exe = os.environ.get("SRT_FORTRAN_EXE")
    if env_exe:
        p = Path(env_exe)
        if p.exists():
            return p
    candidates = [
        resource_path("SRT_topo/exe_SRT_topo_static.exe"),
        ROOT / "SRT_topo" / "exe_SRT_topo_static.exe",
        resource_path("SRT_topo/exe_SRT_topo.exe"),
        ROOT / "SRT_topo" / "exe_SRT_topo.exe",
        resource_path("SRT_topo/exe_SRT_topo_static"),
        ROOT / "SRT_topo" / "exe_SRT_topo_static",
        resource_path("SRT/exe_S_SRT.exe"),
        ROOT / "SRT" / "exe_S_SRT.exe",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "找不到 SRT 可执行文件；请编译 SRT_topo，或设置 SRT_FORTRAN_EXE 指向可执行文件。"
    )


def exe_hash(exe: Path | None = None) -> str:
    exe = exe or find_exe()
    h = hashlib.sha256()
    with open(exe, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _env():
    env = os.environ.copy()
    if MINGW_BIN.exists():
        env["PATH"] = str(MINGW_BIN) + os.pathsep + env.get("PATH", "")
    return env


def _progress_from_line(line: str) -> float | None:
    m = re.search(r"ROUND:\s*(\d)", line)
    if m:
        r = int(m.group(1))
        return {1: 2.0, 2: 32.0, 3: 62.0}.get(r, 0.0)
    m = re.search(r"Step\s+(\d)", line)
    if m:
        step = int(m.group(1))
        round_base = getattr(_progress_from_line, "round_base", 0.0)
        return round_base + min(step, 7) * 4.0
    return None


def _run_process(exe: Path, case_dir: Path, job, manager, progress_base: float, progress_span: float):
    cmd = [str(exe)]
    manager.log(job, f"$ {' '.join(cmd)}  (cwd={case_dir})")
    proc = subprocess.Popen(
        cmd,
        cwd=case_dir,
        env=_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=CREATE_NO_WINDOW,
    )
    manager.set_process(job, proc)
    round_base = 0.0
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if job.cancel_event.is_set():
                proc.terminate()
                raise CancelledError("job cancelled")
            line = line.rstrip()
            if line:
                manager.log(job, line)
            m = re.search(r"ROUND:\s*(\d)", line)
            if m:
                round_base = {1: 0.0, 2: 30.0, 3: 60.0}.get(int(m.group(1)), 0.0)
            m = re.search(r"Step\s+(\d)", line)
            if m:
                frac = min(int(m.group(1)), 7) / 7.0
                manager.progress(
                    job,
                    progress_base + progress_span * (round_base + 28.0 * frac) / 90.0,
                    f"Fortran round/step: {line.strip()[:80]}",
                )
            m = re.search(r"Iteration\s*#:\s*(\d+)", line)
            if m:
                manager.progress(
                    job,
                    progress_base + progress_span * (round_base + 28.0 + min(int(m.group(1)), 10)) / 90.0,
                    f"scattering iteration {m.group(1)}",
                )
        rc = proc.wait(timeout=600)
    finally:
        manager.set_process(job, None)
    if rc != 0:
        raise RuntimeError(f"Fortran 运行失败，退出码 {rc}")
    # Fortran uses STOP 'string'; exit code is normally 0. Validate outputs.
    brf = case_dir / "000output" / "BRF_BSplusS.txt"
    if not brf.exists():
        raise RuntimeError("Fortran 未生成 BRF_BSplusS.txt")


def _clean(obj):
    """Recursively replace NaN/Inf with None so the JSON is browser-safe."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _read_table(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return np.loadtxt(path, skiprows=1)


def _parse_brf(path: Path) -> list[dict]:
    arr = _read_table(path)
    if arr.ndim == 1:
        arr = arr[None, :]
    out = []
    for row in arr:
        out.append(
            {
                "SZA": float(row[0]),
                "SAA": float(row[1]),
                "SZA_T": float(row[2]),
                "SAA_T": float(row[3]),
                "VZA": float(row[4]),
                "VAA": float(row[5]),
                "VZA_T": float(row[6]),
                "VAA_T": float(row[7]),
                "BRF": float(row[8]),
            }
        )
    return out


def _parse_intensity(path: Path) -> np.ndarray:
    arr = _read_table(path)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def _parse_energy(path: Path) -> dict:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    out = {}
    for key in ("Refl", "Trans", "Absor"):
        m = re.search(rf"{key}\s+([-+0-9.eE]+)\s*([-+0-9.eE]*)", text)
        if m:
            out[key] = [float(m.group(1))] + ([float(m.group(2))] if m.group(2) else [])
    m = re.search(r"delta=\s*([-+0-9.eE]+)%", text)
    if m:
        out["delta_percent"] = float(m.group(1))
    m = re.search(r"Flux\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)", text)
    if m:
        out["Flux"] = [float(m.group(i)) for i in (1, 2, 3)]
    return out


def _parse_profile(path: Path) -> dict:
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = None
    for i, line in enumerate(lines):
        if "u_dir" in line and "idir" in line:
            start = i + 1
            break
    if start is None:
        return {}
    cols = ["z", "p", "u_dir", "idir", "psi", "flux_iD", "flux_iU", "flux_uD", "flux_uU"]
    data = {c: [] for c in cols}
    for line in lines[start:]:
        parts = line.split()
        if len(parts) < 8:
            continue
        try:
            vals = [float(x) for x in parts]
        except ValueError:
            continue
        # first row has no psi column
        if len(vals) >= 9:
            row = vals[:9]
        else:
            row = vals[:4] + [float("nan")] + vals[4:8]
        for c, v in zip(cols, row):
            data[c].append(v)
    return data


def run_single(params: SRTParams, job, manager, base_progress=0.0, span=90.0, use_cache=True) -> dict:
    """Run one i0 case and return parsed tables."""
    exe = find_exe()
    key = params.cache_key(exe_hash(exe))
    cache_dir = manager.runs_dir / "_cache" / key
    cache_file = cache_dir / "result.json"
    if use_cache and cache_file.exists():
        manager.log(job, f"cache hit: {key}")
        manager.progress(job, base_progress + span, "cache hit")
        return json.loads(cache_file.read_text(encoding="utf-8"))

    case_dir = Path(job.params["_result_dir"]) / f"i0_{params.i0:g}"
    if case_dir.exists():
        shutil.rmtree(case_dir)
    (case_dir / "000input").mkdir(parents=True)
    (case_dir / "000output").mkdir()
    (case_dir / "000input" / "InputsForSRT_Comb.txt").write_text(
        params.to_fortran_input(), encoding="ascii"
    )
    _run_process(exe, case_dir, job, manager, base_progress, span)

    result = {
        "params": params.to_dict(),
        "brf_table": _parse_brf(case_dir / "000output" / "BRF_BSplusS.txt"),
        "bs_dir": _parse_intensity(case_dir / "000output" / "BS_dir_IntensityAndFlux.txt").tolist(),
        "bs_dif": _parse_intensity(case_dir / "000output" / "BS_dif_IntensityAndFlux.txt").tolist(),
        "s": _parse_intensity(case_dir / "000output" / "S_IntensityAndFlux.txt").tolist(),
        "energy": {
            "bs_dir": _parse_energy(case_dir / "000output" / "0BS_dir_Fluxes_Direct.txt"),
            "bs_dif": _parse_energy(case_dir / "000output" / "0BS_dif_Fluxes_Direct.txt"),
            "s": _parse_energy(case_dir / "000output" / "0S_Fluxes_Direct.txt"),
        },
        "profile": _parse_profile(case_dir / "000output" / "0BS_dir_Fluxes_Direct.txt"),
        "case_dir": str(case_dir),
    }
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False), encoding="utf-8"
        )
    return result


def _combine_tables(dir_run: dict | None, dif_run: dict | None, f: float) -> dict:
    """Combine the final plane-frame BRFs from the direct and diffuse runs.

    ``BRF_BSplusS.txt`` is already plane-frame for i0=1.  For i0=0 the
    direct-beam cosine factor is not applied (see the SRT_topo fix), so the
    diffuse file is already the plane-frame diffuse BRF.  The RT response is
    linear in the source, so a fractional direct beam is a weighted sum.
    """
    ref = dir_run or dif_run
    if ref is None:
        raise ValueError("no Fortran run")
    dir_table = dir_run["brf_table"] if dir_run is not None else None
    dif_table = dif_run["brf_table"] if dif_run is not None else None

    brf_plane = []
    brf_slope = []
    for i, row in enumerate(ref["brf_table"]):
        factor = np.cos(np.deg2rad(row["SZA_T"])) / max(
            np.cos(np.deg2rad(row["SZA"])), 1e-9
        )
        d = float(dir_table[i]["BRF"]) if dir_table is not None else 0.0
        q = float(dif_table[i]["BRF"]) if dif_table is not None else 0.0
        plane = f * d + (1.0 - f) * q
        slope = f * d / factor + (1.0 - f) * q
        brf_plane.append(plane)
        brf_slope.append(slope)
    return {"brf_plane": brf_plane, "brf_slope": brf_slope}


def run_forward(params: SRTParams, job, manager) -> dict:
    """Run the full forward model, handling fractional direct light."""
    manager.progress(job, 1.0, "validating parameters")
    f = float(params.i0)
    if f >= 1.0 - 1e-9:
        dir_run = run_single(params.with_i0(1.0), job, manager, 0.0, 90.0)
        dif_run = None
        f = 1.0
    elif f <= 1e-9:
        dir_run = None
        dif_run = run_single(params.with_i0(0.0), job, manager, 0.0, 90.0)
        f = 0.0
    else:
        dir_run = run_single(params.with_i0(1.0), job, manager, 0.0, 45.0)
        dif_run = run_single(params.with_i0(0.0), job, manager, 45.0, 45.0)

    combined = _combine_tables(dir_run, dif_run, f)
    first = dir_run or dif_run
    brf_table = first["brf_table"]
    for i, row in enumerate(brf_table):
        if i < len(combined["brf_plane"]):
            row["BRF"] = float(combined["brf_plane"][i])
            row["BRF_slope"] = float(combined["brf_slope"][i])

    result = _clean({
        "engine": "fortran",
        "params": params.to_dict(),
        "angles": brf_table,
        "columns": {
            "brf_plane": combined["brf_plane"],
            "brf_slope": combined["brf_slope"],
        },
        "profile": first.get("profile", {}),
        "energy": first.get("energy", {}),
        "log_tail": job.log[-120:],
        "case_dir": first.get("case_dir", ""),
    })
    out_dir = Path(job.params["_result_dir"])
    (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")

    # CSV of angles + BRF
    csv_lines = ["SZA,SAA,SZA_T,SAA_T,VZA,VAA,VZA_T,VAA_T,BRF_plane,BRF_slope"]
    for row, bp, bsl in zip(brf_table, combined["brf_plane"], combined["brf_slope"]):
        csv_lines.append(
            f"{row['SZA']},{row['SAA']},{row['SZA_T']},{row['SAA_T']},"
            f"{row['VZA']},{row['VAA']},{row['VZA_T']},{row['VAA_T']},{bp:.8g},{bsl:.8g}"
        )
    (out_dir / "brf.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    return result
