"""Parameter container and validation for the SRT/TopoSRT GUI."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path


@dataclass
class SRTParams:
    # canopy structure
    H: float = 5.0
    cover: float = 0.49397
    d_L: float = 1.0
    tree_diam: float = 5.0
    tree_type: int = 4          # 1 cylinder, 2 cone, 3 ellipsoid, 4 sphere
    lad: int = 6                # 1..9, 6 = spherical
    # optics
    tau_L: float = 0.4843
    ro_L: float = 0.4746
    ro_soil: float = 0.3379
    ifren: int = 2
    refind: float = 1.0
    sk: float = 0.05
    hot_spot: float = 8.0
    # geometry
    sza: float = 45.0
    saa: float = 270.0
    slope: float = 0.0
    aspect: float = 90.0
    alpha_slope: float = 0.0
    beta_aspect: float = 90.0
    coord_flag: int = 1
    i0: float = 1.0
    # discretization
    nz: int = 20
    epsiter: float = 1e-7
    maxiter: int = 10
    # view angles
    vza: list[float] = field(default_factory=lambda: [0.0, 15.0, 30.0, 45.0, 60.0])
    vaa: list[float] = field(default_factory=lambda: [0.0, 90.0, 180.0, 270.0])

    def __post_init__(self):
        self.vza = [float(x) for x in self.vza]
        self.vaa = [float(x) for x in self.vaa]
        self.validate()

    def validate(self):
        if not (0 < self.cover <= 1):
            raise ValueError("cover 必须在 (0, 1]")
        if self.H <= 0:
            raise ValueError("H 必须 > 0")
        if self.d_L <= 0:
            raise ValueError("d_L 必须 > 0")
        if self.tree_diam <= 0:
            raise ValueError("tree_diam 必须 > 0")
        if self.tree_type not in (1, 2, 3, 4):
            raise ValueError("tree_type 必须是 1/2/3/4")
        if self.lad not in range(1, 10):
            raise ValueError("lad 必须在 1..9")
        if not (0 <= self.ro_soil < 1):
            raise ValueError("ro_soil 必须在 [0, 1)")
        if not (0 <= self.tau_L <= 1 and 0 <= self.ro_L <= 1):
            raise ValueError("tau_L/ro_L 必须在 [0, 1]")
        if self.tau_L + self.ro_L > 1.0 + 1e-9:
            raise ValueError("tau_L + ro_L 不能超过 1")
        if not (0 < self.sza < 90):
            raise ValueError("SZA 必须在 (0, 90)")
        if not (0 <= self.slope < 90):
            raise ValueError("slope 必须在 [0, 90)")
        if not (0 <= self.aspect < 360):
            raise ValueError("aspect 必须在 [0, 360)")
        if not (0 <= self.alpha_slope < 90):
            raise ValueError("alpha_slope 必须在 [0, 90)")
        if not (0 <= self.beta_aspect < 360):
            raise ValueError("beta_aspect 必须在 [0, 360)")
        if not (0 <= self.i0 <= 1):
            raise ValueError("i0 必须在 [0, 1]")
        if not (1 <= self.nz <= 29):
            raise ValueError("nz 必须在 1..29")
        if self.maxiter < 6:
            raise ValueError("maxiter 必须 >= 6")
        if not self.vza or not self.vaa:
            raise ValueError("观测角列表不能为空")
        if any(not (0 <= z < 90) for z in self.vza):
            raise ValueError("VZA 必须在 [0, 90)")
        if any(not (0 <= a < 360) for a in self.vaa):
            raise ValueError("VAA 必须在 [0, 360)")

    @property
    def nfi_usr(self) -> int:
        return len(self.vza) * len(self.vaa)

    def view_grid(self):
        zens = [z for z in self.vza for _ in self.vaa]
        azis = [a for _ in self.vza for a in self.vaa]
        return zens, azis

    def with_i0(self, value: float) -> "SRTParams":
        return replace(self, i0=float(value))

    def to_fortran_input(self) -> str:
        zens, azis = self.view_grid()
        lines = [
            "# Canopy Info",
            "#",
            f"{self.i0} {self.sza} {self.saa} {self.tree_type} {self.H} "
            f"{self.d_L} {self.tree_diam} {self.cover} {self.hot_spot}",
            "#",
            "#",
            f"{self.tau_L} {self.ro_L} {self.lad} {self.ro_soil} "
            f"{self.ifren} {self.refind} {self.sk}",
            "#",
            "#",
            f"{self.slope} {self.aspect} {self.alpha_slope} {self.beta_aspect}",
            "#",
            "#",
            f"{self.nz} {self.epsiter} {self.maxiter}",
            "#",
            "#",
            f"{self.nfi_usr} {self.coord_flag}",
            "#",
            " ".join(f"{z:g}" for z in zens),
            " ".join(f"{a:g}" for a in azis),
        ]
        return "\n".join(lines) + "\n"

    def to_dict(self) -> dict:
        return asdict(self)

    def cache_key(self, exe_hash: str = "") -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256((payload + exe_hash).encode("utf-8")).hexdigest()[:24]

    @classmethod
    def from_dict(cls, data: dict) -> "SRTParams":
        allowed = {f for f in cls.__dataclass_fields__}
        clean = {k: v for k, v in (data or {}).items() if k in allowed}
        return cls(**clean)
