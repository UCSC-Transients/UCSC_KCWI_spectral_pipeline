from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal, Any
import json

Shape = Literal[
    "ellipse",
    "circle",
    "rect",
    "ellipse_annulus",
    "circle_annulus",
]


@dataclass
class ApertureShape:
    """A generic aperture in pixel coordinates.

    Parameters by shape
    -------------------
    ellipse:
        x0, y0, a, b, theta_rad
    circle:
        x0, y0, r
    rect:
        x0, y0, width, height, theta_rad
    ellipse_annulus:
        x0, y0, a_in, b_in, a_out, b_out, theta_rad
    circle_annulus:
        x0, y0, r_in, r_out

    Notes
    -----
    - x0, y0 are in pixel coordinates (0..nx-1, 0..ny-1), matching matplotlib imshow origin='lower'.
    - For ellipse/rect theta is radians CCW from +x axis.
    """

    shape: Shape
    params: Tuple[float, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {"shape": self.shape, "params": list(self.params)}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ApertureShape":
        return ApertureShape(shape=d["shape"], params=tuple(float(x) for x in d["params"]))


@dataclass
class TargetBackgroundApertures:
    """Independent target and background definitions."""

    target: ApertureShape
    background: ApertureShape

    def to_dict(self) -> Dict[str, Any]:
        return {"target": self.target.to_dict(), "background": self.background.to_dict()}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TargetBackgroundApertures":
        return TargetBackgroundApertures(
            target=ApertureShape.from_dict(d["target"]),
            background=ApertureShape.from_dict(d["background"]),
        )


@dataclass
class CalibrationConfig:
    spline_s_init: float = 0.05
    telluric_windows: List[Tuple[float, float]] = field(
        default_factory=lambda: [
            (6270, 6330),
            (6860, 6950),
            (7160, 7340),
            (7590, 7700),
            (8120, 8350),
            (8900, 9260),
            (9265, 9630),
            (9635, 10000),
            (10700, 11000),
        ]
    )
    telluric_template_smooth_s: Optional[float] = 0.001
    telluric_min_T: float = 0.02


@dataclass
class JoinConfig:
    blue_trim: str = ""  # 'min:max'
    red_trim: str = ""
    global_excl_blue: List[Tuple[float, float]] = field(default_factory=list)
    global_excl_red: List[Tuple[float, float]] = field(default_factory=list)





@dataclass
class SpectralTrimArmConfig:
    enabled: bool = False
    lam_min: Optional[float] = None
    lam_max: Optional[float] = None


@dataclass
class SpectralTrimConfig:
    blue: SpectralTrimArmConfig = field(default_factory=SpectralTrimArmConfig)
    red: SpectralTrimArmConfig = field(default_factory=SpectralTrimArmConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {"blue": asdict(self.blue), "red": asdict(self.red)}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SpectralTrimConfig":
        b = d.get("blue", {}) if isinstance(d, dict) else {}
        r = d.get("red", {}) if isinstance(d, dict) else {}
        return SpectralTrimConfig(
            blue=SpectralTrimArmConfig(
                enabled=bool(b.get("enabled", False)),
                lam_min=b.get("lam_min", None),
                lam_max=b.get("lam_max", None),
            ),
            red=SpectralTrimArmConfig(
                enabled=bool(r.get("enabled", False)),
                lam_min=r.get("lam_min", None),
                lam_max=r.get("lam_max", None),
            ),
        )


@dataclass
class CoaddConfig:
    enabled: bool = False
    input_dir: Optional[str] = None
    list_file: Optional[str] = None
    outdir: Optional[str] = None
    method: str = "ivar_sigma_clip_mean"
    sigma: float = 4.0
    iters: int = 3
    require_uncert: bool = True


@dataclass
class PipelineConfig:
    """All user-editable configuration in one place."""

    coadd_dir: str
    outdir: str = "kcwi_fluxcal_out"
    show_plots: bool = False
    interactive: bool = True

    coadd: CoaddConfig = field(default_factory=CoaddConfig)

    # Standards and reference spectra
    fluxstd_blue: Optional[str] = None
    fluxstd_red: Optional[str] = None
    ref_blue: Optional[str] = None
    ref_red: Optional[str] = None

    # Apertures per side: you can set defaults, and optionally per-object overrides later.
    apertures_blue: Optional[TargetBackgroundApertures] = None
    apertures_red: Optional[TargetBackgroundApertures] = None

    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    join: JoinConfig = field(default_factory=JoinConfig)

    spectral_trim: SpectralTrimConfig = field(default_factory=SpectralTrimConfig)

    # Optional per-object wavelength exclusions for joined output
    per_object_exclusions: Dict[str, Dict[str, List[Tuple[float, float]]]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert nested aperture objects
        if self.apertures_blue is not None:
            d["apertures_blue"] = self.apertures_blue.to_dict()
        if self.apertures_red is not None:
            d["apertures_red"] = self.apertures_red.to_dict()
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PipelineConfig":
        # Re-hydrate nested dataclasses
        cfg = PipelineConfig(
            coadd_dir=d["coadd_dir"],
            outdir=d.get("outdir", "kcwi_fluxcal_out"),
            show_plots=bool(d.get("show_plots", False)),
            interactive=bool(d.get("interactive", True)),
            coadd=CoaddConfig(**d.get("coadd", {})),
            fluxstd_blue=d.get("fluxstd_blue"),
            fluxstd_red=d.get("fluxstd_red"),
            ref_blue=d.get("ref_blue"),
            ref_red=d.get("ref_red"),
            calibration=CalibrationConfig(**d.get("calibration", {})),
            join=JoinConfig(**d.get("join", {})),
            per_object_exclusions=d.get("per_object_exclusions", {}),
        )
        if d.get("apertures_blue") is not None:
            cfg.apertures_blue = TargetBackgroundApertures.from_dict(d["apertures_blue"])
        if d.get("apertures_red") is not None:
            cfg.apertures_red = TargetBackgroundApertures.from_dict(d["apertures_red"])
        cfg.spectral_trim = SpectralTrimConfig.from_dict(d.get("spectral_trim", {}))
        return cfg


def save_config(cfg: PipelineConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2, sort_keys=False)


def load_config(path: Path) -> PipelineConfig:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return PipelineConfig.from_dict(d)
