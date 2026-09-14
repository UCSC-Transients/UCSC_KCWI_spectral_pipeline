from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

from astropy.io import fits


@dataclass
class OrganizedFile:
    source: str
    linked_path: str
    object_name: str
    object_dir: str
    side: str
    camera: str
    imtype: Optional[str]
    date_obs: Optional[str]
    airmass: Optional[float]
    product_type: str


def _clean_object_name(value: object) -> str:
    text = str(value or "").strip()
    if not text or text.upper() in {"UNKNOWN", "UNDEF", "NONE", "N/A"}:
        return "UNKNOWN"
    return text


def _safe_filename(value: object) -> str:
    return "".join(c if (c.isalnum() or c in ("-", "_", ".")) else "_" for c in str(value))


def _clean_side(value: object, filename: str) -> str:
    text = str(value or "").strip().upper()
    if text in {"B", "BLUE", "KCWI-BLUE"}:
        return "BLUE"
    if text in {"R", "RED", "KCWI-RED"}:
        return "RED"
    stem = filename.upper()
    if stem.startswith("KB."):
        return "BLUE"
    if stem.startswith("KR."):
        return "RED"
    return text or "UNKNOWN"


def _link_copy_or_move(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(str(src), str(dst))
    else:
        raise ValueError(f"Unknown file organization mode: {mode}")


def discover_cube_products(input_dir: Path) -> List[Path]:
    input_dir = input_dir.expanduser().resolve()
    return sorted({
        path
        for pattern in ("*_icubes.fits", "*_icubed.fits")
        for path in input_dir.rglob(pattern)
    })


def discover_icubes(input_dir: Path) -> List[Path]:
    """Backward-compatible discovery of Level 2 ``*_icubes.fits`` files."""
    input_dir = input_dir.expanduser().resolve()
    return sorted(input_dir.rglob("*_icubes.fits"))


def cube_product_type(path: Path) -> str:
    """Return the pipeline product class encoded by a KCWI cube filename."""
    name = path.name.lower()
    if name.endswith("_icubed.fits"):
        return "icubed"
    if name.endswith("_icubes.fits"):
        return "icubes"
    return "unknown"


def organize_project(
    input_dir: Path,
    project_dir: Path,
    *,
    mode: str = "symlink",
) -> Dict[str, object]:
    """Organize one consistent KCWI cube product type into object/side folders."""
    input_dir = input_dir.expanduser().resolve()
    project_dir = project_dir.expanduser().resolve()
    objects_dir = project_dir / "objects"
    calibrations_dir = project_dir / "calibrations"
    objects_dir.mkdir(parents=True, exist_ok=True)
    calibrations_dir.mkdir(parents=True, exist_ok=True)

    files = discover_cube_products(input_dir)
    if not files:
        raise FileNotFoundError(f"No *_icubes.fits or *_icubed.fits files found under {input_dir}")

    product_types = {cube_product_type(path) for path in files}
    if len(product_types) != 1:
        print(
            "WARNING: Mixed KCWI cube products detected. All standards and science "
            "targets in one project must use the same *_icubed.fits or *_icubes.fits type."
        )
        details = ", ".join(
            f"{product_type}: {sum(cube_product_type(path) == product_type for path in files)}"
            for product_type in sorted(product_types)
        )
        raise ValueError(f"Cannot organize mixed cube products ({details})")
    project_product_type = next(iter(product_types))

    organized: List[OrganizedFile] = []
    for src in files:
        with fits.open(src, memmap=True) as hdul:
            cube_hdu = next(
                (
                    hdu for hdu in hdul
                    if getattr(hdu, "data", None) is not None
                    and getattr(hdu.data, "ndim", 0) == 3
                ),
                hdul[0],
            )
            hdr = cube_hdu.header.copy()
            for key in (
                "OBJECT", "TARGNAME", "CAMERA", "AIRMASS", "IMTYPE",
                "DATE-OBS", "DATE-BEG", "DATE-END", "EXPTIME", "ELAPTIME",
                "XPOSURE", "TELAPSE", "TTIME",
            ):
                if key in hdul[0].header and key not in hdr:
                    hdr[key] = hdul[0].header[key]
            for key, value in hdul[0].header.items():
                if key not in hdr and (
                    key.startswith(("WCSAXES", "CTYPE", "CRVAL", "CRPIX", "CDELT", "CD", "PC"))
                ):
                    hdr[key] = value
            obj = _clean_object_name(hdr.get("OBJECT", hdr.get("TARGNAME")))
            side = _clean_side(hdr.get("CAMERA"), src.name)
            imtype = hdr.get("IMTYPE")
            date_obs = hdr.get("DATE-OBS", hdr.get("DATE_BEG"))
            try:
                airmass = float(hdr["AIRMASS"]) if "AIRMASS" in hdr else None
            except Exception:
                airmass = None

        obj_dirname = _safe_filename(obj)
        dst = objects_dir / obj_dirname / side / src.name
        _link_copy_or_move(src, dst, mode)
        organized.append(
            OrganizedFile(
                source=str(src),
                linked_path=str(dst),
                object_name=obj,
                object_dir=obj_dirname,
                side=side,
                camera=side,
                imtype=imtype,
                date_obs=date_obs,
                airmass=airmass,
                product_type=cube_product_type(src),
            )
        )

    manifest: Dict[str, object] = {
        "input_dir": str(input_dir),
        "project_dir": str(project_dir),
        "mode": mode,
        "product_type": project_product_type,
        "files": [asdict(item) for item in organized],
        "objects": {},
    }
    objects: Dict[str, Dict[str, List[str]]] = {}
    for item in organized:
        objects.setdefault(item.object_dir, {}).setdefault(item.side, []).append(item.linked_path)
    manifest["objects"] = objects

    manifest_path = project_dir / "project_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    registry = calibrations_dir / "calibration_registry.json"
    if not registry.exists():
        with open(registry, "w", encoding="utf-8") as f:
            json.dump({"standards": []}, f, indent=2)

    return manifest


def find_project_root(path: Path) -> Optional[Path]:
    path = path.expanduser().resolve()
    candidates = [path] + list(path.parents)
    for candidate in candidates:
        if (candidate / "project_manifest.json").exists() and (candidate / "calibrations").exists():
            return candidate
    return None
