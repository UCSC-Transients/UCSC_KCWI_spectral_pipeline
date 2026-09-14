#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="KCWI object-local reduction workflow for consistent KCWI cube products"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_org = sub.add_parser("organize", help="Group KCWI cube products by object and side")
    p_org.add_argument("fits_dir", type=str, help="Directory containing KCWI cube products")
    p_org.add_argument("--project", type=str, required=True, help="Project directory to create/update")
    p_org.add_argument(
        "--mode",
        choices=("symlink", "copy", "move"),
        default="symlink",
        help="How to place FITS files into object directories",
    )

    p_ext = sub.add_parser("extract", help="Extract all exposures in one object directory")
    p_ext.add_argument("object_dir", type=str, help="Object directory created by organize")
    role = p_ext.add_mutually_exclusive_group()
    role.add_argument("--standard", action="store_true", help="Treat this object as a standard star")
    role.add_argument("--science", action="store_true", help="Treat this object as a science target")
    p_ext.add_argument(
        "--side",
        choices=("blue", "red", "both"),
        default="both",
        help="Which side to extract",
    )
    p_ext.add_argument("--calib-dir", type=str, default=None, help="Master calibration directory")
    p_ext.add_argument("--show-plots", action="store_true", help="Show interactive/diagnostic plots")
    p_ext.add_argument("--redo-apertures", action="store_true", help="Ignore saved apertures and redefine them")
    p_ext.add_argument(
        "--no-cr-reject",
        action="store_false",
        dest="cr_reject",
        default=True,
        help="Use original cubes and skip CR-cleaned cube reuse/creation",
    )
    p_ext.add_argument(
        "--redo-cr-reject",
        action="store_true",
        help="Rerun CR rejection from original cubes and overwrite derived CR-cleaned/mask products",
    )
    p_ext.add_argument(
        "--cr-workers",
        type=int,
        default=0,
        help="CR track-detection processes; 0 selects a safe automatic count, 1 disables parallelism",
    )
    p_ext.add_argument(
        "--cr-sigma",
        type=float,
        default=5.5,
        help="High peak or integrated sigma threshold required to seed a CR track",
    )
    p_ext.add_argument(
        "--cr-motion-sigma",
        type=float,
        default=3.25,
        help="Sigma threshold for spikes supporting a CR track",
    )
    p_ext.add_argument(
        "--cr-grow-sigma",
        type=float,
        default=1.75,
        help="Lower sigma threshold used to grow the mask around an accepted CR track",
    )
    p_ext.add_argument(
        "--cr-spectral-window",
        type=int,
        default=7,
        help="Median-filter wavelength window for CR model; even values are rounded up",
    )
    p_ext.add_argument(
        "--cr-spatial-window",
        type=int,
        default=21,
        help="Minimum local-background window; raised above twice the allowed CR width if needed",
    )
    p_ext.add_argument(
        "--cr-max-spatial-footprint",
        type=int,
        default=10,
        help="Maximum CR spike width within one wavelength row of one detector slice",
    )
    p_ext.add_argument(
        "--cr-max-spectral-neighbors",
        type=int,
        default=1,
        help="Static-core wavelength persistence limit, used only with --cr-allow-static-core",
    )
    p_ext.add_argument(
        "--cr-allow-static-core",
        action="store_false",
        dest="cr_require_slice_motion",
        default=True,
        help="Also allow high-sigma compact residuals that do not form moving tracks",
    )
    p_ext.add_argument(
        "--cr-min-slice-shift-pixels",
        type=int,
        default=1,
        help="Minimum total fitted displacement along a CR track",
    )
    p_ext.add_argument(
        "--cr-slice-shift-pixels",
        type=int,
        default=3,
        help="Maximum track movement per wavelength pixel along the slice",
    )
    p_ext.add_argument(
        "--cr-slice-motion-axis",
        choices=("x", "y"),
        default="y",
        help="Within-slice spatial axis along which CR tracks move",
    )
    p_ext.add_argument(
        "--cr-min-track-length",
        type=int,
        default=4,
        help="Minimum number of detected wavelength rows in an accepted CR track",
    )
    p_ext.add_argument(
        "--cr-max-track-span",
        type=int,
        default=64,
        help="Maximum accepted CR track span in wavelength pixels; 0 disables the limit",
    )
    p_ext.add_argument(
        "--cr-max-track-gap",
        type=int,
        default=1,
        help="Maximum missing wavelength rows allowed while linking a CR track",
    )
    p_ext.add_argument(
        "--cr-track-fit-tolerance",
        type=float,
        default=1.5,
        help="Allowed centroid scatter in pixels around the fitted CR track",
    )
    p_ext.add_argument(
        "--cr-mask-margin",
        type=int,
        default=1,
        help="Extra within-slice pixels masked on each side of an accepted CR tube",
    )
    p_ext.add_argument(
        "--cr-neighbor-veto-fraction",
        type=float,
        default=0.6,
        help="Reject tracks matched by a fitted adjacent-slice track over at least this row fraction",
    )
    p_ext.add_argument(
        "--no-spectral-cr-review",
        action="store_false",
        dest="spectral_cr_review",
        default=True,
        help="Skip interactive review of coadded features narrower than the expected line-spread function",
    )
    p_ext.add_argument(
        "--spectral-cr-resolving-power",
        type=float,
        default=None,
        help="Override resolving power used for narrow-line review instead of deriving it from the FITS header",
    )
    p_ext.add_argument(
        "--spectral-cr-sigma",
        type=float,
        default=5.0,
        help="Minimum absolute continuum-subtracted significance for a narrow-line review candidate",
    )
    p_ext.add_argument(
        "--spectral-cr-max-lsf-fraction",
        type=float,
        default=0.65,
        help="Flag features with measured FWHM below this fraction of the expected instrumental FWHM",
    )
    p_ext.add_argument(
        "--join-only",
        action="store_true",
        help="Skip extraction/calibration and redo only the BLUE+RED scaling/join from existing fluxcal spectra",
    )

    args = parser.parse_args()

    if args.command == "organize":
        from kcwi_pipeline.project import organize_project

        manifest = organize_project(
            Path(args.fits_dir),
            Path(args.project),
            mode=args.mode,
        )
        objects = manifest.get("objects", {})
        print(f"Organized {len(manifest.get('files', []))} files into {len(objects)} object directories")
        print(f"Project: {Path(args.project).expanduser().resolve()}")
        return

    if args.command == "extract":
        from kcwi_pipeline.cosmic_rays import CosmicRayRejectionConfig
        from kcwi_pipeline.object_workflow import extract_object
        from kcwi_pipeline.spectral_cr import SpectralCRConfig

        cr_config = CosmicRayRejectionConfig(
            sigma=float(args.cr_sigma),
            motion_sigma=float(args.cr_motion_sigma),
            grow_sigma=float(args.cr_grow_sigma),
            spectral_window=int(args.cr_spectral_window),
            spatial_window=int(args.cr_spatial_window),
            max_spatial_footprint=int(args.cr_max_spatial_footprint),
            max_spectral_neighbors=int(args.cr_max_spectral_neighbors),
            require_slice_motion=bool(args.cr_require_slice_motion),
            min_slice_shift_pixels=int(args.cr_min_slice_shift_pixels),
            slice_shift_pixels=int(args.cr_slice_shift_pixels),
            slice_motion_axis=str(args.cr_slice_motion_axis),
            min_track_length=int(args.cr_min_track_length),
            max_track_span=int(args.cr_max_track_span),
            max_track_gap=int(args.cr_max_track_gap),
            track_fit_tolerance=float(args.cr_track_fit_tolerance),
            mask_margin=int(args.cr_mask_margin),
            neighbor_veto_fraction=float(args.cr_neighbor_veto_fraction),
            workers=int(args.cr_workers),
        )
        spectral_cr_config = SpectralCRConfig(
            detection_sigma=float(args.spectral_cr_sigma),
            max_lsf_fraction=float(args.spectral_cr_max_lsf_fraction),
        )
        standard = True if args.standard else False if args.science else None
        extract_object(
            Path(args.object_dir),
            calib_dir=Path(args.calib_dir) if args.calib_dir else None,
            standard=standard,
            side=args.side,
            show_plots=bool(args.show_plots),
            redo_apertures=bool(args.redo_apertures),
            cr_reject=bool(args.cr_reject),
            redo_cr_reject=bool(args.redo_cr_reject),
            cr_config=cr_config,
            spectral_cr_review=bool(args.spectral_cr_review),
            spectral_cr_resolving_power=args.spectral_cr_resolving_power,
            spectral_cr_config=spectral_cr_config,
            join_only=bool(args.join_only),
        )
        return


if __name__ == "__main__":
    main()
