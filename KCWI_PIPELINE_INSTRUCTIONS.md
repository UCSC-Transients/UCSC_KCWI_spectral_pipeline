# KCWI Spectral Reduction Pipeline Instructions

This pipeline organizes KCWI cube products, extracts standard-star and science
spectra, applies flux and RED telluric calibrations, and writes final 1D spectra.
It accepts either `*_icubes.fits` or `*_icubed.fits` products.

For quicklook reduction, use Level 1 `*_icubed.fits` files which are available 
on KOA within minutes of observation.

Use one cube type throughout a project. The pipeline rejects mixed `icubes` and
`icubed` inputs because standards and science targets must be processed
consistently. See [Cube products and exposure time](#cube-products-and-exposure-time)
for the difference between the two products.

Use only one instrumental setup per side in each project. Do not mix cubes
taken with different gratings, central wavelengths, slicers, or other wavelength
setups. A project may contain its matched BLUE and RED sides, but observations
from different setups belong in separate project directories.

## Quick start

Run commands from the pipeline repository with a Python environment containing:

```text
numpy
scipy
astropy
photutils
matplotlib
```

The usual reduction order is:

1. Organize the downloaded cubes into a project.
2. Extract the relevant standard stars.
3. Extract the science targets.
4. Inspect the final `.flm` spectrum, PNG, and diagnostics.

### 1. Organize the cubes

Place all cubes for one project under one input directory, then run:

```bash
python run_kcwi_reduction.py organize /path/to/koa_download \
  --project /path/to/kcwi_project
```

The default mode creates symbolic links, avoiding duplication of large FITS
files. To copy or move the files instead, use `--mode copy` or `--mode move`.
Use `move` only when the source files should actually be relocated.

The resulting project begins with:

```text
kcwi_project/
  objects/
    OBJECT_NAME/
      BLUE/
      RED/
  calibrations/
    calibration_registry.json
  project_manifest.json
```

Object names and sides are read from FITS metadata. The input directory may be
nested; the organizer searches it recursively.

### 2. Extract the standard stars

Process standards before science targets so that sensitivity functions are
available. Run the side or sides covered by each standard:

```bash
python run_kcwi_reduction.py extract \
  /path/to/kcwi_project/objects/STD_OBJECT \
  --standard --side blue

python run_kcwi_reduction.py extract \
  /path/to/kcwi_project/objects/STD_OBJECT \
  --standard --side red
```

Use `--side both` if reducing both sides together. If
`--side` is omitted, `both` is used.

During the run, the pipeline will:

1. Report the common wavelength coverage of the standard exposures.
2. Suggest a usable range and ask you to accept or replace it.
3. Create or reuse cosmic ray (CR)-cleaned cubes.
4. Ask you to define or approve apertures.
5. Extract and coadd the exposures.
6. Ask which built-in flux standard corresponds to the object.
7. Open the continuum/sensitivity editor.
8. Save a calibration entry for each completed side.
9. Build a RED telluric template when processing the RED side.

The automatic wavelength suggestion removes 300 A from both ends of BLUE cubes
and 450 A from both ends of RED cubes. Inspect the cube and wavelength-dependent
data quality before accepting it. Press Enter to accept the suggestion or enter
new limits as `MIN:MAX`. The approved project ranges are saved in
`calibrations/wavelength_ranges.json` and reused by later standards and science
targets.

Calibration products are saved under:

```text
kcwi_project/calibrations/STD_OBJECT/SIDE/
```

The standard's processed spectrum and diagnostics are also retained in its
object directory. See [Standard continuum editor](#standard-continuum-editor)
and [Flux and telluric calibration](#flux-and-telluric-calibration) for details.

### 3. Extract a science target

To process both sides:

```bash
python run_kcwi_reduction.py extract \
  /path/to/kcwi_project/objects/SCIENCE_OBJECT \
  --science --side both
```

To process only one side:

```bash
python run_kcwi_reduction.py extract \
  /path/to/kcwi_project/objects/SCIENCE_OBJECT \
  --science --side red
```

For each requested side, the pipeline will:

1. Load the wavelength range approved while reducing the standard star.
2. Create or reuse CR-cleaned cubes.
3. Define, propagate, or review apertures for every exposure.
4. Extract the 1D spectra and open the coadd review window.
5. Review CR-like narrow features in the coadd.
6. Ask which compatible standard calibration to use.
7. Apply sensitivity calibration and, for RED, review telluric alignment.
8. Save a side-level flux-calibrated spectrum.

When both side-level spectra are available, the pipeline opens the BLUE/RED
scaling window and then writes a combined spectrum. The sides have may not have
spectral overlap; the final file is a wavelength-sorted concatenation of the independently
scaled sides.

### 4. Locate the final products

For a two-sided science reduction:

```text
objects/SCIENCE_OBJECT/final/SCIENCE_OBJECT_BLUE+RED_spectrum.flm
objects/SCIENCE_OBJECT/final/SCIENCE_OBJECT_BLUE+RED_spectrum.png
```

For a one-sided reduction:

```text
objects/SCIENCE_OBJECT/final/SCIENCE_OBJECT_BLUE_spectrum.flm
objects/SCIENCE_OBJECT/final/SCIENCE_OBJECT_BLUE_spectrum.png
objects/SCIENCE_OBJECT/final/SCIENCE_OBJECT_RED_spectrum.flm
objects/SCIENCE_OBJECT/final/SCIENCE_OBJECT_RED_spectrum.png
```

Final PNGs show the 1-sigma uncertainty as gray shading when uncertainty is
available. See [Outputs and units](#outputs-and-units) for the full directory
layout and file format.

## Interactive windows

Required interactive windows open even without `--show-plots`. That option
also displays additional diagnostics.

### Aperture editor

The aperture display has two views of the same white-light image:

- left: original aspect ratio;
- right: vertically compressed for easier visual comparison with sky charts.

Both panels use the same image coordinates. You can draw or drag from either
panel. When the cube contains a valid celestial WCS, the box at the left reports
the mouse pointer's RA and Dec in sexagesimal notation. Display controls are:

- `Wavelength (A)`: changes the wavelength interval used for the white-light
  image only; extraction still uses the full configured side range;
- `Low %` and `High %`: adjust image contrast;
- `Reset`: restores the full wavelength interval and 5--99% contrast.

When defining a new aperture, select the initial target and background shapes
in the terminal and draw them in the image. All new, saved, propagated, and
cross-side aperture proposals then open in the same editor with:

```text
Accept
Move target
Resize target
Target shape...
Move background
Resize background
Background shape...
Enter values...
Cancel
```

Choose a move or resize mode, then drag the aperture in either image panel.
`Accept` continues the pipeline.

Keyboard shortcuts in the combined editor are:

```text
a or Enter  accept
t           move target
b           move background
m           move the active aperture
e           resize the active aperture
q           cancel
```

The first approved aperture is proposed for subsequent exposures. When both
sides are processed, it is transformed through the celestial WCS and proposed
for the other side. Every proposal remains editable before acceptance. Saved
apertures are stored under `apertures/SIDE/`; use `--redo-apertures` to ignore
them and start again.

### Coadd review

The science coadd window shows the offset individual exposures, the current
coadd and uncertainty, and the number of accepted exposures at each wavelength.
Use the `Clip sigma` slider to change rejection strength, `Reset` to return to
the default, and `Approve` (or `a`/Enter) to continue. Pressing `q` aborts the
review.

The approved clipping threshold is recorded in `extraction_state.json`. See
[Extraction, coaddition, and uncertainty](#extraction-coaddition-and-uncertainty)
for the calculation.

### Narrow-feature CR review

After the science coadd, the pipeline proposes unusually narrow positive or
negative features for review. In each candidate window:

- the upper panel shows the candidate and proposed replacement pixels;
- the lower panel shows the full spectrum and highlights the upper panel's
  wavelength range;
- `Accept line` keeps the feature;
- `Remove as CR` replaces the highlighted pixels by interpolation.

After the last candidate, inspect the full result and choose `Accept result` or
`Redo review`.

Keyboard shortcuts are:

```text
Candidate review: a or Enter keeps the line
Candidate review: r, Backspace, or Delete removes it
Final review:     a or Enter accepts; r starts over
```

Use `--no-spectral-cr-review` to skip this stage. Detection controls are listed
under [Useful rerun and override options](#useful-rerun-and-override-options),
with algorithm details in [Cosmic-ray treatment](#cosmic-ray-treatment).

### Standard continuum editor

This window defines the observed standard-star continuum used to construct the
sensitivity function. Previously accepted points are loaded when that standard
and side are rerun.

```text
left-click     add a point
drag marker    move a point
right-click    delete the nearest point
z              enter zoom-box mode
o              restore the original view
r              reset to automatic points
a or Enter     accept
q              quit
```

On RED standards, orange telluric regions are excluded from the continuum fit.

### RED telluric alignment

The science RED alignment window compares the science spectrum with the
airmass-scaled telluric template in the O2 B and A bands. Move the `Shift (A)`
slider until the features align, then click `Accept`. A positive shift moves the
template redward.

If the standard or science airmass is unavailable, the pipeline warns and skips
the telluric correction.

### BLUE/RED scaling and acceptance

The final join window keeps the BLUE and RED traces color-coded and provides:

```text
Blue scale slider   multiply BLUE by 0.1--10
Red scale slider    multiply RED by 0.1--10
Reset               restore both scales to 1
Approve             accept and continue
```

You can also press `a`/Enter to approve, `z` to save the current zoom, or `q` to
abort. If the window is closed without approval, equivalent choices are offered
in the terminal and the window can be reopened.

The accepted factors are applied to flux and uncertainty and saved in:

```text
final/SCIENCE_OBJECT_join_scale.txt
```

## Re-running work

### Redo apertures

Ignore saved apertures and define them again:

```bash
python run_kcwi_reduction.py extract /path/to/kcwi_project/objects/OBJECT \
  --science --side red --redo-apertures
```

Replace `--science` with `--standard` for a standard star.

### Rerun or disable cube-level CR rejection

CR rejection is enabled by default. Valid cached `_crclean.fits` products are
reused. To rerun rejection from the original cubes and replace the derived CR
products:

```bash
python run_kcwi_reduction.py extract /path/to/kcwi_project/objects/OBJECT \
  --science --side red --redo-cr-reject
```

To extract directly from the original cubes without creating or reusing
CR-cleaned cubes:

```bash
python run_kcwi_reduction.py extract /path/to/kcwi_project/objects/OBJECT \
  --science --side red --no-cr-reject
```

Use `--cr-workers 1` for serial detection or `--cr-workers N` for a specific
worker count. The default, `0`, selects a safe automatic count.

### Rerun one side

Run a normal one-side extraction. If the other side already has a spectrum in
`fluxcal/`, the pipeline reuses it and reopens the final scaling window:

```bash
python run_kcwi_reduction.py extract /path/to/kcwi_project/objects/SCIENCE_OBJECT \
  --science --side blue
```

### Redo only the BLUE/RED scaling

When both side-level spectra already exist:

```bash
python run_kcwi_reduction.py extract /path/to/kcwi_project/objects/SCIENCE_OBJECT \
  --science --join-only
```

This skips extraction, coaddition, calibration, and telluric correction. It
loads the two `fluxcal/*.flm` files and rewrites the joined products in `final/`.

### Use a calibration directory explicitly

The project calibration directory is normally found automatically. Override it
with:

```bash
python run_kcwi_reduction.py extract /path/to/kcwi_project/objects/SCIENCE_OBJECT \
  --science --side both --calib-dir /path/to/kcwi_project/calibrations
```

### Useful rerun and override options

```text
--show-plots                       display additional diagnostics
--redo-apertures                   ignore saved aperture JSON files
--no-cr-reject                     bypass cube-level CR cleaning
--redo-cr-reject                   rebuild cube-level CR products
--cr-workers N                     set CR detector process count
--no-spectral-cr-review            skip coadded-spectrum CR review
--spectral-cr-resolving-power R    override header-derived resolving power
--spectral-cr-sigma S              set the candidate significance threshold
--spectral-cr-max-lsf-fraction F   set the maximum candidate/LSF width ratio
--blue-range MIN_A MAX_A           override the BLUE project range
--red-range MIN_A MAX_A            override the RED project range
--join-only                        redo only BLUE/RED scaling and concatenation
```

For all cube-level CR tuning parameters and their current defaults, run:

```bash
python run_kcwi_reduction.py extract --help
```

## Outputs and units

The main object directory is:

```text
objects/OBJECT/
  BLUE/ and RED/     input cubes, *_crclean.fits, and *_crmask.fits
  apertures/         accepted aperture JSON files
  extracted/         individual extracted spectra
  coadded_spectra/   approved side coadds and exposure counts
  fluxcal/           side-level calibrated spectra and telluric diagnostics
  final/             final spectra, plots, and join scale
  diagnostics/       aperture, CR, coadd, and calibration plots
  extraction_state.json
```

Project-level approved wavelength limits are stored in:

```text
calibrations/wavelength_ranges.json
```

Spectrum tables use the `.flm` extension. When uncertainty is available, they
have three columns:

```text
lambda_A  flux_or_counts  sigma_flux_or_counts
```

Otherwise they contain wavelength and flux/counts only. Final and `fluxcal/`
spectra use:

```text
1e-15 erg/s/cm^2/A
```

For example, a saved flux value of `2.4` means
`2.4e-15 erg/s/cm^2/A`. Flux and `sigma_flux` use the same units.

Frequently used files include:

```text
extracted/SIDE/*_counts.flm
coadded_spectra/OBJECT_SIDE_counts_coadd.flm
coadded_spectra/OBJECT_SIDE_nexp.txt
fluxcal/OBJECT_SIDE_fluxcal.flm
final/OBJECT_BLUE+RED_spectrum.flm
final/OBJECT_SIDE_spectrum.flm
```

The standard-star equivalents are:

```text
final/STD_OBJECT_BLUE_standard_processed.flm
final/STD_OBJECT_RED_standard_processed.flm
```

## Data requirements and caveats

- Do not mix `*_icubes.fits` and `*_icubed.fits` in one project or requested
  two-side extraction.
- Do not mix instrumental or wavelength setups for the same side in one
  project. Create separate projects for different gratings, central
  wavelengths, slicers, or other setup changes. Matched BLUE and RED sides may
  remain together.
- Build standards from the same cube type as the science data.
- Reduce a standard for a side before reducing science data on that side so the
  project wavelength range and sensitivity calibration are available.
- Aperture propagation between cubes depends on valid celestial WCS metadata.
  If transformation fails or falls outside the field, define the aperture
  normally.
- RED telluric correction requires valid standard and science airmasses.
- The final uncertainty shading appears only when an uncertainty column exists.
- Use the generated diagnostics and `extraction_state.json` to audit processing
  choices.

## Technical details

The sections below document the calculations and defaults. They are not needed
for a routine run, but explain the behavior referenced above.

### Cube products and exposure time

The two supported suffixes describe different KCWI processing stages:

- `*_icubed.fits`: the extracted spectrum and uncertainty are divided by that
  exposure's positive exposure time before coaddition and calibration. The
  pipeline checks `XPOSURE`, `ELAPTIME`, `EXPTIME`, `TELAPSE`, then `TTIME`.
- `*_icubes.fits`: values are already exposure-normalized by the DRP and are not
  divided by exposure time again.

The standard calibration registry records cube type, input units, exposure
normalization, and schema version. A calibration is offered only when compatible
with the science cube type.

### Cosmic-ray treatment

Cube-level CR rejection analyzes detector slices as wavelength-versus-position
images. It links narrow spatial spikes into tracks, checks them against a
same-spaxel spectral model, rejects tracks that resemble astronomical structure,
and interpolates accepted voxels along wavelength. Current default seed,
track-support, and mask-growth thresholds are 5.5, 3.25, and 1.75 sigma. Tracks
spanning more than 64 wavelength pixels are rejected by default.

Derived products are saved beside each input cube:

```text
*_crclean.fits
*_crmask.fits
```

When an input `UNCERT` extension exists, replaced voxels receive updated
uncertainties that include propagated interpolation uncertainty and local model
scatter. Cached products from an incompatible detector version are rebuilt.

The later 1D CR review uses the grating and slicer metadata to estimate resolving
power. By default, a candidate must have absolute continuum-subtracted
significance of at least 5 sigma and measured FWHM below 0.65 times the expected
instrumental FWHM. Both positive and negative residuals are checked because a CR
in the background aperture can create a negative feature after subtraction.

Removing a candidate replaces only its highlighted pixels. Endpoint variance
and local interpolation scatter are propagated into the replacement
uncertainty. Review decisions and the original coadd are retained under
`coadded_spectra/` and `diagnostics/SIDE/`.

### Extraction, coaddition, and uncertainty

The pipeline extracts every exposure separately. At each wavelength, target
spaxels are summed and a sigma-clipped weighted background is subtracted. If
flags or nonfinite values remove target spaxels, the surviving fractional target
area is used for both background subtraction and its variance.

The background uncertainty is the larger of:

- uncertainty propagated from the input variance; and
- the standard error inferred from retained background-spaxel scatter.

Before coaddition, spectra are placed on the first exposure's wavelength grid
when necessary. Flux is linearly interpolated; variance is propagated with the
squares of the interpolation weights rather than interpolating sigma directly.

The default coadd uses symmetric clipping with:

```text
sigma = 2.0
maxiters = 5
```

The user can change `sigma` in the coadd review window. Surviving samples are
inverse-variance weighted when every exposure has valid uncertainty; otherwise
they are mean-combined. Formal coadd uncertainty is inflated by the square root
of reduced chi-square where accepted exposures disagree more than their reported
uncertainties predict. This check can increase, but never decrease, the formal
uncertainty.

The number of accepted exposures at each wavelength is saved in
`coadded_spectra/OBJECT_SIDE_nexp.txt`.

### Flux and telluric calibration

For a standard star, the built-in AB mag reference spectrum is interpolated onto the
observed wavelength grid. The sensitivity is:

```text
sensitivity = reference_flux / fitted_observed_continuum
```

Applying sensitivity, telluric transmission, or an accepted BLUE/RED scale is a
multiplicative operation; the spectrum uncertainty is multiplied by the
absolute value of the same factor.

For RED standards, the telluric template is the observed standard divided by
its fitted continuum within configured atmospheric windows. For RED science,
the template is shifted using the O2 B and A bands and scaled for the standard
and science airmasses:

```text
T_scaled = T_shifted ** ((X_science / X_standard) ** 0.55)
flux_corrected = flux_uncorrected / T_scaled
```

Correction windows are:

```text
6270--6330 A
6860--6950 A
7160--7340 A
7590--7700 A
8120--8350 A
8900--9260 A
9265--9630 A
9635--10000 A
10700--11000 A
```

The two 5 A gaps inside the broad 8900--10000 A water-vapor region are left
available for continuum anchor points in the interactive spline editor.

Only `6860--6935 A` and `7590--7700 A` are used for the automatic shift
estimate.

### Wavelength ranges

For the first standard reduced on each side, the pipeline measures the common
wavelength overlap of every exposure and suggests trimming 300 A from each
BLUE edge or 450 A from each RED edge. The user can accept or replace that suggestion.
Explicit `--blue-range MIN MAX` and `--red-range MIN MAX` arguments take precedence.

The approved range is recorded in `calibrations/wavelength_ranges.json`, the
standard calibration registry, and each object's `extraction_state.json`. It is
used for white-light aperture images, extraction, coaddition, calibration,
telluric products, diagnostics, and final spectra. Full input and CR-cleaned
FITS cubes are retained without spectral trimming, so the project can be rerun
with a different approved range.
