# GADGET GUI: reusable RvE cuts and multiple runs

This update is based on the `gui_framework_current.zip` supplied on October 18,
2026. It is an update for the existing repository, not a standalone replacement
for the raw_viewer, fit_gui, calibration, pad mapping, and detector-data files.

## Install with a backup

Close the running GUI first. Put this updated ZIP in your repository directory.
Run the following from that directory (the directory containing main_gui.py):

```bash
gui_backup_dir="$HOME/gui_framework_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$gui_backup_dir"
cp -- main_gui.py GadgetRunH5.py RvE_Frame.py RunSelectFrame.py \
  EnergySpectrumFrame.py IndividualEventFrame.py prev_cut_select_window.py \
  gadget_widgets.py "$gui_backup_dir/"
unzip -o gui_framework_current.zip
python main_gui.py
```

Install all files in the ZIP together. Three new application modules are required:
`GadgetRunCollection.py`, `rve_cuts.py`, and `event_image_io.py`.
There are no new third-party dependencies. Keep your existing Python environment.
The single-run GadgetRunH5 class remains available for other scripts.

## Load one run or several runs

Enter the experiment, then type `145` or `145,147,149` in the Runs box and click
Load. Whitespace is allowed and duplicate run numbers are removed. Invalid or
missing run entries are reported. All requested processed ROOT files must exist;
loading does not silently force-reprocess data. If a load fails, the previous
loaded data remain available. The separate force-reprocess button also accepts
comma-separated runs and remains an explicit processing action.

The RvE histogram combines the selected runs. Its filter settings apply to all
currently loaded runs. Timing is obtained separately for each source run before
combining its events. The energy spectrum and existing fitting tools use the
combined arrays. The energy-unit dropdown defaults to the calibrated units;
choosing ADC counts lazily obtains integrated charge with veto pads excluded.

The Individual Events tab and Show event on RvE plot have explicit Run dropdowns.
Choose both run and original event number. The raw viewer, point clouds, and
track/trace images use the H5 file for that selected run. Two runs may have the
same event number; they never share the same event identity.

## Save a reusable polygon

1. Open Range vs Energy and set your current filters.
2. Choose Select Polygon Region and draw the polygon.
3. Click Save Cut and choose a JSON filename.

The default location is `cut_images/cut_definitions` below the processed-run
output directory. A cut file contains exactly these fields:

```json
{
  "experiment": "e25058",
  "source_runs": [145],
  "vertices": [[0.7, 20.0], [0.9, 20.0], [0.9, 35.0], [0.7, 35.0]]
}
```

The coordinates in this example are illustrative. Actual coordinates come from
your drawn polygon, in the same calibrated energy/range coordinates as the plot.

No veto, timing, range, angle, or energy filters are written into a reusable cut.
For a cut drawn on a combined plot, source_runs contains all loaded source runs.

## Apply that cut to other runs

Load the new run(s), then click Load Cut and select the saved JSON. The polygon
is restored and shown in the selection plot. Current GUI filter values are not
changed by loading the cut. A cut is rejected if its experiment does not match.
The original source_runs information is retained when a loaded cut is saved or
exported again. Editing the polygon creates a cut attributed to the currently
loaded runs.

A polygon also stays available when you load a different set of runs during the
same GUI session, within the same experiment. A saved JSON persists after exit.
Filter widgets reset to their defaults when a new set of tabs is constructed;
you can adjust them for the new runs. Loading a JSON itself does not reset them.

Both selection counts and image export use:

    events inside polygon AND current GUI filters

Export re-evaluates the current filters, so change them freely and replot to see
the same selection you will export. Project Cut to X/Y uses that same selection.
Histogram binning is taken from both bin-count boxes; the log/linear controls work
for ordinary and polygon plots.

## Export images

Click Save Polygon Images and choose the export parent folder. The normal parent
is the existing `cut_images` directory. Each run with selected events gets its own
folder named only by run and date:

    cut_images/run145_2026_10_18/
    cut_images/run147_2026_10_18/

Images retain their original identities, for example:

    run145_event221498.png
    run147_event221498.png

Each folder also contains:

- cut_definition.json: polygon geometry and original experiment/source runs.
- selected_events.csv: all selected run/event identities and expected filenames.
- cut_indices_H5list.pkl: original event numbers actually exported for THIS run.
- export_status.json: selected/written counts, skipped empty events, completion
  status, and an error if image generation failed.

A run-specific RvE preview JPG is saved beside the run/date folder. Previous Cuts
browses these previews and their PNGs, including after restarting the GUI. Use
Browse Export Folder there if you exported to a different parent directory.
Opening an empty preview directory does not crash the GUI.

Existing folders or previews are never overwritten. The exporter checks ALL run
names before creating output. To export a second cut on the same run/date, choose
a different parent directory (for example, a directory you create for that named
cut). This keeps the requested run/date-only folder names without random IDs.

Events without hits above the existing image threshold are skipped by the original
image builder. They remain listed in selected_events.csv and export_status.json,
while the pickle list contains only events whose PNGs were written. A failure can
leave a partial folder with status `failed`; it is not reported as a completed
export. Use only `run*_event*.png` for CNN image input.

The previous-cut H5-copy button is retained as Copy Selected H5 Datasets. It now
uses the loaded source run's actual H5 file and supports bucketed raw data. Its
result is a sparse set of selected datasets for inspection, not a full contiguous
run suitable for process_tpc_run. Load that source run before using this tool.

## Validation and first run

19 automated checks passed using synthetic detector input. They exercise the
actual single-run image builder, collection routing, current-filter selection,
cut JSON save/load, PNG writes, per-run manifests, duplicate event numbers,
missing-run handling, export collision protection, spectrum data, event-viewer
routing, and tab rebuilding. Exported images were inspected and remain 224 x 224
RGBA with the existing pad-plane plus trace representation. Existing pad/trace
construction and its threshold/calibration settings were not changed.

Run the checks without opening a desktop or detector files:

```bash
python -m unittest discover -s tests -v
```

The test module substitutes the external detector backend and unavailable viewer
imports; it does not need ROOT/H5 data. It also substitutes a progress bar if tqdm
is unavailable. Application dependencies otherwise remain those of your current
framework.

A full interactive Tk session and real TPC data were not available in the build
environment. First try one processed run, then two runs, and export a small cut.
The on-screen layout and detector-data results still need that smoke check on
your machine.
