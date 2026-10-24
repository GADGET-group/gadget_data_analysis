# gadget_analysis

Analysis code for GADGET II experiments at FRIB. GADGET II combines a gas time projection chamber (P10 gas, a
1024-pad plane read out by GET electronics with four CoBos) with a decay station read out by DDAS digitizers:
CLARION germanium clovers, beam PID detectors (scintillators, silicon and PIN detectors, PPACs) and logic signals (mesh preamp,
accepted GET trigger, beam on/off, chopper on/off, SCA, veto). The beam is implanted in the gas, and charged-particle
decays (beta-delayed protons, alphas) are recorded as tracks in the TPC while the clovers see the gammas.

The code turns raw GET h5 files into per-event TPC quantities (energy, track length, direction, PID), merges the DDAS
event-builder output into ROOT trees with one entry per DDAS event, matches GET events to DDAS triggers so TPC and
decay-station data can be combined in one `TTree::Draw` expression, and provides cached histograms, germanium gain
matching and add-back, and peak fitting. It is used for e23035 (60Ga and 59Zn, November 2025) and e25058; older
experiments (e21072, e23035_prep_*) still appear in the path tables.

## Repository layout

| folder | contents |
|---|---|
| `tpc_data_processing/` | `raw_h5_file.py` (GET h5 reader, baseline subtraction, pad geometry from `padxy.txt`, `PadPlane.csv`, `channel_mappings/flatlookup4cobos.csv`), `process_runs.py` (per-run processing to ROOT files: `process_tpc_run`, `ensure_processed`, `get_quantity`, `get_save_path`, `get_h5_path`), `tpc_processing_configs/*.csv`, `pad_gain_match/` (pad gain maps in `gain_match_results/`), `plots/` (`rve.py` and other plotting scripts), `field_dist/`, `h5_utils/`, `veto_conditioner/`, `dev/`, `heritage_h5_file.py`, `check_cobos_present.py`, `RawEventViewerFrame.py`, `file_formats.txt` |
| `tpc_and_decay_station/` | `ddas_interface.py` (merging DDAS files with TPC events, histogram cache), `degai.py` (CLARION add-back, gamma-gamma, timing), `get_tsbo.py` (time since beam off per GET event), `mesh_gamma_coincidence.py`, `clarion_det_locations.csv`, `degai-angles-table.pdf` |
| `analysis_tools/` | `energy_calibration_tools.py`, `fitting_tools.py`, `root_vis_tools.py`, `spectrum_fitter.py` |
| `main_gui/` | tkinter analysis GUI and raw event viewer |
| `e23035_analysis/`, `e25058_analysis/` | one folder per experiment: `runs.csv`, `channel_map.csv`, `<exp>_runs.py`, `__init__.py`, the experiment's own scripts, and git-ignored caches (`hist_cache/`, `clarion_cache/`, `calibrations/`, `nonlinearity_corrections/`) |
| `fit_gui/` | fit GUI frames |
| `track_fitting/` | track simulation, SRIM tables (`stopping_powers/`), field-distortion studies, `build_sim.get_gas_density` |
| `detector_efficiency/` | efficiency simulations |
| repo root | `setup.py` (distribution `gadget_analysis`, `find_packages()`), `exp_env.py` (the owner's interactive driver script), `.gitignore` |

## Installation

Use a conda env with PyROOT and the scientific stack (the `e23035` env runs Python 3.13 and ROOT 6.36):

```bash
conda create -n myexp -c conda-forge python=3.13 root numpy scipy pandas matplotlib uproot awkward h5py \
    numba scikit-image scikit-learn tqdm dill threadpoolctl
conda activate myexp
# optional, GPU baseline fitting (raw_h5_file falls back to the CPU without them)
conda install -c conda-forge cupy
pip install torch torch-ransac3d   # the e23035 env has torch 2.5.1+cu121; match your CUDA (check)
# GUI only
pip install scikit-spatial
```

Install the repository in editable mode from its root:

```bash
cd /egr/research-tpc/adamsa52/gadget_analysis
pip uninstall -y e23035_analysis e25058_analysis   # old distribution names, if the env has them
pip install -e .
```

Rules:

- Re-run `pip install -e .` after adding any new top-level package (for example a new `<exp>_analysis/` with its
  `__init__.py`). The editable finder's package map is written at install time and does not see new folders.
- One env per checkout. The editable install binds package names (`tpc_data_processing`, `e23035_analysis`, ...) to
  one checkout's folders, so the main checkout uses env `e23035` and the e25058 worktree uses env `gadget_analysis`.
- Activate the env (`conda activate`, or `conda run -n <env> ...`). Calling the env's python by its full path without
  activation breaks PyROOT's JIT.
- Run scripts and sessions from the repo root. Some paths are still relative to the working directory: the
  `flat_lookup_file_path` rows of the processing configs, `track_fitting/stopping_powers/...` in the runs modules.

## Data locations on tpcgpu

| data | path |
|---|---|
| raw GET h5 | `/egr/research-tpc/shared/experiments/<exp>/h5/run_NNNN.h5` (e25058: `newh5/`) |
| processed TPC data | `/egr/research-tpc/shared/proc_runs/<exp>/proc_pkl/` (e25058: `new_h5/`) |
| DDAS data | `/egr/research-tpc/shared/proc_runs/<exp>/ddas/` |

These come from `process_runs.get_h5_path`, `process_runs.get_save_path` and `ddas_interface.get_root_file_path`. A
new experiment name falls into their `else` branches and gets the paths above with `<exp>` replaced.

| file | folder | made by | contents |
|---|---|---|---|
| `run_NNNN.h5` | `h5/` | GET merger (outside this repo) | `[meta][meta]` event range and timestamps; `['get'][evt#_data]` N x 517 arrays (CoBo, AsAd, AGET, channel, pad, 512 samples); see `file_formats.txt` |
| `<exp>_run<GET>_<config>.root` | `proc_pkl/` | `process_runs.process_tpc_run` | trees `events` (per-event quantities) and `metadata` (settings hash and JSON, git state) |
| `*.npy` | `proc_pkl/` | `process_runs` helpers | `gm_ic_`, `veto_mask_`, `veto_counts_`, `max_veto_counts_`, `outer_ring_counts_`, `max_outer_ring_counts_` per run, named with settings and gain/threshold hashes |
| `<exp>_run<GET>_get_tsbo_v2.root` | `proc_pkl/` | `get_tsbo` | tree `events`: `time_since_beam_off`, `ddas_run`, `ddas_entry` per GET event |
| `run-NNNN.root` | `ddas/` | DDAS event builder (outside this repo) | tree `tree`: 160-channel arrays `energies`, `times`, `multiplicity` |
| `run<DDAS>_merged_v2.root`, `run<DDAS>_merge_v2.log` | `ddas/` | `ddas_interface.make_ddas_root_file` | tree `merged_data`: per channel `<name>_e` (keV), `_t` (s), `_m`, `_c` (ADC), `_cr` (ADC + uniform [-0.5, 0.5]); plus `time_since_beam_off`, `time_since_chopper_off`, `rebuilt_window`, `anchor_t`, `n_source_entries`, `mesh_in_window` |
| `run<DDAS>_tpc_<config>_v2.root`, `.log` | `ddas/` | `ddas_interface.make_tpc_friend_file` | tree `tpc_data`, one entry per merged entry: `tpc_energy` (keV), `tpc_track_length` (mm), `tpc_particle_id`, `tpc_should_veto`, `tpc_track_angle` (deg), `get_timestamp`, `tpc_track_centroid[3]`, `get_event_id`, `get_run_id`; provenance as `TNamed` objects |

The `_v2` suffix is `ddas_interface.MERGED_VERSION` (the trigger-anchored rebuild of October 2026). Files without it
are the legacy merged files.

## Processing pipeline

`exp` is the experiment string (`'e23035'`), `cfg` a config file name such as `'smart1_veto2_rpr_fzp.csv'`. Every
library function takes `experiment` first.

| step | call | reads | writes |
|---|---|---|---|
| 0. config | pick `tpc_data_processing/tpc_processing_configs/<cfg>` | | |
| 1. TPC processing | `process_runs.ensure_processed(exp, get_runs, config_filename=cfg)` (parallel over GPUs) or `process_runs.process_tpc_run(exp, run, cfg)` | raw h5 | `<exp>_run<GET>_<config>.root` |
| 1b. TPC quantities | `process_runs.get_quantity(name, exp, get_runs, config_filename=cfg)`; runs module `get_energy_MeV`, `get_length_mm`, masks | processed ROOT | `.npy` caches |
| 2. DDAS merge | `ddas_interface.make_ddas_root_file(exp, ddas_run)`, made on first use by `get_histogram` and `degai` | `run-NNNN.root`, `channel_map.csv` | `run<N>_merged_v2.root` + log |
| 3. TPC friend tree | `ddas_interface.make_tpc_friend_file(exp, ddas_run, tpc_ini_filename)`, made on first use when `tpc_*`, `get_timestamp`, `get_event_id` or `get_run_id` appear in an expression (`tpc_ini_filename` is then required) | merged file, runs module, processed ROOT | `run<N>_tpc_<config>_v2.root` + log |
| 3b. time since beam off | `get_tsbo.get_time_since_beam_off(exp, get_runs, tpc_ini_filename=cfg)` | merged files and friend trees (never made here) | `<exp>_run<GET>_get_tsbo_v2.root` |
| 4. histograms | `ddas_interface.get_histogram(exp, ddas_run, binning, hist_name, hist_title, var_exp, selection="", force_recreate=False, num_workers=1, tpc_ini_filename="")`; `ddas_run` may be a list (summed) | merged (+ friend) | `<exp>_analysis/hist_cache/h_<md5>.root` |
| 5. gamma gain match | copy of `e23035_analysis/align_gamma_energies.py`, which calls `energy_calibration_tools.make_energy_calibration` on the `clover_*_cr` branches | merged files | `<exp>_analysis/calibrations/<run>/<cal>/<branch>/` |
| 6. add-back, coincidences | `degai.get_addback_tree`, `degai.get_histogram`, `degai.get_addback_coincidence_spectrum`, `get_gated_projection`, `get_bg_subtracted_projection` | merged files, calibrations | `<exp>_analysis/clarion_cache/` |
| 7. peak fitting | `analysis_tools.spectrum_fitter` (`spectrum_fitter`, `multi_spectrum_fitter`), `analysis_tools.fitting_tools` | histograms | your choice |

e23035's TPC proton and alpha fit stack (`stepwise_fit.py`, `fit_tpc_spectrum_simultaneous.py`, `fit_ledger.py`,
`fit_diagnostics.py`) stays in `e23035_analysis/` and builds on `analysis_tools`.

`get_quantity` names: `track_center`, `principle_axes`, `variance_along_axes`, `pad_charge`, `endpoints`,
`charge_width`, `width_above_threshold`, `pad_max`, `timestamps`, `railed_pads` (events tree) and `settings_hash`,
`settings_json`, `git_version`, `git_status`, `git_diff` (metadata tree). Arrays are per GET event, runs concatenated
in the order given. Derived helpers: `get_lengths`, `get_angle`, `get_veto_mask`, `get_gm_ic`,
`get_run_and_event_numbers`.

Processing configs (CSV rows `name,type,value`, read by `raw_h5_file.load_config`). The config name is part of every
output file name, so a changed setting needs a new config file:

| config | baseline (`background_subtract_mode`) | railed-pad repair | zero-pad removal | veto pads |
|---|---|---|---|---|
| `smart1_rpr.csv` | `smart` | yes | no | same as pads |
| `smart1_rpr_fzp.csv` | `smart` | yes | yes | same as pads |
| `smart1_veto2_rpr_fzp.csv` (current standard) | `smart` | yes | yes | `smart2` |
| `smart2_rpr.csv` | `smart2` | yes | no | same as pads |
| `smart2_rpr_fzp.csv` | `smart2` | yes | yes | same as pads |
| `smart2_nrpr.csv` | `smart2` | no | no | same as pads |

`rpr` = `reconstruct_railed_pads`, `fzp` = `remove_zero_pad`, `veto2` = `veto_background_subtract_mode smart2`.
`smart2` uses torch RANSAC and needs about 13 GB of GPU memory per worker; `smart1` runs on the CPU.

tpcgpu is shared: keep total CPU use below about 200 of its 256 cores, and check peak memory per worker times the
number of workers against free RAM before a large parallel launch.

### Worked session

```python
# from the repo root, env activated
from tpc_data_processing import process_runs
from tpc_and_decay_station import ddas_interface, degai

exp = 'e23035'
cfg = 'smart1_veto2_rpr_fzp.csv'

print(ddas_interface.validate_experiment(exp) or 'experiment folder OK')

runs = ddas_interface.get_runs_module(exp)
df = runs.run_df
ddas_run = 241                                    # a 60Ga run, GET run 263
get_runs = df['GET'][(df['DDAS'] == ddas_run) & df['GET'].notna()].astype(int).to_numpy()

process_runs.ensure_processed(exp, get_runs, config_filename=cfg)
energy_MeV = runs.get_energy_MeV(get_runs, tpc_ini_filename=cfg)
endpoints = process_runs.get_quantity('endpoints', exp, get_runs, config_filename=cfg)

# proton spectrum from the friend tree (made automatically on first use)
h_p = ddas_interface.get_histogram(exp, ddas_run, (500, 0, 5000), 'h_p', 'protons;TPC energy (keV);counts',
                                   'tpc_energy', 'tpc_particle_id==1', tpc_ini_filename=cfg)
h_p.Draw()

# germanium add-back with geometric neighbours within 30 degrees, using the gain match gm_511and2614_1
adj = degai.get_adjacency_dict(exp, 30)
h_ab = degai.get_histogram(exp, ddas_run, adj, 'gm_511and2614_1', (4000, 0, 4000), 'h_ab',
                           'add-back;E (keV);counts', 'addback_energy')
# proton-gated add-back spectrum
h_ab_p = degai.get_histogram(exp, ddas_run, adj, 'gm_511and2614_1', (4000, 0, 4000), 'h_ab_p',
                             'add-back, protons;E (keV);counts', 'addback_energy', 'tpc_particle_id==1',
                             tpc_ini_filename=cfg)
```

`degai.get_histogram` behaves like `ddas_interface.get_histogram` with the add-back tree as an extra friend
(`addback_energy` and `time` branches; defaults `dt_window_ns=200`, `e_thresh=150`). The add-back needs a
calibration under `cal_name` for every crystal of that run. Other adjacency choices: `degai.get_clover_adj_dict(exp)`
(same clover only) and `degai.get_crystal_adj_dict(exp)` (no add-back).

## The experiment folder

`<exp>_analysis/` holds `__init__.py`, `runs.csv`, `channel_map.csv`, `<exp>_runs.py` and the experiment's own
scripts. Library code never imports it at import time: `ddas_interface.get_experiment_dir(exp)` finds
`<repo>/<exp>_analysis`, `ddas_interface.read_channel_map(exp)` reads `channel_map.csv`, and
`ddas_interface.get_runs_module(exp)` imports `<exp>_analysis.<exp>_runs`.

### runs.csv

One row per GET/DDAS run pairing. `GET` is blank for DDAS-only runs (background, sources). A DDAS run that spans
several GET runs has several rows; a GET run may also span several DDAS runs. `GET` and `DDAS` are what the library
uses; `degrader angle`, `Run Type`, `Field Cage Functional?` and `source type` are used by the e23035 run selectors and
`align_gamma_energies.py`. The e23035 header:

```
GET,DDAS,Start Time,End Time,Duration,Run Type,LN2 fill?,H5 on TPC GPU,Field Cage Functional?,SCA Lower Limit,degrader angle,good tpc data?,final beam settings?,source type,source location,Comments
```

### channel_map.csv

```
Index,Name,Slope,Offset
118, mesh_pre_amp, 1.000000, 0.000000
119, get_trig_accepted, 1.000000, 0.000000
48, clover_1a, 0.190140, 5.332810
```

- The header line is skipped; rows are split on `", "` (comma followed by one space), so keep that spacing.
- `Index` is the DDAS channel, 16 x module + channel (0-159), the position in the event builder's 160-channel arrays.
- `Name` becomes the branch prefix in `merged_data`. Required: `mesh_pre_amp` (118), `get_trig_accepted` (119),
  `beam_off` (123), `chopper_off` (125); these numbers are `ddas_interface.CH_MAP`. PID helpers use `msx100`,
  `cross_scint_b2`, `db_5_scint`.
- Germanium crystals are `clover_<n><a-d>`: `n` is the CLARION detector number (1-24, as in
  `clarion_det_locations.csv`), `a`-`d` are crystals 1-4. `degai` builds its crystal list from these names.
- `Slope`, `Offset` give keV = slope x ADC + offset, applied to `_e` at merge time and used to seed the gain-match
  peak search. Slope 1 means uncalibrated (as for e25058 clovers 12 and 13). `_c` and `_cr` are always raw ADC, and a
  slope change reaches `_e` only after the merged file is deleted and remade.

### Runs-module interface

Required (called by library code with `get_runs` = array of GET run numbers and the keyword `tpc_ini_filename=`;
arrays are per GET event, runs concatenated in `process_runs` order):

- `run_df`: `pandas.DataFrame` read from `runs.csv`, one row per GET/DDAS pairing; numeric columns `GET` and `DDAS`
  (`pd.to_numeric(errors='coerce')`, NaN when absent). A DDAS run spanning several GET runs has several rows.
- `get_energy_MeV(get_runs, tpc_ini_filename="")`: gain-matched TPC energy in MeV; the friend tree stores it x1000
  as `tpc_energy` (keV).
- `get_length_mm(get_runs, tpc_ini_filename="")`: track length (mm).
- `get_veto_mask(get_runs, tpc_ini_filename="")`: bool, True = keep; stored inverted as `tpc_should_veto`.
- `get_proton_mask(...)`, `get_alpha_mask(...)`: bool; `tpc_particle_id` = 1 proton, 2 alpha, 0 other, -1 unmatched.
- `get_pad_gains(tpc_ini_filename="")`: per-pad gains used by `get_energy_MeV`; hashed into the friend-tree
  provenance.

Optional:

- `experiment = '<name>'`
- `get_pad_gain_path(tpc_ini_filename="")` (provenance string)
- `bad_crystals = [(clover:int, 'a'..'d'), ...]` (default `[]`)

Conventions: paths relative to `Path(__file__).parent`, never the working directory; import-time work limited to
reading `runs.csv`; no `degai` calls at import time (import cycle).

### validate_experiment

`ddas_interface.validate_experiment(exp)` returns a list of problems, empty when the folder is usable. It checks the
logic channels in `channel_map.csv`, the runs-module attributes, the `run_df` columns and the `degai` crystal list.
Run it after every change to the experiment folder.

## Caches and when they are rebuilt

| cache | location | rebuilt automatically when its inputs change? |
|---|---|---|
| processed TPC ROOT file | `proc_pkl/<exp>_run<GET>_<config>.root` | no: made once per config name; `get_quantity` raises if the config's settings hash differs from the file's; use `process_tpc_run(..., force_reprocess=True)` |
| `process_runs` `.npy` caches | `proc_pkl/*.npy` | yes (settings and gain/threshold hashes in the names) |
| merged file | `ddas/run<N>_merged_v2.root` | no: made when missing; delete it after editing `channel_map.csv` or the merge constants |
| TPC friend tree | `ddas/run<N>_tpc_<config>_v2.root` | no: made when missing; delete it after changing pad gains, masks, `runs.csv` pairings or the merged file |
| time since beam off | `proc_pkl/<exp>_run<GET>_get_tsbo_v2.root` | no: written only when all its DDAS runs have merged and friend files; delete it after remaking them |
| `ddas_interface.get_histogram` | `<exp>_analysis/hist_cache/` | yes (key has run, binning, expressions, config name and input-file fingerprints) |
| `energy_calibration_tools.get_calibrated_histogram` | `<exp>_analysis/hist_cache/` | no (key has no fingerprint); pass `force_recreate=True` |
| add-back trees, add-back histograms | `<exp>_analysis/clarion_cache/add_back_tree/`, `histograms_flex/` | after a re-merge yes (merged-file fingerprint); after recalibrating under the same `cal_name` no |
| gamma-gamma matrices, timing spectra | `<exp>_analysis/clarion_cache/histograms/`, `timing/` | no |

The `.gitignore` excludes `/*_analysis/{calibrations,clarion_cache,hist_cache,nonlinearity_corrections}/` for every
experiment folder. It also excludes `*.pkl` and `*.root`, so a new pad-gain map needs `git add -f`.

## GUIs

All need an X display.

| command | what |
|---|---|
| `python -m main_gui` | tkinter analysis GUI (run selection, energy spectrum, range vs energy, individual events) |
| `python -m main_gui.RawDataGUI` | raw event viewer (`tpc_data_processing/RawEventViewerFrame.py`) for GET h5 files |
| `python -m main_gui.RawDataGUI -old` | raw event viewer for heritage e17023 files (`heritage_h5_file.py`) |

`main_gui/The_GADGET_GUI.py` is the legacy GUI. Loading a run in the analysis GUI is currently broken (see Known
issues); the raw event viewer is the working one.

## Setting up a new experiment

1. Create `<exp>_analysis/` with an empty `__init__.py`.
2. Write `runs.csv` (columns `GET`, `DDAS` at least, plus whatever your run selectors need).
3. Write `channel_map.csv` from the DDAS cabling: logic channels at the `CH_MAP` numbers, PID detectors, every
   germanium crystal as `clover_<n><a-d>`, with pre-experiment slopes and offsets where known.
4. Copy `e25058_analysis/e25058_runs.py` to `<exp>_analysis/<exp>_runs.py` and edit:
   - `experiment`;
   - veto thresholds in `get_veto_mask`;
   - PID bands in `get_proton_mask_min_max_range` and `get_alpha_mask_min_max_range` (SRIM range plus margins);
   - `get_pad_gain_path` / `GAIN_MATCH_DIR` (start from the e23035 maps in
     `tpc_data_processing/pad_gain_match/gain_match_results/`; a new map is made with the scripts in
     `pad_gain_match/` (check));
   - `bad_crystals`;
   - the run selectors (`get_ddas_runs` is a placeholder);
   - `get_pid_cut` polygons and variables.
5. Add a branch for the experiment to `track_fitting/build_sim.get_gas_density` (it asserts for unknown names).
6. Data paths: put the raw h5 files in `/egr/research-tpc/shared/experiments/<exp>/h5/` and the event-builder files in
   `/egr/research-tpc/shared/proc_runs/<exp>/ddas/`, and create `/egr/research-tpc/shared/proc_runs/<exp>/proc_pkl/`.
   Other hosts need a branch in `process_runs.get_save_path` and `get_h5_path`.
7. `pip install -e .` so the new package is found.
8. `ddas_interface.validate_experiment('<exp>')` until it returns `[]`.
9. Process one GET run (`process_runs.process_tpc_run('<exp>', run, cfg)`), then look at
   `get_energy_MeV` and `get_length_mm` (for example a range vs energy plot with `tpc_data_processing/plots/rve.py`)
   and check the energy scale on a known line.
10. Merge one DDAS run with `ddas_interface.make_ddas_root_file('<exp>', ddas_run)` and read
    `run<N>_merge_v2.log`. The last line has the counters. In e23035 run 241, `mesh_in_window` is 22159 of 31543
    `triggers` (70 %; low-energy events leave no mesh hit above the DDAS threshold); a much lower fraction means the
    mesh delay or gate is wrong for this setup. Measure the delay with
    `get_histogram(exp, run, (200, 0, 10e-6), 'dt', 'trigger - mesh', 'get_trig_accepted_t - mesh_pre_amp_t', 'get_trig_accepted_m>=1 && mesh_pre_amp_m>=1')`.
11. First histograms: `ddas_interface.get_histogram(exp, run, (8000, 0, 8000), 'h', 'clover 1a', 'clover_1a_e')`
    for a crystal, then `tpc_energy` with `tpc_ini_filename=cfg`. The friend log
    (`run<N>_tpc_<config>_v2.log`) reports `GET run N: matched X of Y GET events`; a low fraction means a GET/DDAS
    pairing problem in `runs.csv`.
12. Gamma gain match: copy `e23035_analysis/align_gamma_energies.py` into your folder and adapt `experiment`, the run
    selection (`runs_to_align_desc`), the calibration peaks (`gm_label`; the name becomes `gm_<label>_<degree>`, for
    e23035 `gm_511and2614_1`, `gm_511and1301_1`, `gm_1461and2614_1`) and the non-linearity peak list. Then call
    `do_gain_match(run)` on one run and look at `calibrations/<run>/<gm_name>/gain_match.pdf`, then `process_all()`,
    `make_summary_pdf()` and optionally `make_nonlinearity_correction()` (e23035 correction `c1`).
13. Add-back and coincidences with `degai` as in the worked session.

## Branches and worktrees

| checkout | branch | env |
|---|---|---|
| `/egr/research-tpc/adamsa52/gadget_analysis` | `e23035` | `e23035` |
| `/egr/research-tpc/adamsa52/gadget_analysis_e25058` (git worktree) | `e25058-run-processing` | `gadget_analysis` |

The worktree is e25058's checkout, shared with the e25058 students, and is kept current by merging `e23035` into
it. Library changes go to `e23035` first. A new experiment can live on either branch; it needs its own env if it gets
its own checkout.

## Experiment-specific constants outside the experiment folder

| constant | where | what to do |
|---|---|---|
| `MESH_DELAY_US = 4.34`, `MESH_GATE_US = (3.0, 6.0)`, `REBUILD_WINDOW_US = 15.0` | `tpc_and_decay_station/ddas_interface.py` (kwargs `mesh_delay_us`, `mesh_gate_us`, `window_us` of `make_ddas_root_file`) | measure the trigger minus mesh delay for your setup (step 10) |
| `CH_MAP` logic channel numbers | `ddas_interface.py` | must match the cabling and `channel_map.csv` |
| `show_pid` PID expression and ranges | `ddas_interface.py` | e23035 values (`msx100_e` vs `cross_scint_b2_t - db_5_scint_t`) |
| `get_save_path`, `get_h5_path` | `tpc_data_processing/process_runs.py` | hostname and experiment tables; the `else` branches give the tpcgpu paths, other hosts need a branch |
| `zscale=1.088` | `process_runs.get_h5_file` and the `zscale` row of each config | mm per time bucket; set from drift speed and sampling clock |
| `get_gas_density` | `track_fitting/build_sim.py` | branch per experiment (P10 at 860.3 torr for e23035 and e25058) |
| `VETO_PADS`, `OUTER_RING_PADS` | `raw_h5_file.py`, `process_runs.py` | pad-plane geometry; unchanged for the GADGET II pad plane |
| `TPC_PROCESSING_GPUS` and worker defaults | `process_runs.py` | host specific (four GPUs on tpcgpu) |

## Known issues

- The analysis GUI's "Load" fails: `main_gui/GadgetRunH5.py` calls `raw_h5_file()` without the required `zscale`,
  and the `pca` and `BaselineRemoval` PyPI packages it imports are missing from the envs.
- `main_gui/IndividualEventFrame.py` references a nonexistent `h5_file_path`.
- `fit_gui/FitElementFrame.py` loads `p10_alpha_850torr.npy` relative to the working directory (the file is in
  `fit_gui/`).
- `process_runs.get_save_path` returns None on unknown hosts; on `gadget*` hosts `get_h5_path` tests for `'23035'`
  instead of `'e23035'`.
- The runs modules use `ROOT` in `get_pid_cut` without importing it; `e23035_runs.get_counts_in_pid_cut` calls
  `get_ddas_root_file_path` without the module or the experiment.
- The runs modules open `track_fitting/stopping_powers/...` relative to the working directory.
- `exp_env.py` passes `False` as the TPC config, which `process_tpc_run` rejects.
- `degai.get_addback_tree` writes straight to its cache path; an interrupted or concurrent build can leave a
  truncated tree that later calls reuse. Delete it from `clarion_cache/add_back_tree/` if a spectrum looks short.
- `get_tsbo` never makes merged files or friend trees: GET events of a DDAS run without them get NaN and no cache is
  written.
- `detector_efficiency/DoubleAlphaTrackSim.py` does not parse: it contains committed merge-conflict markers (line 160).
- `main_gui/CNN_Frame_test.py` (unused by the GUI) needs `torchvision`, which the envs do not have.
