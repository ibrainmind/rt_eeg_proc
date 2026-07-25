# rt_eeg_proc

Utilities for real-time EEG/ECG processing experiments and dataset inspection.

## Process ds005873 (MNE-based)

The script [src/process_data.py](src/process_data.py) uses `mne` to load EDF files and plot aligned signals for one run:

- Top subplot: EEG channel 1
- Middle subplot: EEG channel 2
- Bottom subplot: ECG/EKG channel(s)

It is modular, so you can import its functions in other Python files, and it also provides an argparse CLI.

## Environment

Activate the local conda environment:

```bash
source scripts/activate_conda_env.sh
```

If needed, install dependencies into that env:

```bash
MAMBA_ROOT_PREFIX="$PWD/.mamba" ./.tools/micromamba install -y -p "$PWD/conda-env" -c conda-forge mne numpy matplotlib datalad git-annex
```

## CLI usage

Generate and save a figure for `sub-001`, `ses-01`, `run-01`:

```bash
./conda-env/bin/python src/process_data.py \
	--dataset-root datasets/ds005873 \
	--subject sub-001 \
	--session ses-01 \
	--run 01 \
	--duration-sec 20 \
	--out outputs/sub-001_run-01_eeg_ecg.png
```

Show interactive figure instead of saving:

```bash
./conda-env/bin/python src/process_data.py \
	--dataset-root datasets/ds005873 \
	--subject sub-001 \
	--session ses-01 \
	--run 01 \
	--show
```

Run PI tracker comparison on EKG (buffered vs causal real-time):

```bash
./conda-env/bin/python src/process_data.py \
	--dataset-root datasets/ds005873 \
	--subject sub-001 \
	--session ses-01 \
	--run 01 \
	--plot-tracker \
	--duration-sec 20 \
	--out outputs/sub-001_run-01_tracker_compare.png
```

Notes:

- `--pull/--no-pull` controls whether missing run files are fetched with `datalad get`.
- `--start-sec` and `--duration-sec` control the plotted time window.
- `--plot-tracker` switches to PI tracker diagnostics using ECG/EKG with time-linked axes for zooming.

## Import in other Python code

```python
from pathlib import Path
from src.process_data import load_run_data, plot_run_data

run_data = load_run_data(
		dataset_root=Path("datasets/ds005873"),
		subject="sub-001",
		session="ses-01",
		run="01",
		pull=False,
)

plot_run_data(
		run_data,
		start_sec=0.0,
		duration_sec=30.0,
		out_path=Path("outputs/example.png"),
		show=False,
)
```