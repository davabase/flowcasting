# SEVIR radar viewer

Install dependencies and open the interactive viewer in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe view_sevir.py
```

Choose an **Example** (0–1871), then select **Frames** such as `0,3,8`
or `0:12:2` and click **Apply selection**. Ranges exclude the stop index.
Alternatively, enter **N frames** and click **Show first N**.
**Save PNG** saves the current view into `renders/`.

The default dataset, `IN_vil`, contains 13 input radar frames per example.
Change **Dataset** to `OUT_vil` for the corresponding 12 future target frames.
Frame numbers are local to each dataset and start at zero. Colors show encoded
SEVIR values on a fixed 0–255 scale for radar, not rainfall in physical units.
Other available channels are printed at startup; their colors use the minimum
and maximum of the selected frames.

Render six frames, or export a specific selection without opening a window:

```powershell
.\.venv\Scripts\python.exe view_sevir.py --sample 10 -n 6
.\.venv\Scripts\python.exe view_sevir.py --sample 10 --frames 0,3,8,12 --output renders/example10.png
.\.venv\Scripts\python.exe view_sevir.py --dataset OUT_vil --sample 10 --frames 0:12 --output renders/targets10.png
```

Use `--file PATH` for another HDF5 file with datasets shaped
`(examples, height, width, frames)` and `--columns N` to change the grid layout.
Only selected frames are read into memory. The viewer limits grids to 64 frames.

# FlowCast: learning to simulate radar

FlowCast is a proof-of-concept neural radar simulator: given the last 13 VIL frames (65 minutes),
it predicts the next 12 frames (60 minutes) and can be rolled out autoregressively for hours.
It is scored against ground truth on the official SEVIR nowcast test set alongside persistence and
an optical-flow (DIS + semi-Lagrangian) baseline.

## Pipeline

```powershell
python download_sevir_test.py                  # official test set: 1,872 windows, Jun-Nov 2019 (5 GB)
python download_sevir_train.py --budget-gb 40  # raw VIL training events (resumable byte-range fetch)
bash train_supervised.sh --run runs/flowcast --steps 20000 --duty 0.75 --vram-gb 2.5
python evaluate.py                             # metrics for persistence, optical flow and FlowCastNet
python rollout.py                              # 3-hour autoregressive simulation on held-out events
python make_figures.py                         # figures and animations in results/figures
python build_report.py                         # self-contained report in results/report
```

**Training data.** `download_sevir_train.py` reads the SEVIR catalog and selects every storm event plus a
random sample of "random" events dated before 2019-06-01, so nothing overlaps the test period. The raw
SEVIR VIL files store an uncompressed, contiguous `(N, 384, 384, 49)` uint8 array, so each 4-hour event is
fetched with a single HTTP range request instead of downloading whole 10-17 GB files. Events are written
time-first to `data/sevir/vil_train/events.npy` with a manifest (`events.csv`) and an event-level
train/validation split.

**Model.** `flowcast/model.py` is a ConvNeXt U-Net in a 4x space-to-depth latent space. For every lead time it
predicts a displacement field, which advects the latest frame (a learned semi-Lagrangian step), and an
intensity residual for growth and decay. Both heads start at zero, so the untrained model is exactly persistence.

**Shared GPU.** `train.py` caps its own VRAM (`--vram-gb`), can throttle itself (`--duty`), and checkpoints
every `--ckpt-every` steps; `train_supervised.sh` resumes automatically if another process's memory spike
kills a step.

**Metrics.** CSI/POD/SUCC/BIAS are computed from hits, misses and false alarms summed over the whole test set
at the SEVIR thresholds 16, 74, 133, 160, 181 and 219 (CSI-M is their mean), plus max-pooled CSI at 4 and 16 km
and MSE/MAE on 0-1 scaled VIL. These follow the Earthformer convention, so the published numbers are comparable.
