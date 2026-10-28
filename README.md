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
