# Installation

## Windows

Install Python 3.10 or newer, open PowerShell in this directory, then run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

The project-local environment used for this prototype is Python 3.12 with OpenCV 4.14 and NumPy 2.5. The camera test on the build machine found no webcam device; camera unavailability is reported by the app. To use the custom detector training or inference pipeline, install the optional `torch` and `pillow` packages separately; they are intentionally not required for the baseline MVP.

## Offline use

After dependencies are installed, inference and the dashboard are local. To move to an offline machine, install the same Python version and transfer locally built wheels or this environment's compatible wheel cache; no model or camera footage is downloaded by the program. User checkpoints/datasets must be transferred separately.
