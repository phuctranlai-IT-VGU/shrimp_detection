# Shrimp Detection

YOLO-based shrimp detection for images and videos. The launcher automatically selects the correct detector and can run on CPU or NVIDIA CUDA GPU.

## Project layout

- `main.py`: runs shrimp detection first, then passes its output to `shrimp_disease.py`.
- `model1.py`: detects shrimp in one image and saves box metadata beside the annotated output.
- `model2.py`: detects/tracks shrimp in video and saves per-frame box metadata beside the annotated output.
- `best.pt`: shrimp detection weights.
- `shrimp_disease.pt`: shrimp disease weights.



## Setup on Windows

Use Python 3.10+ and create a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

For an NVIDIA GPU, install a CUDA-enabled PyTorch build that matches your driver before running the detector. For example:

```powershell
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

Verify CUDA:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Place `best.pt` and `shrimp_disease.pt` beside the Python files. The code uses project-relative defaults, so it does not depend on a specific installation directory.

## Run

Automatically choose GPU when CUDA is available, otherwise use CPU:

```powershell
python main.py path\to\image.jpg
python main.py path\to\video.mp4
```

The pipeline runs in two steps. First, `model1.py` or `model2.py` creates an annotated shrimp-detection output and box metadata. Then `shrimp_disease.py` reads that output, uses the metadata to crop each shrimp, and adds the predicted disease label.

Use a different disease model file with:

```powershell
python main.py path\to\image.jpg --disease-model path\to\disease.pt
```

Force a device:

```powershell
python main.py path\to\video.mp4 --device gpu
python main.py path\to\video.mp4 --device cpu
```

Intermediate files end in `_detected`; final files end in `_detected_disease`. Intermediate box metadata is saved as `.json` for images and `.jsonl` for videos. Images and their results are written to `output_image/`; video results and metadata are written to `output_video/`.

## Notes

- `--device auto` is the recommended public default.
- `--device gpu` requires a CUDA-enabled NVIDIA PyTorch installation.
- Tune `CONFIDENCE`, `DETECT_INTERVAL`, and `IMAGE_SIZE` in `model2.py` for the target hardware and video quality.
