# Shrimp Detection

YOLO-based shrimp detection for images and videos. The launcher automatically selects the correct detector and can run on CPU or NVIDIA CUDA GPU.

## Project layout

- `main.py`: routes an image to `model1.py` and a video to `model2.py`.
- `model1.py`: detects shrimp in one image.
- `model2.py`: detects shrimp in video, draws confidence boxes, tracks objects between detections, and reports progress/FPS.
- `best.pt`: trained YOLO26.
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

Place the trained `best.pt` beside the Python files. The code uses this project-relative location, so it does not depend on `C:\python`.

## Run

Automatically choose GPU when CUDA is available, otherwise use CPU:

```powershell
python main.py path\to\image.jpg
python main.py path\to\video.mp4
```

Force a device:

```powershell
python main.py path\to\video.mp4 --device gpu
python main.py path\to\video.mp4 --device cpu
```

The video detector writes `<input>_detected.mp4` beside the input video. Image results are written to `output_image/`.

## Notes

- `--device auto` is the recommended public default.
- `--device gpu` requires a CUDA-enabled NVIDIA PyTorch installation.
- Tune `CONFIDENCE`, `DETECT_INTERVAL`, and `IMAGE_SIZE` in `model2.py` for the target hardware and video quality.
