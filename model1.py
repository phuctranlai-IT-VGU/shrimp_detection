import argparse
import sys
from pathlib import Path

import torch
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "best.pt"
OUTPUT_DIR = PROJECT_DIR / "output_image"
DEVICE = "cpu"


parser = argparse.ArgumentParser(description="Detect tôm trong ảnh.")
parser.add_argument("image_path", type=Path, help="Đường dẫn ảnh đầu vào")
parser.add_argument(
	"--device",
	choices=("auto", "gpu", "cpu"),
	default="auto",
	help="Thiết bị detect: auto (mặc định), gpu hoặc cpu",
)
args = parser.parse_args()
if args.device == "gpu" and not torch.cuda.is_available():
	raise RuntimeError("Đã chọn GPU nhưng máy không có CUDA khả dụng.")
DEVICE = "cuda:0" if args.device != "cpu" and torch.cuda.is_available() else "cpu"

IMAGE_PATH = args.image_path.expanduser()
OUTPUT_PATH = OUTPUT_DIR / f"{IMAGE_PATH.stem}_detected{IMAGE_PATH.suffix}"


if not IMAGE_PATH.is_file():
	raise FileNotFoundError(f"Không tìm thấy ảnh đầu vào: {IMAGE_PATH}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
model = YOLO(str(MODEL_PATH))
results = model(str(IMAGE_PATH), conf=0.10, iou=0.45, device=DEVICE, verbose=False)
results[0].save(filename=str(OUTPUT_PATH))

print(f"Thiết bị detect: {DEVICE}")
print(f"Ảnh kết quả đã được lưu tại: {OUTPUT_PATH}")
