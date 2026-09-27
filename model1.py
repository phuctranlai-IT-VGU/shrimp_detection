import argparse
import json
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
parser.add_argument(
	"--output",
	type=Path,
	default=None,
	help="Đường dẫn ảnh output sau khi detect tôm",
)
args = parser.parse_args()
if args.device == "gpu" and not torch.cuda.is_available():
	raise RuntimeError("Đã chọn GPU nhưng máy không có CUDA khả dụng.")
DEVICE = "cuda:0" if args.device != "cpu" and torch.cuda.is_available() else "cpu"

IMAGE_PATH = args.image_path.expanduser()
OUTPUT_PATH = (
	args.output.expanduser()
	if args.output is not None
	else OUTPUT_DIR / f"{IMAGE_PATH.stem}_detected{IMAGE_PATH.suffix}"
)


if not IMAGE_PATH.is_file():
	raise FileNotFoundError(f"Không tìm thấy ảnh đầu vào: {IMAGE_PATH}")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
model = YOLO(str(MODEL_PATH))
result = model(str(IMAGE_PATH), conf=0.10, iou=0.45, device=DEVICE, verbose=False)[0]
result.save(filename=str(OUTPUT_PATH))
boxes = (
	[[int(value) for value in box] for box in result.boxes.xyxy.cpu().numpy()]
	if result.boxes is not None
	else []
)
with OUTPUT_PATH.with_suffix(".json").open("w", encoding="utf-8") as metadata_file:
	json.dump({"boxes": boxes}, metadata_file)

print(f"Thiết bị detect: {DEVICE}")
print(f"Ảnh kết quả đã được lưu tại: {OUTPUT_PATH}")
