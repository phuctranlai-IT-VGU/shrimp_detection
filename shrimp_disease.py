import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm", ".m4v"}


def predict_disease(
	model: YOLO,
	frame: np.ndarray,
	bbox: tuple[int, int, int, int],
	device: str,
) -> tuple[str, float | None]:
	frame_height, frame_width = frame.shape[:2]
	x1, y1, x2, y2 = bbox
	x1 = max(0, min(x1, frame_width))
	y1 = max(0, min(y1, frame_height))
	x2 = max(0, min(x2, frame_width))
	y2 = max(0, min(y2, frame_height))
	if x2 <= x1 or y2 <= y1:
		return "Unknown", None

	crop = frame[y1:y2, x1:x2]
	result = model(crop, device=device, verbose=False)[0]
	if result.probs is not None:
		class_id = int(result.probs.top1)
		return str(result.names[class_id]), float(result.probs.top1conf)

	if result.boxes is not None and len(result.boxes) > 0:
		best_index = int(result.boxes.conf.argmax().item())
		class_id = int(result.boxes.cls[best_index].item())
		confidence = float(result.boxes.conf[best_index].item())
		return str(result.names[class_id]), confidence

	return "No disease detected", None


def resolve_device(requested: str) -> str:
	if requested == "cpu":
		return "cpu"
	if requested == "gpu":
		if not torch.cuda.is_available():
			raise RuntimeError("Đã chọn GPU nhưng máy không có CUDA khả dụng.")
		return "cuda:0"
	return "cuda:0" if torch.cuda.is_available() else "cpu"


def draw_disease_label(
	frame: np.ndarray,
	bbox: tuple[int, int, int, int],
	model: YOLO,
	device: str,
) -> None:
	disease, confidence = predict_disease(model, frame, bbox, device)
	label = f"Disease: {disease}"
	if confidence is not None:
		label += f" {confidence:.2f}"
	x1, _, _, y2 = bbox
	cv2.putText(
		frame,
		label,
		(x1, min(frame.shape[0] - 8, y2 + 22)),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.6,
		(0, 165, 255),
		2,
		cv2.LINE_AA,
	)


def process_image(
	input_path: Path,
	output_path: Path,
	metadata_path: Path,
	model: YOLO,
	device: str,
) -> None:
	image = cv2.imread(str(input_path))
	if image is None:
		raise RuntimeError(f"Không đọc được ảnh đầu vào: {input_path}")
	with metadata_path.open("r", encoding="utf-8") as metadata_file:
		boxes = json.load(metadata_file)["boxes"]
	for box in boxes:
		draw_disease_label(
			image, tuple(int(value) for value in box), model, device
		)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	if not cv2.imwrite(str(output_path), image):
		raise RuntimeError(f"Không thể lưu ảnh kết quả: {output_path}")


def process_video(
	input_path: Path,
	output_path: Path,
	metadata_path: Path,
	model: YOLO,
	device: str,
) -> None:
	capture = cv2.VideoCapture(str(input_path))
	if not capture.isOpened():
		raise RuntimeError(f"Không mở được video: {input_path}")

	fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
	width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
	output_path.parent.mkdir(parents=True, exist_ok=True)
	writer = cv2.VideoWriter(
		str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
	)
	if not writer.isOpened():
		capture.release()
		raise RuntimeError(f"Không thể tạo video kết quả: {output_path}")

	frame_number = 0
	try:
		with metadata_path.open("r", encoding="utf-8") as metadata_file:
			while True:
				success, frame = capture.read()
				if not success:
					break
				line = metadata_file.readline()
				if not line:
					raise RuntimeError("Metadata không đủ frame cho video đầu vào.")
				for box in json.loads(line):
					draw_disease_label(
						frame, tuple(int(value) for value in box), model, device
					)
				writer.write(frame)
				frame_number += 1
				if frame_number % 30 == 0:
					print(f"Đang nhận diện bệnh: {frame_number} frame", flush=True)
			if metadata_file.readline():
				raise RuntimeError("Metadata có nhiều frame hơn video đầu vào.")
	finally:
		capture.release()
		writer.release()

	print(f"Đã xử lý {frame_number} frame bệnh")


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Nhận diện bệnh từ output ảnh/video đã detect tôm."
	)
	parser.add_argument("input_path", type=Path, help="Output ảnh/video từ bước detect tôm")
	parser.add_argument(
		"--detections",
		type=Path,
		default=None,
		help="File tọa độ box do bước detect tôm tạo ra",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=None,
		help="Đường dẫn output cuối cùng",
	)
	parser.add_argument(
		"--disease-model",
		type=Path,
		default=Path(__file__).with_name("shrimp_disease.pt"),
		help="Đường dẫn weights nhận diện bệnh",
	)
	parser.add_argument(
		"--device",
		choices=("auto", "gpu", "cpu"),
		default="auto",
		help="Thiết bị chạy model bệnh",
	)
	args = parser.parse_args()
	input_path = args.input_path.expanduser()
	model_path = args.disease_model.expanduser()
	if not input_path.is_file():
		raise FileNotFoundError(f"Không tìm thấy output bước detect tôm: {input_path}")
	if not model_path.is_file():
		raise FileNotFoundError(f"Không tìm thấy model bệnh: {model_path}")

	extension = input_path.suffix.lower()
	is_image = extension in IMAGE_EXTENSIONS
	if not is_image and extension not in VIDEO_EXTENSIONS:
		raise ValueError(f"Định dạng không được hỗ trợ: {extension}")
	metadata_path = (
		args.detections.expanduser()
		if args.detections is not None
		else input_path.with_suffix(".json" if is_image else ".jsonl")
	)
	if not metadata_path.is_file():
		raise FileNotFoundError(f"Không tìm thấy metadata box: {metadata_path}")
	output_path = (
		args.output.expanduser()
		if args.output is not None
		else input_path.with_name(f"{input_path.stem}_disease{input_path.suffix}")
	)
	device = resolve_device(args.device)
	model = YOLO(str(model_path))
	if is_image:
		process_image(input_path, output_path, metadata_path, model, device)
	else:
		process_video(input_path, output_path, metadata_path, model, device)
	print(f"Thiết bị nhận diện bệnh: {device}")
	print(f"Output cuối cùng: {output_path}")


if __name__ == "__main__":
	main()
