import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "best.pt"
DETECT_INTERVAL = 1
MAX_MISSED_DETECTIONS = 3
CONFIDENCE = 0.25
MIN_TRACK_CONFIDENCE = 0.25
IOU_THRESHOLD = 0.45
PROGRESS_INTERVAL = 30
IMAGE_SIZE = 640
DEVICE = "cpu"


@dataclass
class Track:
	track_id: int
	bbox: tuple[int, int, int, int]
	confidence: float
	class_id: int
	missed_detections: int = 0


def resolve_device(requested: str) -> str:
	if requested == "cpu":
		return "cpu"
	if requested == "gpu":
		if not torch.cuda.is_available():
			raise RuntimeError("Đã chọn GPU nhưng máy không có CUDA khả dụng.")
		return "cuda:0"
	return "cuda:0" if torch.cuda.is_available() else "cpu"


def box_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
	left = max(first[0], second[0])
	top = max(first[1], second[1])
	right = min(first[2], second[2])
	bottom = min(first[3], second[3])
	intersection = max(0, right - left) * max(0, bottom - top)
	first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
	second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
	union = first_area + second_area - intersection
	return intersection / union if union else 0.0


def detect_objects(model: YOLO, frame) -> list[tuple[tuple[int, int, int, int], float, int]]:
	result = model(
		frame,
		conf=CONFIDENCE,
		iou=IOU_THRESHOLD,
		imgsz=IMAGE_SIZE,
		device=DEVICE,
		verbose=False,
	)[0]
	detections = []
	for box, confidence, class_id in zip(
		result.boxes.xyxy.cpu().numpy(),
		result.boxes.conf.cpu().numpy(),
		result.boxes.cls.cpu().numpy(),
	):
		x1, y1, x2, y2 = (int(value) for value in box)
		confidence_value = float(confidence)
		if confidence_value >= MIN_TRACK_CONFIDENCE:
			detections.append(((x1, y1, x2, y2), confidence_value, int(class_id)))
	return detections


def update_tracks(
	tracks: list[Track],
	detections: list[tuple[tuple[int, int, int, int], float, int]],
	next_id: int,
) -> int:
	detections = [
		detection
		for detection in detections
		if detection[1] >= MIN_TRACK_CONFIDENCE
	]
	unmatched_tracks = set(range(len(tracks)))
	for detection_box, confidence, class_id in sorted(
		detections, key=lambda detection: detection[1], reverse=True
	):
		best_index = None
		best_iou = 0.25
		for index in unmatched_tracks:
			track = tracks[index]
			if track.class_id != class_id:
				continue
			current_iou = box_iou(track.bbox, detection_box)
			if current_iou > best_iou:
				best_iou = current_iou
				best_index = index

		if best_index is None:
			tracks.append(Track(next_id, detection_box, confidence, class_id))
			next_id += 1
		else:
			track = tracks[best_index]
			track.bbox = detection_box
			track.confidence = confidence
			track.class_id = class_id
			track.missed_detections = 0
			unmatched_tracks.remove(best_index)

	for index in unmatched_tracks:
		tracks[index].missed_detections += 1
	tracks[:] = [
		track
		for track in tracks
		if (
			track.missed_detections <= MAX_MISSED_DETECTIONS
			and track.confidence >= MIN_TRACK_CONFIDENCE
		)
	]
	return next_id


def update_by_optical_flow(tracks: list[Track], previous_frame, frame) -> None:
	if not tracks:
		return

	previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
	current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	points = []
	for track in tracks:
		x1, y1, x2, y2 = track.bbox
		points.extend(((x1, y1), (x2, y1), (x2, y2), (x1, y2)))

	point_array = np.array(points, dtype="float32").reshape(-1, 1, 2)
	new_points, status, _ = cv2.calcOpticalFlowPyrLK(
		previous_gray, current_gray, point_array, None
	)
	if new_points is None or status is None:
		return

	for index, track in enumerate(tracks):
		point_status = status[index * 4 : index * 4 + 4].reshape(-1)
		if point_status.sum() < 3:
			continue
		moved_points = new_points[index * 4 : index * 4 + 4].reshape(-1, 2)
		valid_points = moved_points[point_status.astype(bool)]
		x_values = valid_points[:, 0]
		y_values = valid_points[:, 1]
		track.bbox = (
			max(0, int(x_values.min())),
			max(0, int(y_values.min())),
			int(x_values.max()),
			int(y_values.max()),
		)


def draw_tracks(frame, tracks: list[Track]) -> None:
	for track in tracks:
		x1, y1, x2, y2 = track.bbox
		label = f"Confidence: {track.confidence:.2f}"
		cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
		cv2.putText(
			frame,
			label,
			(x1, max(20, y1 - 8)),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.6,
			(0, 255, 0),
			2,
			cv2.LINE_AA,
		)


def main() -> None:
	parser = argparse.ArgumentParser(description="Detect tôm trong video.")
	parser.add_argument("video_path", type=Path, help="Đường dẫn video đầu vào")
	parser.add_argument(
		"--device",
		choices=("auto", "gpu", "cpu"),
		default="auto",
		help="Thiết bị detect: auto (mặc định), gpu hoặc cpu",
	)
	args = parser.parse_args()
	global DEVICE
	DEVICE = resolve_device(args.device)

	video_path = args.video_path.expanduser()
	if not video_path.is_file():
		raise FileNotFoundError(f"Không tìm thấy video đầu vào: {video_path}")

	output_path = video_path.with_name(f"{video_path.stem}_detected.mp4")
	model = YOLO(str(MODEL_PATH))
	capture = cv2.VideoCapture(str(video_path))
	if not capture.isOpened():
		raise RuntimeError(f"Không mở được video: {video_path}")

	fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
	total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
	width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
	writer = cv2.VideoWriter(
		str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
	)
	if not writer.isOpened():
		capture.release()
		raise RuntimeError(f"Không thể tạo video đầu ra: {output_path}")
	print(
		f"Bắt đầu render: {video_path} | Tổng số frame: "
		f"{total_frames if total_frames > 0 else 'không xác định'}",
		flush=True,
	)
	device_name = (
		torch.cuda.get_device_name(0) if DEVICE.startswith("cuda") else "CPU"
	)
	print(
		f"Thiết bị: {device_name} ({DEVICE}) | Model: {MODEL_PATH}",
		flush=True,
	)

	tracks: list[Track] = []
	next_id = 1
	previous_frame = None
	frame_number = 0
	start_time = time.perf_counter()
	try:
		while True:
			success, frame = capture.read()
			if not success:
				break

			if frame_number % DETECT_INTERVAL == 0 or previous_frame is None:
				detections = detect_objects(model, frame)
				next_id = update_tracks(tracks, detections, next_id)
			else:
				update_by_optical_flow(tracks, previous_frame, frame)

			draw_tracks(frame, tracks)
			writer.write(frame)
			previous_frame = frame.copy()
			frame_number += 1
			if frame_number % PROGRESS_INTERVAL == 0 or (
				total_frames > 0 and frame_number == total_frames
			):
				elapsed = max(time.perf_counter() - start_time, 1e-6)
				processing_fps = frame_number / elapsed
				if total_frames > 0:
					percent = frame_number / total_frames * 100
					progress = (
						f"Loading: {percent:.1f}% | "
						f"Frame: {frame_number}/{total_frames}"
					)
				else:
					progress = f"Frame: {frame_number}"
				print(
					f"{progress} | Tốc độ: {processing_fps:.2f} FPS",
					end="\r",
					flush=True,
				)
	finally:
		capture.release()
		writer.release()

	print()
	elapsed = max(time.perf_counter() - start_time, 1e-6)
	print(f"Tốc độ trung bình: {frame_number / elapsed:.2f} FPS")
	print(f"Đã xử lý {frame_number} frame")
	print(f"Video kết quả đã được lưu tại: {output_path}")


if __name__ == "__main__":
	main()
