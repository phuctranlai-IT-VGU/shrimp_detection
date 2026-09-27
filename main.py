import argparse
import subprocess
import sys
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm", ".m4v"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tự động detect ảnh/video bằng CPU hoặc GPU."
    )
    parser.add_argument("input_path", type=Path, help="Đường dẫn ảnh hoặc video")
    parser.add_argument(
        "--device",
        choices=("auto", "gpu", "cpu"),
        default="auto",
        help="Thiết bị detect: auto (mặc định), gpu hoặc cpu",
    )
    parser.add_argument(
        "--disease-model",
        type=Path,
        default=Path(__file__).with_name("shrimp_disease.pt"),
        help="Đường dẫn weights nhận diện bệnh của tôm",
    )
    args = parser.parse_args()

    input_path = Path(str(args.input_path).strip()).expanduser()
    disease_model_path = args.disease_model.expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file đầu vào: {input_path}")
    if not disease_model_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy model bệnh: {disease_model_path}")

    extension = input_path.suffix.lower()
    if extension in IMAGE_EXTENSIONS:
        script_name = "model1.py"
        intermediate_path = (
            Path(__file__).with_name("output_image")
            / f"{input_path.stem}_detected{input_path.suffix}"
        )
        metadata_path = intermediate_path.with_suffix(".json")
    elif extension in VIDEO_EXTENSIONS:
        script_name = "model2.py"
        intermediate_path = (
            Path(__file__).with_name("output_video")
            / f"{input_path.stem}_detected.mp4"
        )
        metadata_path = intermediate_path.with_suffix(".jsonl")
    else:
        supported = ", ".join(sorted(IMAGE_EXTENSIONS | VIDEO_EXTENSIONS))
        raise ValueError(
            f"Định dạng không được hỗ trợ: {extension or '(không có phần mở rộng)'}"
            f". Hỗ trợ: {supported}"
        )

    output_path = intermediate_path.with_name(
        f"{intermediate_path.stem}_disease{intermediate_path.suffix}"
    )
    script_path = Path(__file__).with_name(script_name)
    print(f"Bước 1/2: Detect tôm", flush=True)
    subprocess.run(
        [sys.executable, str(script_path), str(input_path), "--device", args.device,
         "--output", str(intermediate_path)],
        check=True,
    )
    print("Bước 2/2: Nhận diện bệnh từ output bước 1", flush=True)
    disease_script = Path(__file__).with_name("shrimp_disease.py")
    subprocess.run(
        [
            sys.executable,
            str(disease_script),
            str(intermediate_path),
            "--detections",
            str(metadata_path),
            "--output",
            str(output_path),
            "--device",
            args.device,
            "--disease-model",
            str(disease_model_path),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()