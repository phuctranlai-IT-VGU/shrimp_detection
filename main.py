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
    args = parser.parse_args()

    input_path = args.input_path.expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file đầu vào: {input_path}")

    extension = input_path.suffix.lower()
    if extension in IMAGE_EXTENSIONS:
        script_name = "model1.py"
    elif extension in VIDEO_EXTENSIONS:
        script_name = "model2.py"
    else:
        supported = ", ".join(sorted(IMAGE_EXTENSIONS | VIDEO_EXTENSIONS))
        raise ValueError(
            f"Định dạng không được hỗ trợ: {extension or '(không có phần mở rộng)'}"
            f". Hỗ trợ: {supported}"
        )

    script_path = Path(__file__).with_name(script_name)
    subprocess.run(
        [sys.executable, str(script_path), str(input_path), "--device", args.device],
        check=True,
    )


if __name__ == "__main__":
    main()