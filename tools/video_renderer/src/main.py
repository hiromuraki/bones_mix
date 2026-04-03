from pathlib import Path
import pickle
from typing import Any
from VideoRenderer import VideoRenderer


def load_keypoints(file_path: Path) -> Any:
    with open(file_path, "rb") as f:
        data = pickle.load(f)
        print(f"已加载数据: {data.shape}")
        return data


if __name__ == "__main__":
    VideoRenderer.render_2d_keypoints(
        video_path=Path("/home/user/bones/sample_data/sample_1.mp4"),
        keypoints_2d=load_keypoints(
            Path("/home/user/bones/tools/yolov8_benchmark/yolo_keypoints.pkl")
        )[0],
        bones_mode="h36m",
        output_video_path=Path(
            "/home/user/bones/tools/yolov8_benchmark/yolo_keypoints.mp4"
        ),
    )
