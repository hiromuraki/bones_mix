import os
from pathlib import Path
from typing import Any
from VideoJoiner import VideoJoiner
from VideoRenderer import VideoRenderer
import pickle

TEST_DATA_DIR = Path(os.getenv("TEST_DATA_DIR", ""))
TEST_OUTPUT_DIR = Path(os.getenv("TEST_OUTPUT_DIR", ""))


def load_keypoints(file_path: Path) -> Any:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        print(f"已加载数据: {data.shape}")
        return data


def render_video(video_name: str):
    input_video = TEST_DATA_DIR / f"{video_name}.mp4"
    out_dir = TEST_OUTPUT_DIR / f"{video_name}"

    # 渲染 2D 关键点
    VideoRenderer.render_2d_keypoints(
        video_path=input_video,
        keypoints_2d=load_keypoints(out_dir / "2d_kp_rtmpose_h36m.pkl"),
        bones_mode="h36m",
        output_video_path=Path(f"{out_dir}/2d_kp_rtmpose_h36m.mp4")
    )
    VideoRenderer.render_2d_keypoints(
        video_path=input_video,
        keypoints_2d=load_keypoints(out_dir / "2d_kp_rtmpose_coco17.pkl"),
        bones_mode="coco17",
        output_video_path=Path(f"{out_dir}/2d_kp_rtmpose_coco17.mp4")
    )

    # 渲染 3D 重建结果
    VideoRenderer.render_3d_keypoints(
        load_keypoints(out_dir / "3d_kp_rtmpose_to_mhformer.pkl"),
        flip_x=True,
        flip_y=True,
        output_video_path=out_dir / f"3d_kp_rtmpose_to_mhformer.mp4"
    )

    VideoRenderer.render_3d_keypoints(
        load_keypoints(out_dir / "3d_kp_rtmpose_to_motionagformer.pkl"),
        flip_x=True,
        flip_y=False,
        output_video_path=out_dir / f"3d_kp_rtmpose_to_motionagformer.mp4"
    )

    # 合并视频
    VideoJoiner.concat_videos(
        [
            (out_dir / f"2d_kp_rtmpose_h36m.mp4", "2D (RTMPose)"),
            (out_dir / f"3d_kp_rtmpose_to_mhformer.mp4", "3D (MHFormer)"),
            (out_dir / f"3d_kp_rtmpose_to_motionagformer.mp4", "3D (MotionAGFormer)"),
        ],
        output_video_file=TEST_OUTPUT_DIR / f"{video_name}_output.mp4"
    )


for video_file in TEST_DATA_DIR.glob("*.mp4"):
    video_name = video_file.stem
    render_video(video_name)
