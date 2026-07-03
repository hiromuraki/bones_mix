import sys
import pickle
import time
from pathlib import Path

sys.path.append(str(Path(__file__).absolute().parent / "MotionAGFormer"))
sys.path.append(str(Path(__file__).absolute().parent / "MHFormer"))

# 这么做是为了避免代码格式化时重排 import 语句。MotionAGFormer 路径的加入必须提前，否则后续导入会报错
if True:
    from core import MHFormerInferencer
    from common import VideoUtils


def benchmark_mhformer_inference(mhformer: MHFormerInferencer, keypoints_2d_h36m, video_size):
    frame_count = int(keypoints_2d_h36m.shape[0])

    warmup_start_time = time.perf_counter()
    mhformer.run_3d_keypoints_inference(
        keypoints_2d_h36m[: min(frame_count, mhformer.window)],
        video_width=video_size[0],
        video_height=video_size[1],
    )
    warmup_elapsed = time.perf_counter() - warmup_start_time

    inference_start_time = time.perf_counter()
    kp_3d_mhformer = mhformer.run_3d_keypoints_inference(
        keypoints_2d_h36m,
        video_width=video_size[0],
        video_height=video_size[1],
    )
    inference_elapsed = time.perf_counter() - inference_start_time

    fps = frame_count / inference_elapsed if inference_elapsed > 0 else 0.0
    latency_per_frame_ms = (inference_elapsed / frame_count * 1000) if frame_count > 0 else 0.0
    latency_per_clip_ms = inference_elapsed * 1000

    return {
        "总帧数": f"{frame_count} 帧",
        "输出关键点形状": f"{tuple(kp_3d_mhformer.shape)}",
        "预热耗时": f"{warmup_elapsed * 1000:.2f} ms",
        "正式推理总耗时": f"{latency_per_clip_ms:.2f} ms",
        "平均单帧延迟": f"{latency_per_frame_ms:.2f} ms/帧",
        "平均吞吐率": f"{fps:.2f} 帧/秒",
        "窗口大小": f"{mhformer.window} 帧",
        "滑动步长": f"{mhformer.stride} 帧",
    }

TEST_DATA_DIR = Path("/home/user/bones/sample_data")
TEST_OUTPUT_DIR = Path("/home/user/bones/temp")

video_name = "微信视频_20260402224013"
input_video = TEST_DATA_DIR / f"{video_name}.mp4"
video_size = VideoUtils.get_video_size(input_video)
out_dir = TEST_OUTPUT_DIR / f"{video_name}"
out_dir.mkdir(parents=True, exist_ok=True)
input_keypoints_path = Path(R"/home/user/bones/sample_output/微信视频_20260402224013/2d_kp_rtmpose_h36m.pkl")

if not input_keypoints_path.exists():
    raise FileNotFoundError(
        f"找不到 H36M 2D 关键点文件: {input_keypoints_path}"
    )

with open(input_keypoints_path, "rb") as file:
    kp_2d_h36m = pickle.load(file)

mhformer = MHFormerInferencer(
    weight_path=Path("MHFormer/checkpoint/pretrained/351/model_4294.pth"),
    window=351,
    stride=60,
)

mhformer_metrics = benchmark_mhformer_inference(
    mhformer,
    kp_2d_h36m,
    video_size,
)

print("MHFormer 性能测试结果：")
for metric_name, metric_value in mhformer_metrics.items():
    print(f"  - {metric_name}：{metric_value}")
