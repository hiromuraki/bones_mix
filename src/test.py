import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).absolute().parent / "MotionAGFormer"))
sys.path.append(str(Path(__file__).absolute().parent / "MHFormer"))

# 这么做是为了避免代码格式化时重排 import 语句。MotionAGFormer 路径的加入必须提前，否则后续导入会报错
if True:
    from core import MotionAGFormerInferencer, RTMPoseInferencer, MHFormerInferencer
    from common import DataConverter, VideoUtils, Serializer

TEST_DATA_DIR = Path(os.getenv("TEST_DATA_DIR", ""))
TEST_OUTPUT_DIR = Path(os.getenv("TEST_OUTPUT_DIR", ""))

rtmpose = RTMPoseInferencer()
mhformer = MHFormerInferencer(
    weight_path=Path("MHFormer/checkpoint/pretrained/351/model_4294.pth"),
    window=351,
)
motion_ag_former = MotionAGFormerInferencer(
    config_path=Path("MotionAGFormer/configs/h36m/MotionAGFormer-base.yaml"),
    weight_path=Path("MotionAGFormer/checkpoint/motionagformer-b-h36m.pth.tr"),
    window=243,
    stride=243 // 2,
)


def test_inference_pipeline(video_name: str):
    input_video = TEST_DATA_DIR / f"{video_name}.mp4"
    video_size = VideoUtils.get_video_size(input_video)
    out_dir = TEST_OUTPUT_DIR / f"{video_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ========
    # RTMPose
    # ========
    kp_2d = rtmpose.run_2d_keypoints_inference(input_video)
    kp_2d_coco17 = kp_2d[0]
    kp_2d_h36m = DataConverter.coco_to_h36m_with_confidence(kp_2d[0])
    print(kp_2d_coco17[0])  # 打印第一帧第一个人的 2D 关键点和置信度，进行 sanity check
    print(kp_2d_h36m[0])  # 打印第一人的第一帧 2D 关键点，看看转换结果

    Serializer.save_keypoints_to_pickle(
        kp_2d_coco17, out_dir / "2d_kp_rtmpose_coco17.pkl"
    )
    Serializer.save_keypoints_to_pickle(kp_2d_h36m, out_dir / "2d_kp_rtmpose_h36m.pkl")

    # ========
    # MHFormer
    # ========
    kp_3d_mhformer = mhformer.run_3d_keypoints_inference(
        kp_2d_h36m,
        video_width=video_size[0],
        video_height=video_size[1],
        stride=351 // 2,
    )
    Serializer.save_keypoints_to_pickle(
        kp_3d_mhformer, out_dir / "3d_kp_rtmpose_to_mhformer.pkl"
    )

    # ==============
    # MotionAGFormer
    # ==============
    kp_3d_motionagformer = motion_ag_former.run_3d_keypoints_inference(
        kp_2d_h36m,
        video_width=video_size[0],
        video_height=video_size[1],
    )
    Serializer.save_keypoints_to_pickle(
        kp_3d_motionagformer, out_dir / "3d_kp_rtmpose_to_motionagformer.pkl"
    )


for video_file in TEST_DATA_DIR.glob("*.mp4"):
    video_name = video_file.stem
    test_inference_pipeline(video_name)
