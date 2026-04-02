import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent / "MotionAGFormer"))
sys.path.append(str(Path(__file__).absolute().parent / "MHFormer"))

# 这么做是为了避免代码格式化时重排 import 语句。MotionAGFormer 路径的加入必须提前，否则后续导入会报错
if True:
    from core import MotionAGFormerInferencer, RTMPoseInferencer, MHFormerInferencer
    from common import DataConverter, VideoUtils, VideoRenderer


for i in ["1", "2", "3", "4"]:
    video_name = f"sample_{i}"
    input_video = Path(f"/home/user/bones/sample_data/{video_name}.mp4")
    video_size = VideoUtils.get_video_size(input_video)
    out_dir = Path(f"/home/user/bones/sample_output/{video_name}")

    kp_2d = RTMPoseInferencer().run_2d_keypoints_inference(input_video)
    kp_2d_coco17 = kp_2d[0]
    kp_2d_h36m = DataConverter.coco_to_h36m_with_confidence(kp_2d[0])
    print(kp_2d_coco17[0])  # 打印第一帧第一个人的 2D 关键点和置信度，进行 sanity check
    print(kp_2d_h36m[0])  # 打印第一人的第一帧 2D 关键点，看看转换结果

    # RTMPose
    VideoRenderer.render_2d_keypoints(input_video, kp_2d_h36m, "h36m", Path(out_dir / f"2d_kp_rtmpose_h36m.mp4"))
    VideoRenderer.render_2d_keypoints(input_video, kp_2d_coco17, "coco17", Path(out_dir / f"2d_kp_rtmpose_coco17.mp4"))

    # MHFormer
    kp_3d_mhformer = MHFormerInferencer(
        weight_path=Path("MHFormer/checkpoint/pretrained/351/model_4294.pth"),
        window=351,
    ).run_3d_keypoints_inference(
        kp_2d_h36m,
        video_width=video_size[0],
        video_height=video_size[1],
        stride=351//2
    )

    VideoRenderer.render_3d_keypoints(kp_3d_mhformer,
                                      flip_x=True,
                                      flip_y=True,
                                      output_video_path=out_dir / f"3d_kp_rtmpose_to_mhformer.mp4")

    # MotionAGFormer
    kp_3d_motionagformer = MotionAGFormerInferencer(
        config_path=Path("MotionAGFormer/configs/h36m/MotionAGFormer-base.yaml"),
        weight_path=Path("MotionAGFormer/checkpoint/motionagformer-b-h36m.pth.tr"),
        window=243,
    ).run_3d_keypoints_inference(
        kp_2d_h36m,
        video_width=video_size[0],
        video_height=video_size[1],
        stride=243//2
    )

    VideoRenderer.render_3d_keypoints(kp_3d_motionagformer,
                                      flip_x=True,
                                      flip_y=False,
                                      output_video_path=out_dir / f"3d_kp_rtmpose_to_motionagformer.mp4")