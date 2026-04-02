from typing import List, Dict
from pathlib import Path
from typing import Any, List, Dict
import cv2
import torch
from common import DataConverter, Serializer, VideoRenderer
import numpy as np
from mmpose.apis import MMPoseInferencer
from core import MotionAGFormerInferencer


def _build_instances(keypoints: Any, keypoint_scores: Any = None) -> List[Dict[str, np.ndarray]]:
    keypoints_array = np.asarray(keypoints, dtype=np.float32)
    if keypoints_array.ndim == 2:
        keypoints_array = keypoints_array[None, ...]
    if keypoints_array.ndim != 3:
        return []

    scores_array = None
    if keypoint_scores is not None:
        scores_array = np.asarray(keypoint_scores, dtype=np.float32)
        if scores_array.ndim == 1:
            scores_array = scores_array[None, ...]

    instances: List[Dict] = []
    for index, instance_keypoints in enumerate(keypoints_array):
        instance: Dict[str, np.ndarray] = {"keypoints": instance_keypoints}
        if scores_array is not None and index < len(scores_array):
            instance["keypoint_scores"] = scores_array[index]
        instances.append(instance)
    return instances


def _extract_instances(prediction: Any) -> List[Dict[str, np.ndarray]]:
    if prediction is None:
        return []

    if isinstance(prediction, dict):
        if "keypoints" in prediction:
            return _build_instances(prediction["keypoints"], prediction.get("keypoint_scores"))

        if "predictions" in prediction:
            return _extract_instances(prediction["predictions"])

        instances: List[Dict[str, np.ndarray]] = []
        for value in prediction.values():
            instances.extend(_extract_instances(value))
        return instances

    if isinstance(prediction, (list, tuple)):
        instances: List[Dict[str, np.ndarray]] = []
        for item in prediction:
            instances.extend(_extract_instances(item))
        return instances

    return []


def extract_2d_key_points(video_path: Path) -> List[List[Dict[str, np.ndarray]]]:
    """
    结构 [Frames, Persons, 17, 2]
    """
    input_video = Path(video_path).expanduser().resolve()
    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    print("正在初始化 RTMPose-l 和 RTMDet-m（首次运行可能需要下载权重）...")
    inferencer = MMPoseInferencer(det_model="rtmdet-m", pose2d="rtmpose-l")

    print(f"开始提取 2D 关键点: {input_video}")
    frame_predictions: List[List[Dict[str, np.ndarray]]] = []
    result_generator = inferencer(str(input_video), show=False, return_vis=False, save_predictions=False)

    for frame_index, result in enumerate(result_generator, start=1):
        frame_predictions.append(_extract_instances(result))
        if frame_index % 30 == 0:
            print(f"已提取 {frame_index} 帧关键点...")

    print(f"关键点提取完成，总帧数: {len(frame_predictions)}")
    return frame_predictions


def convert_to_h36m_format(keypoints_2d: List[List[Dict[str, np.ndarray]]]) -> List[List[Dict[str, np.ndarray]]]:
    h36m_keypoints_2d: List[List[Dict[str, np.ndarray]]] = []
    for frame_instances in keypoints_2d:
        h36m_frame_instances: List[Dict[str, np.ndarray]] = []
        for instance in frame_instances:
            coco_kp = instance["keypoints"]
            h36m_kp = DataConverter.coco_to_h36m(np.expand_dims(coco_kp, axis=0)).squeeze(0)
            h36m_frame_instances.append({"keypoints": h36m_kp})
        h36m_keypoints_2d.append(h36m_frame_instances)
    return h36m_keypoints_2d


def convert_to_h36m_format_batch(keypoints_2d: List[List[Dict[str, np.ndarray]]]) -> List[List[Dict[str, np.ndarray]]]:
    """
    全向量化的高效转换版本 (Batch Processing)
    假设每帧提取主体人物（index 0）进行 3D 升维。
    """
    frames_count = len(keypoints_2d)
    if frames_count == 0:
        return []

    # ==========================================
    # 步骤 1: 组装大矩阵 (抽取原油)
    # ==========================================
    # 准备一个空矩阵，形状 [Frames, 17, 2]
    batch_coco_kps = np.zeros((frames_count, 17, 2), dtype=np.float32)

    # 记录哪些帧真的有人，防止空帧干扰
    valid_frames = []
    for i, frame_instances in enumerate(keypoints_2d):
        if len(frame_instances) > 0:
            # 我们提取每帧的第一个人作为主体 (17, 2)
            batch_coco_kps[i] = frame_instances[0]["keypoints"][:17, :2]
            valid_frames.append(i)

    # ==========================================
    # 步骤 2: 一键批处理 (核心加速区)
    # ==========================================
    # 直接把 [Frames, 17, 2] 的大矩阵扔给铁桶转换器！
    # 底层 NumPy 会用 C++ 并行计算所有帧，没有任何 for 循环，速度极快！
    batch_h36m_kps = DataConverter.coco_to_h36m(batch_coco_kps)

    # ==========================================
    # 步骤 3: 重新装盒 (为了兼容现有的渲染器格式)
    # ==========================================
    h36m_keypoints_2d: List[List[Dict[str, np.ndarray]]] = []
    for i in range(frames_count):
        if i in valid_frames:
            # 把算好的单帧 (17, 2) 重新包成字典装进列表
            h36m_frame_instances = [{"keypoints": batch_h36m_kps[i]}]
        else:
            # 这一帧没人，保持空列表
            h36m_frame_instances = []

        h36m_keypoints_2d.append(h36m_frame_instances)

    return h36m_keypoints_2d


def convert_to_2d_h36m_tensor(keypoints_2d: List[List[Dict[str, np.ndarray]]], obj_index: int = 0) -> np.ndarray:
    """
    将 RTMPose 提取的嵌套字典数据，直接转换为 H36M 格式的 Tensor。
    输入: RTMPose 的原始输出列表 (包含 keypoints)
    输出: 形状为 [Frames, 17, 2] 的 NumPy 数组 (代表 x, y)
    """
    frames_count = len(keypoints_2d)
    if frames_count == 0:
        return np.zeros((0, 17, 2), dtype=np.float32)

    batch_coco_kps = np.zeros((frames_count, 17, 2), dtype=np.float32)

    # 记录哪些帧真的有人，防止空帧干扰
    valid_frames = []

    for i, frame_instances in enumerate(keypoints_2d):
        # 防止索引越界
        if len(frame_instances) > obj_index:
            # 提取指定人物作为主体 (17, 2)
            batch_coco_kps[i] = frame_instances[obj_index]["keypoints"][:17, :2]
            valid_frames.append(i)

    batch_h36m_kps = DataConverter.coco_to_h36m(batch_coco_kps)

    # 重新装盒
    h36m_tensor = np.zeros((frames_count, 17, 2), dtype=np.float32)
    if valid_frames:
        h36m_tensor[valid_frames] = batch_h36m_kps[valid_frames]

    return h36m_tensor


def generate_3d_key_points(keypoints_2d: np.ndarray, video_width: int = 1920, video_height: int = 1080) -> np.ndarray:
    """
    执行 2D 到 3D 的升维推理
    :param keypoints_2d: 形状为 [Frames, 17, 2] 的 H36M 格式 NumPy 数组
    :param video_width: 原始视频宽度 (用于坐标归一化)
    :param video_height: 原始视频高度 (用于坐标归一化)
    :return: 形状为 [Frames, 17, 3] 的 3D 关键点数组
    """
    print("🚀 开始 3D 升维推理 (MotionAGFormer)...")
    frames = keypoints_2d.shape[0]

    # ==========================================
    # 步骤 1: 数据预处理 (极其关键！决定了输出是人还是怪物)
    # ==========================================
    # 复制一份数据防污染
    processed_2d = keypoints_2d.copy()

    # 1.1 屏幕坐标归一化 -> 缩放到 [-1, 1] 之间
    processed_2d[:, :, 0] = (processed_2d[:, :, 0] / video_width) * 2.0 - 1.0
    processed_2d[:, :, 1] = (processed_2d[:, :, 1] / video_height) * 2.0 - 1.0

    # 1.2 根节点相对化 (Root-relative)
    # 3D 升维模型通常只关心“动作姿态”，不关心人站在屏幕的左边还是右边。
    # 所以我们需要把每一帧的所有点，都减去当帧的“骨盆(Pelvis, index=0)”坐标，让骨盆始终在 (0, 0)
    pelvis_2d = processed_2d[:, 0:1, :]  # 提取骨盆，保留维度为 [Frames, 1, 2]
    processed_2d = processed_2d - pelvis_2d

    # ==========================================
    # 步骤 2: 转换为 PyTorch Tensor 并送入 GPU
    # ==========================================
    # 转换类型并增加 Batch 维度: [Frames, 17, 2] -> [1, Frames, 17, 2]
    input_tensor = torch.from_numpy(processed_2d).float().unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to(device)

    # ==========================================
    # 步骤 3: 初始化并加载 MotionAGFormer
    # ==========================================
    # TODO: 这里的超参数 (n_layers, d_model等) 必须与你下载的预训练权重严格对齐！
    # MotionAGFormer 对序列长度有要求，通常预训练模型是基于 243, 81 或 27 帧训练的。
    '''
    model = MotionAGFormer(
        n_layers=12,
        d_model=256,
        n_frames=frames,  # 如果模型不支持动态帧数，这里可能需要做 padding(补齐) 或 chunking(分块)
        # ... 其他参数
    )

    # 加载预训练权重
    checkpoint = torch.load("path/to/motionagformer_weights.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict']) # 具体 key 看官方源码
    model.to(device)
    model.eval() # 开启推理模式，冻结 Dropout 和 BatchNorm
    '''

    # ==========================================
    # 步骤 4: 前向推理 (黑盒魔法)
    # ==========================================
    '''
    with torch.no_grad(): # 推理时不计算梯度，省显存提速度
        # 输出的形状通常是 [1, Frames, 17, 3]
        pred_3d_tensor = model(input_tensor)
    '''

    # --- ⚠️ 模拟输出 (为了让你现在的代码不报错能跑通，我先伪造一个输出) ---
    print(f"🔧 [模拟] 模型处理了形状为 {input_tensor.shape} 的张量")
    pred_3d_tensor = torch.zeros((1, frames, 17, 3)).to(device)
    # -------------------------------------------------------------------

    # ==========================================
    # 步骤 5: 后处理与返回
    # ==========================================
    # 剥离 Batch 维度，转回 NumPy: [1, Frames, 17, 3] -> [Frames, 17, 3]
    pred_3d_numpy = pred_3d_tensor.squeeze(0).cpu().numpy()

    print("✅ 3D 升维完成！")
    return pred_3d_numpy


def main():
    for i in ['1']:
        input_video = Path(f"/home/user/bones/sample_data/sample.mp4")

        if not Path(input_video).exists():
            print(f"输入视频不存在，跳过: {input_video}")
            continue

        # 创建输出目录
        output_dir = Path(f"../sample_output/sample")
        output_dir.mkdir(parents=True, exist_ok=True)

        # COCO 格式的关键点预览
        keypoints_2d_coco = extract_2d_key_points(input_video)
        Serializer.save_keypoints_to_json(
            keypoints_2d_coco,
            output_path=output_dir / "2d_keypoints_coco.json")
        Serializer.save_keypoints_to_pickle(
            keypoints_2d_coco,
            output_path=output_dir / "2d_keypoints_coco.pkl")
        VideoRenderer.render_2d_keypoints(
            input_video,
            keypoints_2d_coco,
            "coco",
            output_dir / "2d_kp_coco.mp4")

        # H36M 格式关键点预览
        keypoints_2d_h36m = convert_to_h36m_format_batch(keypoints_2d_coco)
        Serializer.save_keypoints_to_json(
            keypoints_2d_h36m,
            output_path=output_dir / "2d_keypoints_h36m.json")
        Serializer.save_keypoints_to_pickle(
            keypoints_2d_h36m,
            output_path=output_dir / "2d_keypoints_h36m.pkl")
        VideoRenderer.render_2d_keypoints(
            input_video,
            keypoints_2d_h36m,
            "h36m",
            output_dir / "2d_kp_h36m.mp4")

        break


def main2():
    for i in ['1', '2', '3', '4', 'x']:
        input_video = Path(f"/home/user/bones/sample_data/sample_{i}.mp4")

        if not Path(input_video).exists():
            print(f"输入视频不存在，跳过: {input_video}")
            return

        capture = cv2.VideoCapture(str(input_video))
        if not capture.isOpened():
            print(f"无法打开视频文件: {input_video}")
            return

        # 使用 cv2.CAP_PROP_FRAME_WIDTH 和 HEIGHT 获取属性
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()

        keypoints_2d_coco = extract_2d_key_points(input_video)
        keypoints_2d_h36m = convert_to_2d_h36m_tensor(keypoints_2d_coco)
        keypoints_3d = Keypoints3DInferencer.run_3d_inference(
            keypoints_2d_h36m, video_width=width, video_height=height)

        # 渲染 2d 视频
        VideoRenderer.render_2d_keypoints(
            input_video,
            convert_to_h36m_format_batch(keypoints_2d_coco),
            "h36m",
            Path(f"/home/user/bones/sample_output/sample_{i}/2d_kp_h36m.mp4"))
        # 渲染 3d 视频
        VideoRenderer.render_3d_keypoints(
            keypoints_3d,
            Path(f"/home/user/bones/sample_output/sample_{i}/3d_kp.mp4"))


if __name__ == "__main__":
    main()
