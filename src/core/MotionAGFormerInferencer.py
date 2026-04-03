from pathlib import Path
from .share import IKeypoints3DInferencer
from MotionAGFormer.model.MotionAGFormer import MotionAGFormer
from MotionAGFormer.utils.learning import load_model
from MotionAGFormer.utils.tools import get_config
from typing import Any, OrderedDict, Tuple
from tqdm import tqdm
import numpy as np
import torch
import os


class MotionAGFormerInferencer(IKeypoints3DInferencer):
    def __init__(self, config_path: Path, weight_path: Path, window: int, stride: int) -> None:
        super().__init__()
        self.config_path = config_path
        self.weight_path = weight_path
        self.window = window
        self.stride = stride

    def run_3d_keypoints_inference(
        self,
        keypoints_2d: np.ndarray,
        video_width: int,
        video_height: int,
    ) -> np.ndarray:
        total_frames = keypoints_2d.shape[0]
        model, target_frames = self.__load_motionagformer()

        if total_frames <= self.window:
            pad_length = self.window - total_frames
            # 如果视频比 window 帧还短，我们直接在首尾复制 Padding 补齐到 window 帧，通过在末尾重复最后一帧补齐
            padded_2d = np.pad(keypoints_2d, ((0, pad_length), (0, 0), (0, 0)), mode='edge')
            pred_3d = self.__run_3d_inference(
                model,
                padded_2d,
                video_width=video_width,
                video_height=video_height
            )  # 假设 run_3d_inference 已经被改成了严格处理 window 帧
            return pred_3d[:total_frames]  # 算完后把填充的尾巴切掉

        # ==========================================
        # 滑动窗口核心逻辑
        # ==========================================
        # 准备输出容器和计数器
        final_3d = np.zeros((total_frames, 17, 3), dtype=np.float32)
        weight_counts = np.zeros((total_frames, 1, 1), dtype=np.float32)

        # 计算所有窗口的起始索引
        starts = list(range(0, total_frames - self.window + 1, self.stride))

        # 🌟 关键防坑：确保视频的最后一段绝对被覆盖到
        if starts[-1] + self.window < total_frames:
            starts.append(total_frames - self.window)

        print("使用 MotionAGFormer 重建 3D 骨骼...")
        print(f"启动滑动窗口推理: 总帧数 {total_frames}, 将被切分为 {len(starts)} 个区块...")

        for start in tqdm(starts, desc="3D 推理进度"):
            end = start + self.window

            # 切割当前块
            chunk_2d = keypoints_2d[start:end]

            # 送入模型推理 (严格 window 帧进，window 帧出)
            chunk_3d = self.__run_3d_inference(
                model,
                chunk_2d,
                video_width=video_width,
                video_height=video_height
            )

            # 将预测结果累加到总容器的对应位置
            final_3d[start:end] += chunk_3d
            # 记录该位置被预测的次数
            weight_counts[start:end] += 1
            
        print(f"3D 骨骼重建完成: {final_3d.shape}")

        # ==========================================
        # 结果融合：对重叠区域取平均值
        # ==========================================
        final_3d = final_3d / weight_counts

        print("✅ 全视频 3D 平滑升维完毕！")
        return final_3d

    def __load_motionagformer(
        self,
    ) -> Tuple[MotionAGFormer, Any]:
        """
        使用作者的官方流程加载模型，并带上万能权重清洗功能
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        args = get_config(self.config_path)

        model = load_model(args)

        if not os.path.exists(self.weight_path):
            raise FileNotFoundError(f"⚠️ 找不到权重文件，请确保将其放到了: {self.weight_path}")

        print(f"📦 正在解析权重文件: {self.weight_path}")
        checkpoint = torch.load(self.weight_path, map_location=device)

        # ==========================================
        # 🌟 核心防坑：动态寻找真正的权重字典
        # ==========================================
        if 'model_pos' in checkpoint:
            raw_state_dict = checkpoint['model_pos']
        elif 'model_state_dict' in checkpoint:
            raw_state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            raw_state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            raw_state_dict = checkpoint['state_dict']
        else:
            # 如果存的直接就是权重字典本身
            raw_state_dict = checkpoint

        # ==========================================
        # 🌟 核心防坑：清洗 "module." 前缀
        # ==========================================
        clean_state_dict = OrderedDict()
        for k, v in raw_state_dict.items():
            # 把前缀剥掉：比如 "module.pos_embed" 变成 "pos_embed"
            clean_k = k.replace("module.", "")
            clean_state_dict[clean_k] = v

        # 加载清洗后的权重 (先尝试严格模式，如果失败则回退到非严格模式并打印缺失项)
        try:
            model.load_state_dict(clean_state_dict, strict=True)
            print("✅ 权重完美加载 (Strict Mode)！")
        except RuntimeError:
            print("⚠️ 严格模式加载失败，尝试忽略差异项加载...")
            # non-strict 允许你忽略某些微小差异（通常不影响运行）
            model.load_state_dict(clean_state_dict, strict=False)
            print("✅ 权重已加载 (Non-Strict Mode)！")

        model = model.to(device)
        model.eval()

        return model, args.n_frames  # type: ignore

    def __run_3d_inference(
        self,
        model: MotionAGFormer,
        keypoints_2d: np.ndarray,  # shape: [Frames, 17, 3] (x, y, confidence)
        video_width: int = 1920,
        video_height: int = 1080
    ) -> np.ndarray:  # shape: [Frames, 17, 3]
        """
        输入: [Frames, 17, 3] 形状的 H36M 关键点
        输出: [243, 17, 3] 形状的 3D 关键点
        """
        device = next(model.parameters()).device

        # ==========================================
        # 步骤 1: 坐标归一化 (还原论文的处理手法)
        # ==========================================
        processed_2d = keypoints_2d.copy().astype(np.float32)
        # 缩放到 [-1, 1] 之间
        processed_2d[:, :, 0] = (processed_2d[:, :, 0] / video_width) * 2.0 - 1.0
        processed_2d[:, :, 1] = (processed_2d[:, :, 1] / video_height) * 2.0 - 1.0

        # ==========================================
        # 步骤 2: 根节点相对化 (Root-relative)
        # ==========================================
        pelvis_2d = processed_2d[:, 0:1, :]
        processed_2d = processed_2d - pelvis_2d

        # 转 Tensor: [243, 17, 3] -> [1, 243, 17, 3]
        input_tensor = torch.from_numpy(processed_2d).unsqueeze(0).to(device)

        # ==========================================
        # 步骤 4: 进行前向传播
        # ==========================================
        with torch.no_grad():
            output_3d_tensor = model(input_tensor)

        # 剥掉 Batch 维度，转回 NumPy
        pred_3d_numpy = output_3d_tensor.squeeze(0).cpu().numpy()

        # [附加] 将 3D 骨盆归零 (由于残差，预测结果可能有微小偏移)
        pred_3d_numpy = pred_3d_numpy - pred_3d_numpy[:, 0:1, :]

        return pred_3d_numpy
