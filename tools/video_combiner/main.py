from pathlib import Path
from tqdm import tqdm
import argparse
import cv2
import numpy as np
import glob


class VideoLayoutBuilder:
    """
    负责计算对比视频的布局，并提供单帧合成方法。
    """

    def __init__(self, sample_img_l: np.ndarray, sample_img_r: np.ndarray,
                 left_title: str, right_title: str):

        h_l, w_l, _ = sample_img_l.shape
        h_r, w_r, _ = sample_img_r.shape

        # 约束：以高度最低的为准
        self.target_h = min(h_l, h_r)

        # 约束：等比例缩放 (不拉伸)
        self.new_w_l = int(w_l * (self.target_h / h_l))
        self.new_w_r = int(w_r * (self.target_h / h_r))

        # 布局设计
        self.title_bar_h = 80  # 顶部留给标题的黑边高度
        self.margin_w = 50     # 左右及中间的间隔留白

        # 计算最终视频的画幅
        self.out_w_l = self.new_w_l + self.margin_w * 2
        self.out_w_r = self.new_w_r + self.margin_w * 2
        self.total_w = self.out_w_l + self.out_w_r
        self.total_h = self.target_h + self.title_bar_h

        # 字体配置与预计算 (避免在每帧循环中重复计算)
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.font_scale = 1.2
        self.thickness = 2
        self.text_color = (255, 255, 255)

        self.left_title = left_title
        self.right_title = right_title

        # 计算左侧标题坐标
        (text_w_l, text_h_l), _ = cv2.getTextSize(left_title, self.font, self.font_scale, self.thickness)
        self.text_x_l = (self.out_w_l - text_w_l) // 2
        self.text_y_l = (self.title_bar_h + text_h_l) // 2

        # 计算右侧标题坐标
        (text_w_r, text_h_r), _ = cv2.getTextSize(right_title, self.font, self.font_scale, self.thickness)
        self.text_x_r = self.out_w_l + (self.out_w_r - text_w_r) // 2
        self.text_y_r = (self.title_bar_h + text_h_r) // 2

    def combine_frame(self, img_l: np.ndarray, img_r: np.ndarray) -> np.ndarray:
        """
        合成单帧
        """
        # 等比缩放
        img_l_resized = cv2.resize(img_l, (self.new_w_l, self.target_h), interpolation=cv2.INTER_CUBIC)
        img_r_resized = cv2.resize(img_r, (self.new_w_r, self.target_h), interpolation=cv2.INTER_CUBIC)

        # 创建纯黑画布 [H, W, C]
        canvas = np.zeros((self.total_h, self.total_w, 3), dtype=np.uint8)

        # 把画面贴到画布对应的位置
        canvas[self.title_bar_h:self.total_h, self.margin_w: self.margin_w + self.new_w_l] = img_l_resized
        canvas[self.title_bar_h:self.total_h, self.out_w_l +
               self.margin_w: self.out_w_l + self.margin_w + self.new_w_r] = img_r_resized

        # 渲染标题
        cv2.putText(canvas, self.left_title, (self.text_x_l, self.text_y_l),
                    self.font, self.font_scale, self.text_color, self.thickness, cv2.LINE_AA)
        cv2.putText(canvas, self.right_title, (self.text_x_r, self.text_y_r),
                    self.font, self.font_scale, self.text_color, self.thickness, cv2.LINE_AA)

        return canvas


def combine_video_2(
    left_video_file: str,
    left_video_title: str,
    right_video_file: str,
    right_video_title: str,
    output_video_path: str,
    fps: int = 30
):
    """
    将左右两个视频文件合并为一个带标题的左右布局对比视频。
    """
    cap_l = cv2.VideoCapture(left_video_file)
    cap_r = cv2.VideoCapture(right_video_file)

    if not cap_l.isOpened() or not cap_r.isOpened():
        raise FileNotFoundError("无法打开其中一个或全部输入视频文件！")

    # 获取视频帧数，以较少的一方为准
    frames_l = int(cap_l.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_r = int(cap_r.get(cv2.CAP_PROP_FRAME_COUNT))
    num_frames = min(frames_l, frames_r)

    # 读取首帧用于布局计算
    ret_l, img_l_test = cap_l.read()
    ret_r, img_r_test = cap_r.read()

    if not ret_l or not ret_r:
        raise ValueError("视频文件为空或无法读取首帧！")

    # 初始化布局器
    layout = VideoLayoutBuilder(img_l_test, img_r_test, left_video_title, right_video_title)

    # 初始化视频写入器
    output_path_obj = Path(output_video_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path_obj), fourcc, fps, (layout.total_w, layout.total_h))

    print(f"🎬 开始合成对比视频: {output_video_path}")
    print(f"📏 统一高度: {layout.target_h}px, 最终分辨率: {layout.total_w}x{layout.total_h}")

    # 因为已经读取了首帧，所以先写入第一帧
    writer.write(layout.combine_frame(img_l_test, img_r_test))

    # 逐帧处理剩余的帧
    for _ in tqdm(range(1, num_frames), desc="视频合成中"):
        ret_l, img_l = cap_l.read()
        ret_r, img_r = cap_r.read()

        if not ret_l or not ret_r:
            break

        canvas = layout.combine_frame(img_l, img_r)
        writer.write(canvas)

    cap_l.release()
    cap_r.release()
    writer.release()
    print(f"✅ 视频合成完毕！已保存至 {output_video_path}")


def combine_video(
    left_video_source_image_dir: str,
    left_video_title: str,
    right_video_source_image_dir: str,
    right_video_title: str,
    output_video_path: str,
    fps: int = 30
):
    """
    将左右两个图像序列目录合并为一个带标题的左右布局对比视频。
    """
    left_dir = Path(left_video_source_image_dir).expanduser().resolve()
    right_dir = Path(right_video_source_image_dir).expanduser().resolve()

    left_images = sorted(glob.glob(str(left_dir / "*.png")))
    right_images = sorted(glob.glob(str(right_dir / "*.png")))

    if not left_images or not right_images:
        raise FileNotFoundError(f"未能在 {left_dir} 或 {right_dir} 中找到图像序列！")

    num_frames = min(len(left_images), len(right_images))

    # 读取首帧计算布局
    img_l_test = cv2.imread(left_images[0])
    img_r_test = cv2.imread(right_images[0])

    # 初始化布局器
    layout = VideoLayoutBuilder(img_l_test, img_r_test, left_video_title, right_video_title)

    output_path_obj = Path(output_video_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path_obj), fourcc, fps, (layout.total_w, layout.total_h))

    print(f"🎬 开始合成对比视频: {output_video_path}")
    print(f"📏 统一高度: {layout.target_h}px, 最终分辨率: {layout.total_w}x{layout.total_h}")

    # 逐帧处理
    for i in tqdm(range(num_frames), desc="图像序列合成中"):
        img_l = cv2.imread(left_images[i])
        img_r = cv2.imread(right_images[i])

        canvas = layout.combine_frame(img_l, img_r)
        writer.write(canvas)

    writer.release()
    print(f"✅ 视频合成完毕！已保存至 {output_video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将左右两个 MP4 视频合并为一个带标题的对比视频")

    # 必填参数：左视频、右视频
    parser.add_argument("-lv", "--left_video", required=True, help="左侧输入视频路径")
    parser.add_argument("-rv", "--right_video", required=True, help="右侧输入视频路径")

    # 必填参数：输出路径 (-o)
    parser.add_argument("-o", "--output", required=True, help="合成后的输出视频路径")

    # 选填参数：标题和 FPS
    parser.add_argument("-lt", "--left_title", default="2D", help="左侧视频标题 (默认: 2D)")
    parser.add_argument("-rt", "--right_title", default="3D", help="右侧视频标题 (默认: 3D)")
    parser.add_argument("--fps", type=int, default=30, help="输出视频帧率 (默认: 30)")

    args = parser.parse_args()

    # 调用你的函数
    combine_video_2(
        left_video_file=args.left_video,
        left_video_title=args.left_title,
        right_video_file=args.right_video,
        right_video_title=args.right_title,
        fps=args.fps,
        output_video_path=args.output
    )
