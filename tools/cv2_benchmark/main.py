import cv2
import time
import numpy as np
from tqdm import tqdm


def test_opencv_cpu_speed(video_path: str):
    print("==================================================")
    print("🚀 启动纯 CPU 极限抗压测试 (解码 + 预处理)")
    print("==================================================")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频文件: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📂 视频总帧数: {total_frames}")

    frames_processed = 0
    start_time = time.time()

    # 使用我们刚刚聊过的 tqdm 进度条
    with tqdm(total=total_frames, desc="CPU 疯狂打工中") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # ==========================================
            # 还原 MMPose 内部的 CPU 预处理“苦力活”
            # ==========================================
            # 1. Resize：深度学习最常见的操作，极度消耗 CPU
            resized_frame = cv2.resize(frame, (640, 640))

            # 2. 颜色空间转换：BGR 转 RGB (模型通常吃 RGB)
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

            # 3. 归一化模拟：转成浮点数并除以 255
            # 注意：只是把变量存一下，不打印不保留，模拟“丢弃结果”
            tensor_mock = rgb_frame.astype(np.float32) / 255.0

            frames_processed += 1
            pbar.update(1)

    end_time = time.time()
    cap.release()

    elapsed = end_time - start_time
    fps = frames_processed / elapsed

    print("\n==================================================")
    print("✅ 测试结束！真相大白：")
    print(f"⏱️  总耗时: {elapsed:.2f} 秒")
    print(f"⚡  纯 CPU 极限处理速度: {fps:.2f} 帧/秒 (FPS)")
    print("==================================================")
    print("💡 结论：如果这个数字在 10 上下徘徊，那 4090 确实被 CPU 饿死了！")


if __name__ == "__main__":
    test_opencv_cpu_speed("/home/user/bones/sample_data/sample_1.mp4")
