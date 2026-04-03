import cv2
import time
import pickle
import numpy as np
from pathlib import Path
from ultralytics import YOLO


def benchmark_yolo_pipeline(video_path: str, output_pkl: str = "yolo_keypoints.pkl"):
    print("==================================================")
    print("🚀 启动 YOLOv8-Pose 极速单流测试 (并保存 pkl)")
    print("==================================================")

    video_file = Path(video_path).expanduser().resolve()
    if not video_file.exists():
        print(f"❌ 找不到视频文件: {video_file}")
        return

    # ---------------------------------------------------------
    # 阶段一：纯读图
    # ---------------------------------------------------------
    print("\n[1/3] 正在将整个视频加载到内存中...")
    cap = cv2.VideoCapture(str(video_file))
    frames = []

    start_io = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    end_io = time.time()
    cap.release()

    total_frames = len(frames)
    io_time = end_io - start_io
    io_fps = total_frames / io_time if io_time > 0 else 0

    print(f"✅ 视频加载完成！共 {total_frames} 帧 (耗时: {io_time:.2f}s)")

    # ---------------------------------------------------------
    # 阶段二：模型初始化
    # ---------------------------------------------------------
    print("\n[2/3] 正在加载 YOLOv8l-Pose 模型权重到 GPU...")
    init_start = time.time()
    model = YOLO("yolov8l-pose.pt")
    init_time = time.time() - init_start
    print(f"✅ 模型初始化完成！耗时: {init_time:.2f} 秒")

    # ---------------------------------------------------------
    # 阶段三：纯推理性能测试 & 关键点收集
    # ---------------------------------------------------------
    print(f"\n[3/3] 正在测试纯推理性能并提取关键点...")

    start_inf = time.time()
    all_frames_kps = []  # 用于暂存每一帧的关键点数组

    for i, frame in enumerate(frames):
        # 跑推断
        results = model(frame, verbose=False, device="cuda:0")

        # 💥 提取关键点数据 (形状为 [Persons, 17, 3], 包含 x, y, conf)
        # 如果画面里有人，results[0].keypoints 就不为空
        if results[0].keypoints is not None:
            # 放到 cpu 上并转为 numpy 数组
            kps_np = results[0].keypoints.data.cpu().numpy()
            all_frames_kps.append(kps_np)
        else:
            # 画面里没人，塞个空数组占位
            all_frames_kps.append(np.zeros((0, 17, 3), dtype=np.float32))

        if (i + 1) % 100 == 0:
            print(f"   已处理 {i + 1}/{total_frames} 帧...")

    end_inf = time.time()
    inf_time = end_inf - start_inf
    inf_fps = total_frames / inf_time if inf_time > 0 else 0

    # ---------------------------------------------------------
    # 阶段四：张量重构与保存
    # ---------------------------------------------------------
    print("\n[4/4] 正在重构张量并保存至本地...")
    max_persons = max([kps.shape[0] for kps in all_frames_kps] + [0])

    if max_persons == 0:
        print("⚠️ 视频中未检测到任何人！")
        output_tensor = np.zeros((0, total_frames, 17, 3), dtype=np.float32)
    else:
        # 初始化标准化张量 [Persons, Frames, 17, 3]
        output_tensor = np.zeros((max_persons, total_frames, 17, 3), dtype=np.float32)
        for frame_idx, kps in enumerate(all_frames_kps):
            num_persons = kps.shape[0]
            if num_persons > 0:
                # 把当前帧里所有人按顺序填进张量
                output_tensor[:num_persons, frame_idx, :, :] = kps

    # 序列化保存到本地
    with open(output_pkl, "wb") as f:
        pickle.dump(output_tensor, f)

    # ---------------------------------------------------------
    # 终极宣判
    # ---------------------------------------------------------
    print("\n==================================================")
    print("🏆 YOLOv8-Pose 测试与导出报告 🏆")
    print("==================================================")
    print(f"📹 视频总帧数 : {total_frames} 帧 (Batch=1)")
    print(f"🚀 阶段一 (纯读图)  : {io_fps:.2f} FPS")
    print(f"🧠 阶段二 (纯推理)  : {inf_fps:.2f} FPS (耗时 {inf_time:.2f}s)")
    print(f"💾 数据已保存至     : {output_pkl}")
    print(
        f"📐 最终张量形状     : {output_tensor.shape} (Persons, Frames, Joints, Channels)"
    )
    print("==================================================")
    print("💡 结论：你可以直接把生成的 pkl 扔给 VideoRenderer 看看效果了！")


if __name__ == "__main__":
    test_video_path = "/home/user/bones/sample_data/sample_1.mp4"
    # 保存到当前目录
    benchmark_yolo_pipeline(test_video_path, output_pkl="./yolo_keypoints.pkl")
