import cv2
import time
from pathlib import Path
from mmpose.apis import MMPoseInferencer


def benchmark_rtmpose_pipeline(video_path: str, batch_size: int = 1):
    print("==================================================")
    print("🚀 启动两段式 RTMPose 性能隔离测试 (控制变量法)")
    print("==================================================")

    video_file = Path(video_path).expanduser().resolve()
    if not video_file.exists():
        print(f"❌ 找不到视频文件: {video_file}")
        return

    # ---------------------------------------------------------
    # 阶段一：纯粹的 I/O 与内存预热 (OpenCV 读全量视频)
    # ---------------------------------------------------------
    print("\n[1/3] 正在将整个视频加载到内存中 (请确保内存足够)...")
    cap = cv2.VideoCapture(str(video_file))
    frames = []

    start_io = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)  # 直接塞进内存列表
    end_io = time.time()
    cap.release()

    total_frames = len(frames)
    io_time = end_io - start_io
    io_fps = total_frames / io_time if io_time > 0 else 0

    print(f"✅ 视频加载完成！共 {total_frames} 帧")
    print(f"⏱️  纯读图耗时: {io_time:.2f} 秒 | 速度: {io_fps:.2f} FPS")

    # ---------------------------------------------------------
    # 阶段二：模型初始化 (将这部分时间独立，不干扰纯推理测试)
    # ---------------------------------------------------------
    print("\n[2/3] 正在加载 RTMDet 和 RTMPose 模型权重到 GPU...")
    init_start = time.time()
    inferencer = MMPoseInferencer(
        det_model="rtmdet-m", pose2d="rtmpose-l", device="cuda:0"
    )
    init_time = time.time() - init_start
    print(f"✅ 模型初始化完成！耗时: {init_time:.2f} 秒")

    # ---------------------------------------------------------
    # 阶段三：纯推理性能测试 (排除读图干扰)
    # ---------------------------------------------------------
    print(f"\n[3/3] 正在测试纯推理性能 (强制 Batch Size = {batch_size})...")

    # 将装满整个视频所有帧的内存列表传给推断器
    result_generator = inferencer(
        frames,
        show=False,
        return_vis=False,
        save_predictions=False,
        batch_size=batch_size,
    )

    start_inf = time.time()
    processed_count = 0

    # 必须遍历 generator 才会真正驱使 MMPose 执行推理
    for _ in result_generator:
        processed_count += 1
        if processed_count % 100 == 0:
            print(f"   已处理 {processed_count}/{total_frames} 帧...")

    end_inf = time.time()

    inf_time = end_inf - start_inf
    inf_fps = total_frames / inf_time if inf_time > 0 else 0

    # ---------------------------------------------------------
    # 终极宣判
    # ---------------------------------------------------------
    print("\n==================================================")
    print("🏆 性能隔离测试报告 🏆")
    print("==================================================")
    print(f"📹 视频总帧数 : {total_frames} 帧 (Batch={batch_size})")
    print(f"🚀 阶段一 (纯读图)  : {io_fps:.2f} FPS (耗时 {io_time:.2f}s)")
    print(f"🧠 阶段二 (纯推理)  : {inf_fps:.2f} FPS (耗时 {inf_time:.2f}s)")
    print("==================================================")

    if inf_fps < 20:
        print("💡 终极结论：")
        print("    在完全排除了 OpenCV 解码干扰、且图片全在内存的情况下，")
        print("    如果纯推理速度依然只有几帧或十几帧，那就铁证如山地证明了：")
        print("    MMPose 内部极其臃肿的『CPU/GPU 数据搬运与抠图』机制，")
        print("    彻底勒死了你的 RTX 4090！")
    else:
        print("💡 终极结论：")
        print("    纯推理速度飙升！说明瓶颈确实是之前 I/O 与推理的串行等待导致的。")


if __name__ == "__main__":
    test_video_path = "/home/user/bones/sample_data/sample_1.mp4"
    benchmark_rtmpose_pipeline(test_video_path, batch_size=1)
