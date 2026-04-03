import cv2
import time
import numpy as np
from pathlib import Path
from mmpose.apis import MMPoseInferencer


def get_bbox_from_keypoints(
    keypoints: np.ndarray, frame_w: int, frame_h: int, scale: float = 1.25
):
    """
    核心算法：将 17 个关键点转换为当前帧的 Bounding Box
    scale=1.25 表示将长宽向外扩展 25%，给动作幅度留出缓冲空间
    """
    # 过滤掉无效点 (坐标为 0,0 的点)
    valid_kps = keypoints[np.any(keypoints > 0, axis=1)]
    if len(valid_kps) == 0:
        return None

    x_min, y_min = np.min(valid_kps, axis=0)
    x_max, y_max = np.max(valid_kps, axis=0)

    w = x_max - x_min
    h = y_max - y_min
    cx = x_min + w / 2
    cy = y_min + h / 2

    # 外扩 BBox
    w_new = w * scale
    h_new = h * scale

    # 边界保护，防止框出画
    x1 = max(0, cx - w_new / 2)
    y1 = max(0, cy - h_new / 2)
    x2 = min(frame_w, cx + w_new / 2)
    y2 = min(frame_h, cy + h_new / 2)

    return [float(x1), float(y1), float(x2), float(y2)]


def benchmark_rtmpose_tracking(video_path: str):
    print("==================================================")
    print("🚀 启动 RTMPose 单次检测 + 关键点追踪极限测试")
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
    frame_h, frame_w = frames[0].shape[:2]
    print(f"✅ 视频加载完成！共 {total_frames} 帧")

    # ---------------------------------------------------------
    # 阶段二：模型初始化
    # ---------------------------------------------------------
    print("\n[2/3] 正在加载模型权重到 GPU...")
    inferencer = MMPoseInferencer(
        det_model="rtmdet-m", pose2d="rtmpose-l", device="cuda:0"
    )
    print("✅ 模型初始化完成！")

    # ---------------------------------------------------------
    # 阶段三：基于追踪的纯推理性能测试
    # ---------------------------------------------------------
    print(f"\n[3/3] 正在测试追踪推理性能 (跳过逐帧检测器)...")

    start_inf = time.time()
    current_bbox = None
    processed_count = 0

    for i, frame in enumerate(frames):
        # 💥 魔法分歧点
        if current_bbox is None:
            # 1. 丢帧或第 1 帧时：走完整的 Det + Pose 流程
            result_gen = inferencer(
                frame, show=False, return_vis=False, save_predictions=False
            )
        else:
            # 2. 追踪模式：直接把上一帧算出的 BBox 强塞给 MMPose
            # bboxes 参数格式要求：传入包含一个 bbox 列表的列表
            result_gen = inferencer(
                frame,
                bboxes=[[current_bbox]],
                show=False,
                return_vis=False,
                save_predictions=False,
            )

        res = next(result_gen)

        # 提取关键点并更新下一帧的 bbox
        try:
            preds = res["predictions"][0]
            if len(preds) > 0:
                # 假设我们只追踪画面里的第一个人 (主角)
                kps = np.array(preds[0]["keypoints"])
                current_bbox = get_bbox_from_keypoints(
                    kps, frame_w, frame_h, scale=1.25
                )
            else:
                current_bbox = None  # 画面没人了，下一帧重新唤醒 RTMDet 搜索
        except Exception:
            current_bbox = None

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
    print("🏆 RTMPose 追踪模式性能测试报告 🏆")
    print("==================================================")
    print(f"📹 视频总帧数 : {total_frames} 帧")
    print(f"🧠 纯推理耗时 : {inf_time:.2f} 秒")
    print(f"⚡ 最终 FPS   : {inf_fps:.2f} FPS")
    print("==================================================")
    print("💡 结论：")
    print("    这次我们砍掉了 RTMDet 和它背后的 CPU 抠图开销，")
    print("    看看原汁原味的 RTMPose 能跑多快！")


if __name__ == "__main__":
    test_video_path = "/home/user/bones/sample_data/sample_1.mp4"
    benchmark_rtmpose_tracking(test_video_path)
