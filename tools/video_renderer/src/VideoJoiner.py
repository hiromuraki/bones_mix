import cv2
import numpy as np
from pathlib import Path


class VideoJoiner:
    @classmethod
    def concat_videos(cls, videos: list[tuple[Path, str]], output_video_file: Path | str):
        if not videos:
            print("警告：传入的视频列表为空。")
            return

        caps_info = []
        min_h = float('inf')
        fps = None

        # 1. 第一遍遍历：获取所有视频属性，寻找"最矮"的基准高度
        for path, label in videos:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise ValueError(f"无法打开视频文件: {path}")

            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if fps is None:
                fps = cap.get(cv2.CAP_PROP_FPS)  # 以第一个视频的FPS为准

            min_h = min(min_h, orig_h)
            caps_info.append({
                'cap': cap, 'label': label, 'orig_w': orig_w, 'orig_h': orig_h
            })

        if min_h == float('inf') or min_h <= 0:
            raise ValueError("无法获取有效的基准视频高度。")

        target_h = min_h

        # 2. 计算缩放尺寸并预制黑屏帧
        target_sizes = []
        black_frames = []
        total_w = 0

        for info in caps_info:
            # 等比缩放计算新宽度
            new_w = int(info['orig_w'] * (target_h / info['orig_h']))
            target_sizes.append((new_w, target_h))
            total_w += new_w

            # 预先创建一个纯黑帧 (高度, 宽度, 通道数)
            black_frame = np.zeros((target_h, new_w, 3), dtype=np.uint8)

            # 在黑屏上依然保留模型名称标签，让观众知道是哪个模型先结束了
            if info['label']:
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_pos = (20, 40)
                cv2.putText(black_frame, info['label'], text_pos, font, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(black_frame, info['label'], text_pos, font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

            black_frames.append(black_frame)

        # 3. 初始化 VideoWriter
        output_video_file = Path(output_video_file)
        output_video_file.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_file), fourcc, fps, (total_w, target_h))

        print(f"正在合成... 基准高度(最矮): {target_h}, 输出分辨率: {total_w}x{target_h}, FPS: {fps:.2f}")

        # 4. 逐帧读取、处理并拼接（最长视频耗尽前不停止）
        frame_count = 0
        while True:
            frames_to_concat = []
            all_videos_ended = True  # 标记是否所有视频都播放完毕

            for i, info in enumerate(caps_info):
                cap = info['cap']
                label = info['label']
                ret, frame = cap.read()

                if ret:
                    all_videos_ended = False  # 只要还有一个视频有画面，就不停止
                    # 缩放至目标尺寸
                    resized_frame = cv2.resize(frame, target_sizes[i], interpolation=cv2.INTER_LINEAR)

                    # 绘制标签
                    if label:
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        text_pos = (20, 40)
                        cv2.putText(resized_frame, label, text_pos, font, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
                        cv2.putText(resized_frame, label, text_pos, font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

                    frames_to_concat.append(resized_frame)
                else:
                    # 核心逻辑：当前视频已结束，使用对应尺寸的预制黑屏帧凑数
                    frames_to_concat.append(black_frames[i])

            # 如果所有视频都读不出新帧了，退出循环
            if all_videos_ended:
                break

            # 使用 numpy 进行横向拼接
            combined_frame = np.hstack(frames_to_concat)
            out.write(combined_frame)
            frame_count += 1

        # 5. 释放系统资源
        for info in caps_info:
            info['cap'].release()
        out.release()
        print(f"合成完成！共处理 {frame_count} 帧（以最长视频为准）。保存至：{output_video_file}")
