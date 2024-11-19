import cv2

def extract_frames(video_path):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Couldn't open video file")
        return []
    frames = []
    frame_number = 0
    while True:
        # 读取每一帧
        ret, frame = cap.read()
        if not ret:
            break  # 如果没有帧可读取，则退出
        # 将帧序号和帧图像作为元组添加到列表中
        frames.append((frame_number, frame))

        frame_number += 1

    # 释放视频对象
    cap.release()

    return frames


# 调用函数
video_path = './assets/1.mp4'  # 替换成你的视频文件路径
frames = extract_frames(video_path)