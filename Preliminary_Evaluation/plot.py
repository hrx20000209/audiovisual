import re
import matplotlib.pyplot as plt

# File path
file_path = './record_log.txt'

# 初始化存储时长的列表
video_durations = []
audio_durations = []

# 正则表达式匹配模式
video_duration_pattern = re.compile(r'Video Frame \d+ - .+ Duration: (\d+\.\d+)')
audio_duration_pattern = re.compile(r'Audio Frame \d+ - .+ Duration: (\d+\.\d+)')

# 读取文件并提取时长
with open(file_path, 'r') as file:
    for line in file:
        video_match = video_duration_pattern.search(line)
        audio_match = audio_duration_pattern.search(line)

        if video_match:
            video_durations.append(float(video_match.group(1)))

        if audio_match:
            audio_durations.append(float(audio_match.group(1)))

base_video = [1 / 30 * 1000 for _ in range(len(video_durations))]
base_audio = [4096 / 16000 * 1000 for _ in range(len(audio_durations))]

# 检查提取的数据是否为空
if not video_durations or not audio_durations:
    print("No video or audio durations were extracted. Please check the file format.")
else:
    # 画图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # 画视频时长的折线图
    ax1.plot(video_durations, marker='s', linestyle='-', color='blue')
    ax1.plot(base_video, linestyle='-', color='green')
    ax1.set_title('Video Durations')
    ax1.set_xlabel('Frame Number')
    ax1.set_ylabel('Duration (seconds)')

    # 画音频时长的折线图
    ax2.plot(audio_durations, marker='s', linestyle='-', color='red')
    ax2.plot(base_audio, linestyle='-', color='green')
    ax2.set_title('Audio Durations')
    ax2.set_xlabel('Frame Number')
    ax2.set_ylabel('Duration (seconds)')

    # 调整布局
    plt.tight_layout()

    # 显示图形
    plt.show()