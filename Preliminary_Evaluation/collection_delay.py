import os
import threading
import logging
from time import time, sleep
import cv2
import numpy as np
import pyaudio
import wave
from utils import get_time

# 初始化日志记录器
logging.basicConfig(filename='record_log.txt', level=logging.INFO)
barrier = threading.Barrier(2)
video_frame_duration = []
audio_frame_duration = []


def record_video(cap, start_time=0.0, output_dir="video_frames"):
    barrier.wait()
    frame_count = 0
    while cap.isOpened():
        video_collection_start_time = get_time(start_time)
        ret, frame = cap.read()
        if not ret:
            break

        # 保存每一帧为单独的图像文件
        frame_filename = os.path.join(output_dir, f"frame_{frame_count:06d}.png")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

        video_collection_end_time = get_time(start_time)
        duration = video_collection_end_time - video_collection_start_time
        video_frame_duration.append(duration)
        logging.info(
            f'Video Frame {frame_count} - Start Time: {video_collection_start_time}, '
            f'End Time: {video_collection_end_time}, Duration: {duration}'
        )

    cap.release()
    cv2.destroyAllWindows()


def record_audio(stream, wavefile, chunk=2048, start_time=0.0):
    barrier.wait()

    audio_frame_count = 0

    print(f'Audio Ok')

    while True:
        audio_collection_start_time = get_time(start_time)
        audio_stream = stream.read(chunk)

        # 保存音频数据到WAV文件
        wavefile.writeframes(audio_stream)
        audio_frame_count += 1

        audio_collection_end_time = get_time(start_time)
        duration = audio_collection_end_time - audio_collection_start_time
        audio_frame_duration.append(duration)
        logging.info(
            f'Audio Frame {audio_frame_count} - Start Time: {audio_collection_start_time}, '
            f'End Time: {audio_collection_end_time}, Duration: {duration}'
        )

    stream.stop_stream()
    stream.close()
    wavefile.close()
    audio.terminate()


if __name__ == '__main__':
    CHUNK = 4096
    RATE = 16000
    frame_size = (256, 256)
    start_time = time()

    # 启动视频和音频收集线程
    output_dir = "../output/video_frames"
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])
    cap.set(cv2.CAP_PROP_FPS, 40)
    logging.info(f'FPS: {cap.get(cv2.CAP_PROP_FPS)}')

    print(f'Video Ok')

    audio_output = "../output/output_audio.wav"
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    wavefile = wave.open(audio_output, 'wb')
    wavefile.setnchannels(1)
    wavefile.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    wavefile.setframerate(RATE)

    video_thread = threading.Thread(target=record_video, args=(cap, start_time, output_dir))
    audio_thread = threading.Thread(target=record_audio, args=(stream, wavefile, CHUNK, start_time))

    video_thread.start()
    audio_thread.start()

    print('start recording')
    # 停止线程（因为没有自然退出机制，可以使用手动方式或其他机制来停止）
    video_thread.join(timeout=1)
    audio_thread.join(timeout=1)

    end_time = get_time(start_time)
    print(f'Start Time: 0')
    print(f'End Time: {end_time}')
