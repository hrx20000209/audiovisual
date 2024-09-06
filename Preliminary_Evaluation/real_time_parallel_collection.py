import math
import os
import wave
from queue import Queue

import cv2
import torch
import pyaudio
import threading
import logging
import numpy as np
import torch.nn.functional as F
from utils import get_time
from time import time, sleep

from matplotlib import pyplot as plt
from torchvision import transforms
from cvtransforms import CenterCrop, ColorNormalize

from audio_model import lipreading as audio_model_lipreading
from video_model import lipreading as video_model_lipreading
from concat_model import lipreading as concat_model_lipreading

init_time = time()

total_fusion_latency = []
total_video_collection_latency = []
total_audio_collection_latency = []
total_video_preprocessing_latency = []
total_audio_preprocessing_latency = []
total_video_encode_latency = []
total_audio_encode_latency = []

audio_data_buffer = Queue()
video_data_buffer = Queue()

barrier = threading.Barrier(2)

def normalisation(inputs):
    inputs = np.array(inputs)
    inputs_std = inputs.std()
    if inputs_std == 0.:
        inputs_std = 1.
    return (inputs - inputs.mean()) / inputs_std


def calculate_differences(input_list):
    # 检查列表长度是否足够计算差值
    if len(input_list) < 2:
        return []

    # 计算差值列表
    differences = [input_list[i] - input_list[i - 1] for i in range(1, len(input_list))]
    return differences


def reload_model(model, logger, path=""):
    if not bool(path):
        logger.info('train from scratch')
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location=torch.device('cpu'))
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logger.info('*** model has been successfully loaded! ***')
        return model


class VideoRecorder:
    def __init__(self, model, fps=30, frame_size=(256, 256), start_time=0.0):
        self.fps = fps
        self.frame_size = frame_size
        self.open = True
        self.cap = cv2.VideoCapture(1)
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_size[1])
        self.start_time = start_time
        self.model = model
        self.thread = threading.Thread(target=self.record)

    def record(self):
        while self.open and self.cap.isOpened():
            barrier.wait()
            frames = []

            video_collection_start_time = get_time(self.start_time)
            # frame_count = 0
            while get_time(self.start_time) - video_collection_start_time <= 1000.:
                # sample_start = get_time(self.start_time)
                ret, frame = self.cap.read()
                if not ret:
                    break
                frames.append(frame[..., ::-1][112:368, 192:448, :])

            video_data_buffer.put(frames)
            video_collection_end_time = get_time(self.start_time)
            video_collection_latency = video_collection_end_time - video_collection_start_time
            total_video_collection_latency.append(video_collection_latency)

            print(f'Video Collection Start Time:    {video_collection_start_time}')
            print(f'Video Collection End Time:      {video_collection_end_time}')

    def stop(self):
        self.open = False
        self.cap.release()
        cv2.destroyAllWindows()
        self.thread.join()

    def start(self):
        self.thread.start()


class AudioRecorder:
    def __init__(self, model, rate=16000, chunk=2048, start_time=0.0, output_file="../output/real-time/output_audio.wav"):
        self.rate = rate
        self.chunk = chunk
        self.open = True
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=pyaudio.paInt16,
                                      channels=1,
                                      rate=self.rate,
                                      input=True,
                                      frames_per_buffer=self.chunk
                                      )
        self.start_time = start_time
        self.model = model
        self.audio_buffer = Queue()
        self.thread = threading.Thread(target=self.record)

    def record(self):

        while self.open:
            barrier.wait()
            batch_data = []
            audio_collection_start_time = get_time(self.start_time)

            frame_num = 0
            while get_time(self.start_time) - audio_collection_start_time < 1000.:
                frame_num += 1
                audio_stream = self.stream.read(self.chunk)
                data = np.frombuffer(audio_stream, dtype=np.int16)
                batch_data.append(data)

            audio_data_buffer.put(batch_data)

            audio_collection_end_time = get_time(self.start_time)
            audio_collection_latency = audio_collection_end_time - audio_collection_start_time
            total_audio_collection_latency.append(audio_collection_latency)

            print(f'Audio Collection Start Time:    {audio_collection_start_time}')
            print(f'Audio Collection End Time:      {audio_collection_end_time}')

    def stop(self):
        self.open = False
        # self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        self.thread.join()

    def start(self):
        self.thread.start()


if __name__ == '__main__':
    model_path = '../checkpoint'
    mode = 'backendGRU'
    audio_frame_len = 29
    video_frame_len = 29

    save_path = '../backendGRU_real_time'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    logger = logging.getLogger("latency_log")
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(save_path + '/backendGRU_latency_log.txt')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    audio_model = audio_model_lipreading(mode=mode, inputDim=512, hiddenDim=512, nClasses=500,
                                         frameLen=audio_frame_len, every_frame=False)
    video_model = video_model_lipreading(mode=mode, inputDim=256, hiddenDim=512, nClasses=500,
                                         frameLen=video_frame_len, every_frame=False)
    concat_model = concat_model_lipreading(mode=mode, inputDim=2048, hiddenDim=512, nLayers=2, nClasses=500,
                                           every_frame=False)

    print('reload audio model')
    audio_path = os.path.join(model_path, 'Audiovisual_a_part.pt')
    audio_model = reload_model(audio_model, logger, audio_path)
    print("reload video model")
    video_path = os.path.join(model_path, 'Audiovisual_v_part.pt')
    video_model = reload_model(video_model, logger, video_path)
    print("reload concat model")
    concat_path = os.path.join(model_path, 'Audiovisual_c_part.pt')
    concat_model = reload_model(concat_model, logger, concat_path)

    label_list = []
    with open('../label_sorted.txt', 'r') as f:
        for line in f:
            label_list.append(line[:-1])

    CHUNK = 8192  # Samples: 1024,  512, 256, 128 frames per buffer
    RATE = 16000

    fps = 30
    frame_size = (256, 256)

    start_time = time()

    video_start_time = get_time(start_time)
    video_recorder = VideoRecorder(model=video_model, fps=fps, frame_size=frame_size, start_time=start_time)
    video_recorder.start()
    video_start_finish_time = get_time(start_time)
    print(f'Video Start Time: {video_start_finish_time:.6} ms')

    audio_start_time = get_time(start_time)
    audio_recorder = AudioRecorder(model=audio_model, rate=RATE, chunk=CHUNK, start_time=start_time)
    audio_recorder.start()
    audio_start_finish_time = get_time(start_time)
    print(f'audio Start Time: {audio_start_finish_time:.6} ms')

    idx = 0

    while True:
        if not (audio_data_buffer.empty() and video_data_buffer.empty()):
            try:
                audio_data = audio_data_buffer.get()
                video_data = video_data_buffer.get()

                video_preprocessing_start_time = get_time(start_time)
                # start_pre = get_time(start_time)
                frames = np.array(video_data)[..., ::-1][:, 115:211, 79:175, :][:29]  # 29, 96, 96, 3
                # slice_latency = get_time(start_time) - start_pre
                # print(f'slice {slice_latency}')

                video_frames = np.stack([cv2.cvtColor(item, cv2.COLOR_BGR2GRAY) for item in frames], axis=0)
                video_frames = video_frames / 255.  # 29, 96, 96
                video_frames = np.reshape(video_frames,
                                          (1, video_frames.shape[0], video_frames.shape[1], video_frames.shape[2]))

                batch_img = CenterCrop(video_frames, (88, 88))
                batch_img = ColorNormalize(batch_img)
                batch_img = np.reshape(batch_img, (
                batch_img.shape[0], batch_img.shape[1], batch_img.shape[2], batch_img.shape[3], 1))

                video_tensor = torch.from_numpy(batch_img)
                video_tensor = video_tensor.float().permute(0, 4, 1, 2, 3)

                video_preprocessing_end_time = get_time(start_time)
                video_preprocessing_latency = video_preprocessing_end_time - video_preprocessing_start_time
                total_video_preprocessing_latency.append(video_preprocessing_latency)

                video_encoding_start_time = get_time(start_time)
                video_feature = video_model(video_tensor)
                video_encoding_end_time = get_time(start_time)
                video_encoding_latency = video_encoding_end_time - video_encoding_start_time
                total_video_encode_latency.append(video_encoding_latency)

                audio_preprocessing_start_time = get_time(start_time)

                audio_inputs = torch.Tensor(np.array(audio_data)).view(1, -1)  # 变成 [1, length] 形状
                audio_inputs = torch.Tensor(normalisation(audio_inputs))

                # 检查长度是否小于 19456，如果是则补齐
                if audio_inputs.size(1) < 19456:
                    padding_size = 19456 - audio_inputs.size(1)
                    audio_tensor = F.pad(audio_inputs, (0, padding_size), "constant", 0)
                else:
                    audio_tensor = audio_inputs[:, :19456]

                audio_preprocessing_end_time = get_time(start_time)
                audio_preprocessing_latency = audio_preprocessing_end_time - audio_preprocessing_start_time
                total_audio_preprocessing_latency.append(audio_preprocessing_latency)

                audio_encoding_start_time = get_time(start_time)
                audio_feature = audio_model(audio_tensor)
                audio_encoding_end_time = get_time(start_time)
                audio_encoding_latency = audio_encoding_end_time - audio_encoding_start_time
                total_audio_encode_latency.append(audio_encoding_latency)

                fusion_start_time = get_time(start_time)
                cat_input = torch.cat((audio_feature, video_feature), dim=2)
                outputs = concat_model(cat_input)
                fusion_end_time = get_time(start_time)
                fusion_latency = fusion_end_time - fusion_start_time
                total_fusion_latency.append(fusion_latency)

                # outputs = torch.mean(outputs, 1)
                _, preds = torch.max(F.softmax(outputs, dim=1).data, 1)

                print(f'Video Preprocessing Start Time: {video_preprocessing_start_time}')
                print(f'Video Preprocessing End Time:   {video_preprocessing_end_time}')
                print(f'Video Encode Start:             {video_encoding_start_time}')
                print(f'Video Encode End:               {video_encoding_end_time}')
                print(f'Audio Preprocessing Start Time: {audio_preprocessing_start_time}')
                print(f'Audio Preprocessing End Time:   {audio_preprocessing_end_time}')
                print(f'Audio Encode Start:             {audio_encoding_start_time}')
                print(f'Audio Encode End:               {audio_encoding_end_time}')
                print(f'Fusion Start:                   {fusion_start_time}')
                print(f'Fusion End:                     {fusion_end_time}')
                print(f'Predicted words:                {label_list[preds[0]]} \n')
                idx += 1
            except Exception as e:
                print(f"Exception: {e}")
                logger.error(f"Exception: {e}")
        if idx >= 10:
            break

    audio_recorder.stop()
    video_recorder.stop()


    def calculate_average_latency(latencies):
        return sum(latencies) / len(latencies) if latencies else 0


    logger.info(f'Average video collection latency: {calculate_average_latency(total_video_collection_latency)}')
    logger.info(f'Average video preprocessing latency: {calculate_average_latency(total_video_preprocessing_latency)}')
    logger.info(f'Average video encode latency: {calculate_average_latency(total_video_encode_latency)}')
    logger.info(f'Average audio collection latency: {calculate_average_latency(total_audio_collection_latency)}')
    logger.info(f'Average audio preprocessing latency: {calculate_average_latency(total_audio_preprocessing_latency)}')
    logger.info(f'Average audio encode latency: {calculate_average_latency(total_audio_encode_latency)}')
    logger.info(f'Average fusion latency: {calculate_average_latency(total_fusion_latency)}')
