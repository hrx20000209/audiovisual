import math
import os
import threading

import cv2

import keyboard
import logging
import torch
import numpy as np
import speech_recognition as sr
import torch.nn.functional as F

from queue import Queue
from time import time
from utils import get_time

from matplotlib import pyplot as plt
from torchvision.transforms import transforms

from audio_model import lipreading as audio_model_lipreading
from video_model import lipreading as video_model_lipreading
from concat_model import lipreading as concat_model_lipreading



transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop((88, 88)),
    transforms.Normalize(mean=[0.413621], std=[0.1700239])
])

total_fusion_latency = []
total_video_collection_latency = []
total_audio_collection_latency = []
total_video_preprocessing_latency = []
total_audio_preprocessing_latency = []
total_video_encode_latency = []
total_audio_encode_latency = []

audio_feature_buffer = Queue()
video_feature_buffer = Queue()

start_event = threading.Event()

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


def normalisation(inputs):
    inputs = np.array(inputs)
    inputs_std = inputs.std()
    if inputs_std == 0.:
        inputs_std = 1.
    return (inputs - inputs.mean()) / inputs_std


def audio_process(wav_file, model):
    r = sr.Recognizer()

    audio_collection_start_time = get_time(start_time)
    audio_collection_end_time = get_time(start_time)
    with sr.AudioFile(wav_file) as source:
        audio = r.record(source)
    data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
    audio_collection_duration = (audio_collection_end_time - audio_collection_start_time)
    total_audio_collection_latency.append(audio_collection_duration)

    audio_preprocessing_start_time = get_time(start_time)
    data = torch.Tensor(normalisation(data)).unsqueeze(0)
    audio_preprocessing_end_time = get_time(start_time)
    audio_preprocessing_latency = (audio_preprocessing_end_time - audio_preprocessing_start_time)
    total_audio_preprocessing_latency.append(audio_preprocessing_latency)

    print(f'Audio Collection Start Time:    {audio_collection_start_time}')
    print(f'Audio Collection End Time:      {audio_collection_end_time}')
    print(f'Audio Preprocessing Start Time: {audio_preprocessing_start_time}')
    print(f'Audio Preprocessing End Time:   {audio_preprocessing_end_time}')

    print(f'Audio Collection Latency:    {total_audio_collection_latency[0]}')
    print(f'Audio Preprocessing Latency: {total_audio_preprocessing_latency[0]}')

    for i in range(sample_num):
        audio_tensor = data[:, 19456 * i: 19456 * (i + 1)]

        audio_encoding_start_time = get_time(start_time)
        audio_output = model(audio_tensor)
        audio_encoding_end_time = get_time(start_time)
        audio_encoding_latency = (audio_encoding_end_time - audio_encoding_start_time)
        total_audio_encode_latency.append(audio_encoding_latency)

        audio_feature_buffer.put(audio_output)

        print(f'Audio Encode Start:     {audio_encoding_start_time}')
        print(f'Audio Encode End:       {audio_encoding_end_time}')
        print(f'Audio Encode Latency    {audio_encoding_latency}')


def video_process(video_file, model):
    video_collection_start_time = get_time(start_time)
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # frames.append(np.array(frame)[115:211, 79:175, :])
        frames.append(frame)

    video_collection_end_time = get_time(start_time)
    video_collection_duration = (video_collection_end_time - video_collection_start_time)
    total_video_collection_latency.append(video_collection_duration)

    cap.release()
    cv2.destroyAllWindows()

    video_preprocessing_start_time = get_time(start_time)
    video_frames = [transform(cv2.cvtColor(item, cv2.COLOR_BGR2GRAY)) for item in frames]
    video_tensor = torch.stack(video_frames).unsqueeze(-1).permute(0, 4, 1, 2, 3)
    video_preprocessing_end_time = get_time(start_time)
    video_preprocessing_latency = (video_preprocessing_end_time - video_preprocessing_start_time)
    total_video_preprocessing_latency.append(video_preprocessing_latency)

    frame_cnt = math.floor(19456 / 16000 * 30)

    print(f'Video Collection Start Time:    {video_collection_start_time}')
    print(f'Video Collection End Time:      {video_collection_end_time}')
    print(f'Video Preprocessing Start Time: {video_preprocessing_start_time}')
    print(f'Video Preprocessing End Time:   {video_preprocessing_end_time}')

    print(f'Video Collection Latency:    {total_video_collection_latency[0]}')
    print(f'Video Preprocessing Latency: {total_video_preprocessing_latency[0]}')

    for i in range(sample_num):
        video_sample = video_tensor[i * frame_cnt: (i + 1) * frame_cnt]
        video_encoding_start_time = get_time(start_time)
        video_output = model(video_sample[:video_frame_len])
        video_encoding_end_time = get_time(start_time)
        video_encoding_latency = (video_encoding_end_time - video_encoding_start_time)

        total_video_encode_latency.append(video_encoding_latency)

        video_feature_buffer.put(video_output)

        print(f'Video Encode Start:     {video_encoding_start_time}')
        print(f'Video Encode End:       {video_encoding_end_time}')
        print(f'Video Encode Latency    {video_encoding_latency}')


if __name__ == '__main__':
    video_file = '../data/video.mp4'
    audio_file = '../data/audio.WAV'

    model_path = '../checkpoint'
    mode = 'backendGRU'
    audio_frame_len = 29
    video_frame_len = 29

    label_list = []
    with open('../label_sorted.txt', 'r') as f:
        for line in f:
            label_list.append(line[:-1])

    save_path = '../backendGRU_real_time'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    logger_name = "mylog"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(save_path + '/backendGRU.txt', mode='a')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    audio_model = audio_model_lipreading(mode=mode, inputDim=512, hiddenDim=512, nClasses=500,
                                         frameLen=audio_frame_len)
    video_model = video_model_lipreading(mode=mode, inputDim=256, hiddenDim=512, nClasses=500,
                                         frameLen=video_frame_len)
    concat_model = concat_model_lipreading(mode=mode, inputDim=2048, hiddenDim=512, nLayers=2, nClasses=500)

    print('reload audio model')
    audio_path = os.path.join(model_path, 'Audiovisual_a_part.pt')
    audio_model = reload_model(audio_model, logger, audio_path)
    print("reload video model")
    video_path = os.path.join(model_path, 'Audiovisual_v_part.pt')
    video_model = reload_model(video_model, logger, video_path)
    print("reload concat model")
    concat_path = os.path.join(model_path, 'Audiovisual_c_part.pt')
    concat_model = reload_model(concat_model, logger, concat_path)

    start_time = time()

    # sample_num = math.floor(audio_data.shape[0] / 19456)
    sample_num = 10

    audio_thread = threading.Thread(target=audio_process, args=(audio_file, audio_model))
    video_thread = threading.Thread(target=video_process, args=(video_file, video_model))

    audio_thread.start()
    video_thread.start()

    while True:
        if audio_feature_buffer.qsize() > 0 and video_feature_buffer.qsize() > 0:
            epoch_start_time = get_time(start_time)
            audio_feature = audio_feature_buffer.get()
            video_feature = video_feature_buffer.get()

            fusion_start_time = get_time(start_time)
            cat_input = torch.cat((audio_feature, video_feature), dim=2)
            outputs = concat_model(cat_input)
            fusion_end_time = get_time(start_time)
            fusion_latency = fusion_end_time - fusion_start_time
            total_fusion_latency.append(fusion_latency)

            _, preds = torch.max(F.softmax(outputs.squeeze(), dim=1).data, 1)

            print(f'Fusion Start:           {fusion_start_time}')
            print(f'Fusion End:             {fusion_end_time}')
            print(f'Fusion Latency          {fusion_latency}')
            print('')

            print(f'Predicted words:        {label_list[preds[0]]}')

            epoch_end_time = get_time(start_time)

        if keyboard.is_pressed('q'):  # 检测是否按下了 'q' 键
            break

    end_time = get_time(start_time)
    print(f'Avg Fusion Latency:              {sum(total_fusion_latency) / sample_num:.4f} ms')
    # print(f'Avg Audio Collection Latency:    {sum(total_audio_collection_latency) / sample_num:.4f} ms')
    # print(f'Avg Video Collection Latency:    {sum(total_video_collection_latency) / sample_num:.4f} ms')
    # print(f'Avg Audio Preprocessing Latency: {sum(total_audio_preprocessing_latency) / sample_num:.4f} ms')
    # print(f'Avg Video Preprocessing Latency: {sum(total_video_preprocessing_latency) / sample_num:.4f} ms')
    print(f'Avg Audio Encoding Latency:      {sum(total_audio_encode_latency) / sample_num:.4f} ms')
    print(f'Avg Video Encoding Latency:      {sum(total_video_encode_latency) / sample_num:.4f} ms')
    print(f'Avg Fusion Latency:              {sum(total_fusion_latency) / sample_num:.4f} ms')


