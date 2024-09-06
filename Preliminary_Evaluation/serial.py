import math
import os
import cv2
import wave
import ffmpeg
import pyaudio
import logging
import torch
import speech_recognition as sr
import torch.nn.functional as F
import numpy as np

from time import time

from cvtransforms import CenterCrop, ColorNormalize
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


def collect_audio(wav_file):
    r = sr.Recognizer()
    with sr.AudioFile(wav_file) as source:
        audio = r.record(source)  # 读取整个文件

    # 将音频数据转换为 NumPy 数组
    data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)

    return data


def collect_video(file_path):
    cap = cv2.VideoCapture(file_path)

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

    cap.release()
    cv2.destroyAllWindows()

    return frames


if __name__ == '__main__':
    sample_num = 88
    video_file = '../data/video_2.mp4'
    audio_file = '../data/audio_2.WAV'

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

    total_fusion_latency = []
    total_video_collection_latency = []
    total_audio_collection_latency = []
    total_video_preprocessing_latency = []
    total_audio_preprocessing_latency = []
    total_video_encode_latency = []
    total_audio_encode_latency = []

    start_time = time()

    audio_collection_start_time = get_time(start_time)
    audio_data = collect_audio(audio_file)
    audio_collection_end_time = get_time(start_time)
    audio_collection_duration = (audio_collection_end_time - audio_collection_start_time) 
    total_audio_collection_latency.append(audio_collection_duration)

    # sample_num = math.floor(audio_data.shape[0] / 19456)
    sample_num = 10

    video_collection_start_time = get_time(start_time)
    video_data = collect_video(video_file)
    video_collection_end_time = get_time(start_time)
    video_collection_duration = (video_collection_end_time - video_collection_start_time) 
    total_video_collection_latency.append(video_collection_duration)

    audio_preprocessing_start_time = get_time(start_time)
    audio_tensor = torch.Tensor(normalisation(audio_data)).unsqueeze(0)
    audio_preprocessing_end_time = get_time(start_time)
    audio_preprocessing_latency = (audio_preprocessing_end_time - audio_preprocessing_start_time) 
    total_audio_preprocessing_latency.append(audio_preprocessing_latency)

    video_preprocessing_start_time = get_time(start_time)
    video_frames = np.stack([cv2.cvtColor(item, cv2.COLOR_BGR2GRAY) for item in video_data], axis=0)
    video_frames = video_frames / 255.  # 29, 96, 96
    video_frames = np.reshape(video_frames, (1, video_frames.shape[0], video_frames.shape[1], video_frames.shape[2]))

    # print(video_frames.shape)

    batch_img = CenterCrop(video_frames, (88, 88))
    batch_img = ColorNormalize(batch_img)
    batch_img = np.reshape(batch_img,
                           (batch_img.shape[0], batch_img.shape[1], batch_img.shape[2], batch_img.shape[3], 1))
    # (1, 29, 88, 88, 1)

    video_tensor = torch.from_numpy(batch_img)
    video_tensor = video_tensor.float().permute(0, 4, 1, 2, 3)

    video_preprocessing_end_time = get_time(start_time)
    video_preprocessing_latency = (video_preprocessing_end_time - video_preprocessing_start_time)
    total_video_preprocessing_latency.append(video_preprocessing_latency)

    print(f'Audio Collection Start Time:    {audio_collection_start_time}')
    print(f'Audio Collection End Time:      {audio_collection_end_time}')
    print(f'Video Collection Start Time:    {video_collection_start_time}')
    print(f'Video Collection End Time:      {video_collection_end_time}')
    print(f'Audio Preprocessing Start Time: {audio_preprocessing_start_time}')
    print(f'Audio Preprocessing End Time:   {audio_preprocessing_end_time}')
    print(f'Video Preprocessing Start Time: {video_preprocessing_start_time}')
    print(f'Video Preprocessing End Time:   {video_preprocessing_end_time}')

    print('')

    print(f'Audio Collection Latency:    {total_audio_collection_latency[0]}')
    print(f'Video Collection Latency:    {total_video_collection_latency[0]}')
    print(f'Audio Preprocessing Latency: {total_audio_preprocessing_latency[0]}')
    print(f'Video Preprocessing Latency: {total_video_preprocessing_latency[0]}')

    for i in range(sample_num):
        epoch_start_time = get_time(start_time)

        audio_sample = audio_tensor[:, 24000 * i: 24000 * (i + 1)][:, -19456:]
        frame_cnt = 30
        video_sample = video_tensor[:, :, frame_cnt * i: frame_cnt * (i + 1)][:, :, :29]

        audio_encoding_start_time = get_time(start_time)
        audio_output = audio_model(audio_sample)
        audio_encoding_end_time = get_time(start_time)
        audio_encoding_latency = (audio_encoding_end_time - audio_encoding_start_time) 

        # print(audio_sample.shape)  # torch.Size([1, 19456])
        # print(video_sample.shape)  # torch.Size([48, 1, 1, 88, 88])

        video_encoding_start_time = get_time(start_time)
        video_output = video_model(video_sample[:video_frame_len])
        video_encoding_end_time = get_time(start_time)
        video_encoding_latency = (video_encoding_end_time - video_encoding_start_time) 

        total_audio_encode_latency.append(audio_encoding_latency)
        total_video_encode_latency.append(video_encoding_latency)

        fusion_start_time = get_time(start_time)
        cat_input = torch.cat((audio_output, video_output), dim=2)
        outputs = concat_model(cat_input)
        fusion_end_time = get_time(start_time)
        fusion_latency = fusion_end_time - fusion_start_time
        total_fusion_latency.append(fusion_latency)

        _, preds = torch.max(F.softmax(outputs.squeeze(), dim=1).data, 1)

        print(f'Audio Encode Start:     {audio_encoding_start_time}')
        print(f'Audio Encode End:       {audio_encoding_end_time}')
        print(f'Video Encode Start:     {video_encoding_start_time}')
        print(f'Video Encode End:       {video_encoding_end_time}')
        print(f'Fusion Start:           {fusion_start_time}')
        print(f'Fusion End:             {fusion_end_time}')

        print('')
        print(f'Audio Encode Latency    {audio_encoding_latency}')
        print(f'Video Encode Latency    {video_encoding_latency}')
        print(f'Fusion Latency          {fusion_latency}')
        print('')

        print(f'Predicted words:        {label_list[preds[0]]}')

        epoch_end_time = get_time(start_time)

    end_time = get_time(start_time)
    print(f'Avg Fusion Latency:              {sum(total_fusion_latency) / sample_num:.4f} ms')
    # print(f'Avg Audio Collection Latency:    {sum(total_audio_collection_latency) / sample_num:.4f} ms')
    # print(f'Avg Video Collection Latency:    {sum(total_video_collection_latency) / sample_num:.4f} ms')
    # print(f'Avg Audio Preprocessing Latency: {sum(total_audio_preprocessing_latency) / sample_num:.4f} ms')
    # print(f'Avg Video Preprocessing Latency: {sum(total_video_preprocessing_latency) / sample_num:.4f} ms')
    print(f'Avg Audio Encoding Latency:      {sum(total_audio_encode_latency) / sample_num:.4f} ms')
    print(f'Avg Video Encoding Latency:      {sum(total_video_encode_latency) / sample_num:.4f} ms')
    print(f'Avg Fusion Latency:              {sum(total_fusion_latency) / sample_num:.4f} ms')

    x = range(1, sample_num + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(x, total_fusion_latency, label='Fusion Latency')
    # plt.plot(x, total_audio_collection_latency, label='Audio Collection Latency')
    # plt.plot(x, total_video_collection_latency, label='Video Collection Latency')
    # plt.plot(x, total_audio_preprocessing_latency, label='Audio Preprocessing Latency')
    # plt.plot(x, total_video_preprocessing_latency, label='Video Preprocessing Latency')
    plt.plot(x, total_audio_encode_latency, label='Audio Encoding Latency')
    plt.plot(x, total_video_encode_latency, label='Video Encoding Latency')

    # 添加图例
    plt.legend()

    # 添加标题和标签
    plt.title('Latency Analysis')
    plt.xlabel('Sample Number')
    plt.ylabel('Latency (ms)')

    # 显示图表
    plt.grid(True)
    plt.show()
