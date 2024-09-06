import os
import logging

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import threading
from time import time
from queue import Queue
from dataset import *
from cvtransforms import *
from audio_model import lipreading as audio_model_lipreading
from video_model import lipreading as video_model_lipreading
from concat_model import lipreading as concat_model_lipreading

from Preliminary_Evaluation.utils import get_time


video_feature_buffer = Queue()
audio_feature_buffer = Queue()


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


def load_video_file(filename):
    cap = np.load(filename)['data']
    arrays = np.stack([cv2.cvtColor(cap[_], cv2.COLOR_RGB2GRAY) for _ in range(29)], axis=0)
    arrays = arrays / 255.
    return arrays


def load_audio_file(filename):
    return np.load(filename)['data']


def normalisation(inputs):
    inputs_std = np.std(inputs)
    if inputs_std == 0.:
        inputs_std = 1.
    return (inputs - np.mean(inputs)) / inputs_std


if __name__ == '__main__':
    audio_data_path = './data/lrw_np_audio'
    video_data_path = './data/lrw_np_video'
    batch_size = 1
    num_workers = 1

    model_path = '../checkpoint'
    mode = 'backendGRU'
    audio_frame_len = 29
    video_frame_len = 29

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

    label_list = []
    with open('../label_sorted.txt', 'r') as f:
        for line in f:
            label_list.append(line[:-1])

    audio_model = audio_model_lipreading(mode=mode, inputDim=512, hiddenDim=512, nClasses=500, frameLen=audio_frame_len)
    video_model = video_model_lipreading(mode=mode, inputDim=256, hiddenDim=512, nClasses=500, frameLen=video_frame_len)
    concat_model = concat_model_lipreading(mode=mode, inputDim=2048, hiddenDim=512, nLayers=2, nClasses=500)

    logger.info('Reloading audio model')
    audio_path = os.path.join(model_path, 'Audiovisual_a_part.pt')
    audio_model = reload_model(audio_model, logger, audio_path)
    logger.info("Reloading video model")
    video_path = os.path.join(model_path, 'Audiovisual_v_part.pt')
    video_model = reload_model(video_model, logger, video_path)
    logger.info("Reloading concat model")
    concat_path = os.path.join(model_path, 'Audiovisual_c_part.pt')
    concat_model = reload_model(concat_model, logger, concat_path)

    start_time = time()

    video_data = load_video_file('video.npz')
    audio_data = normalisation(load_audio_file('audio.npz'))
    print(audio_data.shape)
    print(video_data.shape)
    video_inputs = np.reshape(video_data, (1, video_data.shape[0], video_data.shape[1], video_data.shape[2]))
    # audio_inputs = torch.Tensor(np.reshape(audio_data, (1, audio_data.shape[0])))

    audio_data = np.array(audio_data)

    # 将一维数组转换为 PyTorch 张量
    audio_inputs = torch.Tensor(audio_data).view(1, -1)  # 变成 [1, length] 形状

    # 检查长度是否小于 19456，如果是则补齐
    if audio_inputs.size(1) < 19456:
        padding_size = 19456 - audio_inputs.size(1)
        audio_inputs = F.pad(audio_inputs, (0, padding_size), "constant", 0)
    else:
        audio_inputs = audio_inputs[:, :19456]


    video_preprocessing_latency_list = []
    video_encode_latency_list = []
    audio_encode_latency_list = []
    fusion_latency_list = []

    print(audio_inputs.shape)
    print(video_inputs.shape)

    with torch.no_grad():
        # 记录音频编码的开始时间
        audio_encode_start_time = get_time(start_time)
        audio_outputs = audio_model(audio_inputs)
        audio_encode_end_time = get_time(start_time)
        audio_encode_latency_list.append(audio_encode_end_time - audio_encode_start_time)


    video_preprocessing_start_time = get_time(start_time)

    batch_img = CenterCrop(video_inputs, (88, 88))
    batch_img = ColorNormalize(batch_img)
    batch_img = np.reshape(batch_img,
                           (batch_img.shape[0], batch_img.shape[1], batch_img.shape[2], batch_img.shape[3], 1))
    video_inputs = torch.from_numpy(batch_img)
    video_inputs = video_inputs.float().permute(0, 4, 1, 2, 3)

    # 记录视频预处理的结束时间
    video_preprocessing_end_time = get_time(start_time)

    with torch.no_grad():
        # 记录视频编码的开始时间
        video_encode_start_time = get_time(start_time)
        video_outputs = video_model(video_inputs)
        video_encode_end_time = get_time(start_time)

    video_preprocessing_latency_list.append(video_preprocessing_end_time - video_preprocessing_start_time)
    video_encode_latency_list.append(video_encode_end_time - video_encode_start_time)

    with torch.no_grad():
        # 记录融合的开始时间
        fusion_start_time = get_time(start_time)

        # print(audio_outputs.shape, video_outputs.shape)  # torch.Size([20, 29, 1024]) torch.Size([20, 29, 1024])
        inputs = torch.cat((audio_outputs, video_outputs), dim=2)
        outputs = concat_model(inputs)

        outputs = torch.mean(outputs, 1)
        _, preds = torch.max(F.softmax(outputs, dim=1).data, 1)
        print(preds)
        print(f'Predicted words: {label_list[preds[0]]} \n')

    # 记录融合的结束时间
    fusion_end_time = get_time(start_time)
    fusion_latency_list.append(fusion_end_time - fusion_start_time)

    # logger.info(f'Batch {batch_idx}: Correct: {correct}, Total: {audio_inputs.shape[0]}')
    #
    # logger.info(f'Final Accuracy: {total_correct / total * 100:.4f}%')
    # logger.info(f'Avg Video Preprocessing Latency: {sum(video_preprocessing_latency_list) / len(video_preprocessing_latency_list):.6f} ms')
    # logger.info(f'Avg Audio Encode Latency:        {sum(audio_encode_latency_list) / len(audio_encode_latency_list):.6f} ms')
    # logger.info(f'Avg Video E(ncode Latency:       {sum(video_encode_latency_list) / len(video_encode_latency_list):.6f} ms')
    # logger.info(f'Avg Fusion Latency:              {sum(fusion_latency_list) / len(fusion_latency_list):.6f} ms')

    logging.shutdown()
