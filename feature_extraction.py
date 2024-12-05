import os
import sys

sys.path.append("LAVIS")

import torch
import requests
import decord
import torch
import argparse
import glob
import cv2

from io import BytesIO
from PIL import Image
from lavis.models import load_model_and_preprocess

import matplotlib.pyplot as plt
import numpy as np
import tqdm
import pickle
import random

random.seed(42)

SAMPLING_RATE = 50
TARGET_FPS = 12
CLIP_IDX = -1 # val : -1, test : self._spatial_temporal_index[index] // self.cfg.TEST.NUM_SPATIAL_CROPS
NUM_CLIPS = 1
NUM_FRAMES = 8

def interval_based_sampling(vid_length, vid_fps, clip_idx, num_clips, num_frames, interval):
    if num_frames == 1:
        index = [random.randint(0, vid_length-1)]
    else:
        # if split == "train" and hasattr(cfg.DATA, "SAMPLING_RATE_TRAIN"):
        #     interval = cfg.DATA.SAMPLING_RATE_TRAIN
        #     clip_length = num_frames * interval * vid_fps / cfg.DATA.TARGET_FPS
        # elif hasattr(cfg.DATA, "SAMPLING_RATE_TEST") and cfg.DATA.SAMPLING_RATE_TEST>40:
        #     interval = vid_length//num_frames
        #     clip_length = vid_length//num_frames * num_frames
        #     index = [random.randint(ind*interval, ind*interval+interval-1) for ind in range(num_frames)]
        #     return index
        # if SAMPLING_RATE >40:  # SAMPLING_RATE_TEST
        #     interval = vid_length//num_frames
        #     clip_length = vid_length//num_frames * num_frames
        #     index = [random.randint(ind*interval, ind*interval+interval-1) for ind in range(num_frames)]
        #     return index
        # else:
        # transform FPS
        clip_length = num_frames * interval * vid_fps / TARGET_FPS
        
        if clip_length > vid_length:
            clip_length = vid_length//num_frames * num_frames

        max_idx = max(vid_length - clip_length + 1, 0)
        if clip_idx == -1: # random sampling
            start_idx = random.uniform(0, max_idx)
        else:
            if num_clips == 1:
                start_idx = max_idx / 2
            else:
                start_idx = max_idx * clip_idx / num_clips
        end_idx = start_idx + clip_length - interval

        index = torch.linspace(start_idx, end_idx, num_frames)
        index = torch.clamp(index, 0, vid_length-1).long()

    return index

parser = argparse.ArgumentParser()
parser.add_argument("--video_path", required=True, type=str)
parser.add_argument("--seg", type=int, default=8)
parser.add_argument("--save_path", required=True, type=str)

args = parser.parse_args()

def get_videos(video_path):
    # video_list = glob.glob(os.path.join(video_path, "*.webm"))
    video_list = glob.glob(os.path.join(video_path, "*", "*.avi"))
    return video_list

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device)

video_list = get_videos(args.video_path)

# os.makedirs(args.save_path, exist_ok=True)

for video_path in tqdm.tqdm(video_list):
    file_name = os.path.basename(video_path).replace('.avi', '')
    dir_name = os.path.dirname(video_path).split(os.sep)[-1]
    os.makedirs(os.path.join(args.save_path, dir_name), exist_ok=True)
    with open(video_path, 'rb') as f:
        vr = decord.VideoReader(f, ctx=decord.cpu(0), num_threads=1)
    
    video_length = len(vr)
    
    frame_idx = interval_based_sampling(video_length, vr.get_avg_fps(), -1, NUM_CLIPS, NUM_FRAMES, SAMPLING_RATE)
    # print(frame_idx)
    # break
    # frame_idx = list(range(0, video_length, video_length//args.seg))
    # frame_idx = [idx + (video_length/args.seg) // 2 for idx in frame_idx][:8]
    frames = vr.get_batch(frame_idx).numpy()
    image = torch.stack([vis_processors["eval"](cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]).to(device)
    # image = vis_processors["eval"](frames).unsqueeze(0).to(device)
    sample = {"image": image, "text_input": [f"a photo of {dir_name.replace('_', ' ')}"] * image.shape[0]}
    with torch.no_grad():
        image_features = model.extract_features(sample, mode='image')
        multimodal_features = model.extract_features(sample, mode='multimodal')
    image_embeds = image_features['image_embeds']
    image_embeds = image_embeds.detach().cpu().numpy()
    multimodal_embeds = multimodal_features['multimodal_embeds']
    multimodal_embeds = multimodal_embeds.detach().cpu().numpy()
    # save
    with open(os.path.join(args.save_path, dir_name, file_name+'.pickle'), 'wb') as f:
        pickle.dump({"image_embeds" : image_embeds, 
                     "multimodal_embeds" : multimodal_embeds, 
                     'label' : dir_name, 
                     'input_text' : f"a photo of {dir_name.replace('_', ' ')}",
                     "video_length" : video_length,
                     "frame_index" : frame_idx}, f, pickle.HIGHEST_PROTOCOL)
    # np.save(os.path.join(args.save_path, dir_name, file_name+'.npy'), image_embeds)
    