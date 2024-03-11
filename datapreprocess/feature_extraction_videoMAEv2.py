# from collections import defaultdict
import argparse
import cv2
import numpy as np
import os
from copy import deepcopy
from datetime import datetime
import torch
# import torch.nn as nn
# import torch.nn.functional as F
import albumentations as A
import matplotlib.pyplot as plt
# from datetime import datetime
import models
from timm.models import create_model
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser(description="Feature Extraction")

parser.add_argument('--name', type=str, help="folder_name")

args = parser.parse_args()


my_folder = "/data/ephemeral/home/DATASET/automated_store/validation/"
root = os.path.join(my_folder, args.name) + "/"

npy_root = "./npys/"

if not os.path.exists(npy_root):
    os.makedirs(npy_root)
if not os.path.exists(npy_root+args.name):
    os.makedirs(npy_root+args.name)

print(torch.cuda.is_available())


file_list = os.listdir(root)
print(f"==>> file_list: {file_list}")
file_list.sort()
print(f"==>> file_list: {file_list}")

segments_num = 1
# 모델에 들어갈 frame수는 16 * segments_num

model = create_model(
    "vit_small_patch16_224",
    img_size=224,
    pretrained=True,
    num_classes=710,
    all_frames=16 * segments_num,
)

load_dict = torch.load(
    "/data/ephemeral/home/level2-3-cv-finalproject-cv-06/datapreprocess/vit_s_k710_dl_from_giant.pth"
)
# load_dict = torch.load(
#     "/data/ephemeral/home/level2-3-cv-finalproject-cv-06/datapreprocess/vit_b_k710_dl_from_giant.pth"
# )
# backbone pth 경로

model.load_state_dict(load_dict["module"])

model.to(device)
model.eval()

tf = A.Resize(224, 224)

# # test
# file_list = file_list[4:6]
# print(f"==>> file_list: {file_list}")

batch_size = 16

# Loop through the video frames
for file_name in file_list:
    path = root + file_name

    time_start = datetime.now()

    print(f"{file_name} feature extracting starts")

    cap = cv2.VideoCapture(path)

    # 710차원 feature array 저장할 list
    np_list = []

    # 16 * segments_num 프레임씩 저장할 list
    frames = []
    frame_count = 0

    # input tensor 저장할 list
    input_list = []
    input_count = 0

    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        # frame.shape = (height, width, 3)

        frame_count += 1  # Increment frame count

        if success:
            frame = tf(image=frame)["image"]
            # frame.shape = (224, 224, 3)

            frame = np.expand_dims(frame, axis=0)
            # frame.shape = (1, 224, 224, 3)
            frames.append(frame.copy())

            if frame_count == 16 * segments_num:
                assert len(frames) == 16 * segments_num
                frames = np.concatenate(frames)
                # in_frames.shape = (16 * segments_num, 224, 224, 3)
                in_frames = frames.transpose(3, 0, 1, 2)
                # # in_frames.shape = (RGB 3, frame T=16 * segments_num, H=224, W=224)
                in_frames = np.expand_dims(in_frames, axis=0)
                # in_frames.shape = (1, 3, 16 * segments_num, 224, 224)
                in_frames = torch.from_numpy(in_frames).float()
                # in_frames.shape == torch.Size([1, 3, 16 * segments_num, 224, 224])

                input_list.append(in_frames.detach().clone())

                frame_count = 0
                frames = []

                input_count += 1

                if input_count == batch_size:
                    # input_batch.shape == torch.Size([batch_size, 3, 16 * segments_num, 224, 224])
                    input_batch = torch.cat(input_list, dim=0).to(device)

                    with torch.no_grad():
                        output = model(input_batch)
                        # output.shape == torch.Size([batch_size, 710])

                    np_list.append(output.cpu().numpy())

                    input_count = 0
                    input_list = []
        else:
            # 남은 프레임, input_list가 지정 개수에서 모자를 때 예외 처리
            if frame_count != 0:
                len_frames_left = 16 * segments_num - len(frames)
                # len_input_list_left = batch_size - len(input_list)
                for i in range(len_frames_left):
                    frames.append(frames[-1].copy())

                assert len(frames) == 16 * segments_num

                frames = np.concatenate(frames)
                # in_frames.shape = (16 * segments_num, 224, 224, 3)
                in_frames = frames.transpose(3, 0, 1, 2)
                # # in_frames.shape = (RGB 3, frame T=16 * segments_num, H=224, W=224)
                in_frames = np.expand_dims(in_frames, axis=0)
                # in_frames.shape = (1, 3, 16 * segments_num, 224, 224)
                in_frames = torch.from_numpy(in_frames).float()
                # in_frames.shape == torch.Size([1, 3, 16 * segments_num, 224, 224])

                input_list.append(in_frames.detach().clone())

                # assert len(input_list) == batch_size

                # input_batch.shape == torch.Size([batch_size, 3, 16 * segments_num, 224, 224])
                input_batch = torch.cat(input_list, dim=0).to(device)

                with torch.no_grad():
                    output = model(input_batch)
                    # output.shape == torch.Size([len(input_list), 710])

                np_list.append(output.cpu().numpy())

                frame_count = 0
                frames = []
                input_count = 0
                input_list = []

            # Break the loop if the end of the video is reached
            break

    file_outputs = np.concatenate(np_list)

    np.save((npy_root + args.name + "/" + file_name), file_outputs)

    cap.release()

    time_end = datetime.now()
    total_time = time_end - time_start
    total_time = str(total_time).split(".")[0]

    print(f"{file_name} feature extracting ended. Elapsed time: {total_time}")

# cv2.destroyAllWindows()

if not os.path.exists(npy_root + args.name +"_base"):
    os.makedirs(npy_root + args.name + "_base")

segments_num = 1
# 모델에 들어갈 frame수는 16 * segments_num

model = create_model(
    "vit_small_patch16_224",
    # "vit_base_patch16_224",
    img_size=224,
    pretrained=True,
    num_classes=710,
    all_frames=16 * segments_num,
    # tubelet_size=args.tubelet_size,
    # drop_rate=args.drop,
    # drop_path_rate=args.drop_path,
    # attn_drop_rate=args.attn_drop_rate,
    # head_drop_rate=args.head_drop_rate,
    # drop_block_rate=None,
    # use_mean_pooling=args.use_mean_pooling,
    # init_scale=args.init_scale,
    # with_cp=args.with_checkpoint,
)

# load_dict = torch.load(
#     "/data/ephemeral/home/level2-3-cv-finalproject-cv-06/datapreprocess/vit_s_k710_dl_from_giant.pth"
# )
load_dict = torch.load(
    "/data/ephemeral/home/level2-3-cv-finalproject-cv-06/datapreprocess/vit_b_k710_dl_from_giant.pth"
)
# backbone pth 경로

model.load_state_dict(load_dict["module"])

model.to(device)
model.eval()

tf = A.Resize(224, 224)

batch_size = 16

# Loop through the video frames
for file_name in tqdm(file_list):
    path = root + file_name

    time_start = datetime.now()

    print(f"{file_name} feature extracting starts")

    cap = cv2.VideoCapture(path)

    # 710차원 feature array 저장할 list
    np_list = []

    # 16 * segments_num 프레임씩 저장할 list
    frames = []
    frame_count = 0

    # input tensor 저장할 list
    input_list = []
    input_count = 0

    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        # frame.shape = (height, width, 3)

        frame_count += 1  # Increment frame count

        if success:
            frame = tf(image=frame)["image"]
            # frame.shape = (224, 224, 3)

            frame = np.expand_dims(frame, axis=0)
            # frame.shape = (1, 224, 224, 3)
            frames.append(frame.copy())

            if frame_count == 16 * segments_num:
                assert len(frames) == 16 * segments_num
                frames = np.concatenate(frames)
                # in_frames.shape = (16 * segments_num, 224, 224, 3)
                in_frames = frames.transpose(3, 0, 1, 2)
                # # in_frames.shape = (RGB 3, frame T=16 * segments_num, H=224, W=224)
                in_frames = np.expand_dims(in_frames, axis=0)
                # in_frames.shape = (1, 3, 16 * segments_num, 224, 224)
                in_frames = torch.from_numpy(in_frames).float()
                # in_frames.shape == torch.Size([1, 3, 16 * segments_num, 224, 224])

                input_list.append(in_frames.detach().clone())

                frame_count = 0
                frames = []

                input_count += 1

                if input_count == batch_size:
                    # input_batch.shape == torch.Size([batch_size, 3, 16 * segments_num, 224, 224])
                    input_batch = torch.cat(input_list, dim=0).to(device)

                    with torch.no_grad():
                        output = model(input_batch)
                        # output.shape == torch.Size([batch_size, 710])

                    np_list.append(output.cpu().numpy())

                    input_count = 0
                    input_list = []
        else:
            # 남은 프레임, input_list가 지정 개수에서 모자를 때 예외 처리
            if frame_count != 0:
                len_frames_left = 16 * segments_num - len(frames)
                # len_input_list_left = batch_size - len(input_list)
                for i in range(len_frames_left):
                    frames.append(frames[-1].copy())

                assert len(frames) == 16 * segments_num

                frames = np.concatenate(frames)
                # in_frames.shape = (16 * segments_num, 224, 224, 3)
                in_frames = frames.transpose(3, 0, 1, 2)
                # # in_frames.shape = (RGB 3, frame T=16 * segments_num, H=224, W=224)
                in_frames = np.expand_dims(in_frames, axis=0)
                # in_frames.shape = (1, 3, 16 * segments_num, 224, 224)
                in_frames = torch.from_numpy(in_frames).float()
                # in_frames.shape == torch.Size([1, 3, 16 * segments_num, 224, 224])

                input_list.append(in_frames.detach().clone())

                # assert len(input_list) == batch_size

                # input_batch.shape == torch.Size([batch_size, 3, 16 * segments_num, 224, 224])
                input_batch = torch.cat(input_list, dim=0).to(device)

                with torch.no_grad():
                    output = model(input_batch)
                    # output.shape == torch.Size([len(input_list), 710])

                np_list.append(output.cpu().numpy())

                frame_count = 0
                frames = []
                input_count = 0
                input_list = []

            # Break the loop if the end of the video is reached
            break

    file_outputs = np.concatenate(np_list)

    np.save((npy_root + args.name + "_base/" + file_name), file_outputs)

    cap.release()

    time_end = datetime.now()
    total_time = time_end - time_start
    total_time = str(total_time).split(".")[0]

    print(f"{file_name} feature extracting ended. Elapsed time: {total_time}")

# cv2.destroyAllWindows()


