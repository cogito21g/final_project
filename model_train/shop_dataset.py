import os
import os.path as osp

from torch.utils.data import Dataset

# from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np
import torch

import json
from collections import defaultdict as dd


class NormalDataset(Dataset):

    def __init__(
        self,
        sequence_length=20,
        prediction_time=1,
        root="/data/ephemeral/home/level2-3-cv-finalproject-cv-06/datapreprocess/csv/normal/val",
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.prediction_time = prediction_time

        # self.scaler = MinMaxScaler()
        # 데이터 값 [0,1] 범위로 scaling할때 사용

        # Load the dataset
        file_list = os.listdir(root)

        df_list = []

        self.length = 0
        self.range_table = []

        self.real_length = 0
        self.real_idx_table = []

        for i, file_name in enumerate(file_list):
            dat = pd.read_csv(root + "/" + file_name)
            dat.drop(columns=["Frame"], inplace=True)  # Remove the 'Frame' column

            print(f"==>>{i}번째 dat.shape: {dat.shape}")

            id_counter = pd.Series(dat["ID"]).value_counts(sort=False)

            for id_to_del in id_counter[id_counter < sequence_length + prediction_time].index:
                dat.drop(dat[dat["ID"] == id_to_del].index, inplace=True)

            id_counter = pd.Series(dat["ID"]).value_counts(sort=False)

            print(f"==>>{i}번째 처리 후 dat.shape: {dat.shape}")
            assert len(id_counter[id_counter < sequence_length + prediction_time].index) == 0

            for count in id_counter:
                cur_id_length = count - sequence_length - prediction_time + 1
                self.range_table.append(self.length + cur_id_length)
                self.real_idx_table.append(self.real_length + count)
                self.length += cur_id_length
                self.real_length += count

            dat["ID"] = dat["ID"].astype("str") + f"_{i}"
            df_list.append(dat.copy())

        self.dat = pd.concat(df_list, ignore_index=True)
        # self.dat.drop(columns=["Frame"], inplace=True)  # Remove the 'Frame' column

        id_counter = pd.Series(self.dat["ID"]).value_counts(sort=False)

        # # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # DF를 다 합치고 나서 ID를 거르면 데이터셋 초기화에 1분정도 걸리지만
        # DF 조각마다 ID를 거르고나서 합치면 6초 밖에 안 걸린다
        # self.checker = []

        # for id_to_del in id_counter[id_counter < sequence_length + prediction_time].index:
        #     self.checker.append((id_to_del, id_counter[id_to_del]))

        #     self.dat.drop(self.dat[self.dat["ID"] == id_to_del].index, inplace=True)

        # # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # # sequence_length + prediction_time 보다 짧은 ID를 지우는 것을 한번만 하면
        # # 13개 ID가 sequence_length + prediction_time보다 짧은데도 남아 있다???
        # id_counter = pd.Series(self.dat["ID"]).value_counts(sort=False)

        # if len(id_counter[id_counter < sequence_length + prediction_time].index) != 0:
        #     for id_to_del in id_counter[id_counter < sequence_length + prediction_time].index:
        #         self.checker.append((id_to_del, id_counter[id_to_del]))

        #         self.dat.drop(self.dat[self.dat["ID"] == id_to_del].index, inplace=True)

        # id_counter = pd.Series(self.dat["ID"]).value_counts(sort=False)
        # # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        assert len(id_counter[id_counter < sequence_length + prediction_time].index) == 0

        # self.length = 0

        # self.range_table = []

        # for count in id_counter:
        #     cur_id_length = count - sequence_length - prediction_time + 1
        #     self.range_table.append(self.length + cur_id_length)
        #     self.length += cur_id_length

        # self.dat.drop(columns=["ID"], inplace=True)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        real_idx = self.find_real_idx(idx)

        sequence = self.dat[real_idx : real_idx + self.sequence_length].copy()
        sequence.drop(columns=["ID"], inplace=True)
        sequence = np.array(sequence)
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # sequence = self.scaler.fit_transform(sequence.values)
        # # 데이터 값 [min, max] -> [0,1] 범위로 scaling
        # scale 된 후에는 numpy array로 변환된다
        # sequence나 target은 이미 yolo v8에서 xywhn, xyn으로 0~1 범위인데 scaling을 한번 더 할 필요가 있을지?
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # (self.sequence_length, 38)
        target = self.dat[
            real_idx + self.sequence_length : real_idx + self.sequence_length + self.prediction_time
        ].copy()
        target.drop(columns=["ID"], inplace=True)
        target = np.array(target)
        # target = self.scaler.fit_transform(target.values)
        # (self.prediction_time, 38)

        label = torch.LongTensor([0 for i in range(self.prediction_time)])

        return (
            torch.from_numpy(sequence).float(),
            torch.from_numpy(target).float(),
            label,
        )

    def find_real_idx(self, idx):

        start = 0
        end = len(self.range_table) - 1
        while start <= end:
            mid = (start + end) // 2
            if self.range_table[mid] == idx:
                real_idx = idx + ((mid + 1) * (self.sequence_length + self.prediction_time - 1))
                return real_idx

            if self.range_table[mid] > idx:
                end = mid - 1
            else:
                start = mid + 1

        real_idx = idx + (start * (self.sequence_length + self.prediction_time - 1))

        return real_idx


class AbnormalDataset(Dataset):

    def __init__(
        self,
        sequence_length=20,
        prediction_time=1,
        root="/data/ephemeral/home/level2-3-cv-finalproject-cv-06/datapreprocess/csv/abnormal/val",
        label_root="/data/ephemeral/home/level2-3-cv-finalproject-cv-06/datapreprocess/json/abnormal/val",
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.prediction_time = prediction_time

        # self.scaler = MinMaxScaler()
        # 데이터 값 [0,1] 범위로 scaling할때 사용

        # Load the dataset
        file_list = os.listdir(root)

        df_list = []

        self.length = 0
        self.range_table = []

        self.real_length = 0
        self.real_idx_table = []

        for i, file_name in enumerate(file_list):
            dat = pd.read_csv(root + "/" + file_name)
            # dat.drop(columns=["Frame"], inplace=True)  # Remove the 'Frame' column

            print(f"==>>{i}번째 dat.shape: {dat.shape}")

            id_counter = pd.Series(dat["ID"]).value_counts(sort=False)

            for id_to_del in id_counter[id_counter < sequence_length + prediction_time].index:
                dat.drop(dat[dat["ID"] == id_to_del].index, inplace=True)

            id_counter = pd.Series(dat["ID"]).value_counts(sort=False)

            print(f"==>>{i}번째 처리 후 dat.shape: {dat.shape}")
            assert len(id_counter[id_counter < sequence_length + prediction_time].index) == 0

            for count in id_counter:
                cur_id_length = count - sequence_length - prediction_time + 1
                self.range_table.append(self.length + cur_id_length)
                self.real_idx_table.append(self.real_length + count)
                self.length += cur_id_length
                self.real_length += count

            dat["ID"] = dat["ID"].astype("str") + f"_{i}"
            df_list.append(dat.copy())

        self.dat = pd.concat(df_list, ignore_index=True)
        # self.dat.drop(columns=["Frame"], inplace=True)  # Remove the 'Frame' column

        id_counter = pd.Series(self.dat["ID"]).value_counts(sort=False)

        assert len(id_counter[id_counter < sequence_length + prediction_time].index) == 0

        # 정답 frame 담은 dict 만들기
        self.frame_label = dd(lambda: [-1, -1])

        folder_list = os.listdir(label_root)

        for folder in folder_list:
            json_list = os.listdir(label_root + "/" + folder)

            for js in json_list:
                with open(label_root + "/" + folder + "/" + js, "r") as j:
                    json_dict = json.load(j)

                count = 0

                for dict in json_dict["annotations"]["track"]:
                    if dict["@label"].endswith("_start"):
                        self.frame_label[js[:-5]][0] = dict["box"][0]["@frame"]
                        count += 1
                    elif dict["@label"].endswith("_end"):
                        self.frame_label[js[:-5]][1] = dict["box"][0]["@frame"]
                        count += 1
                    elif count == 2:
                        break

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        real_idx = self.find_real_idx(idx)

        sequence = self.dat[real_idx : real_idx + self.sequence_length].copy()
        sequence.drop(columns=["ID"], inplace=True)
        sequence.drop(columns=["Frame"], inplace=True)
        sequence.drop(columns=["Filename"], inplace=True)
        sequence = np.array(sequence)
        # (self.sequence_length, 38)
        target = self.dat[
            real_idx + self.sequence_length : real_idx + self.sequence_length + self.prediction_time
        ].copy()
        target_frames = target["Frame"].unique()
        target_filename = target["Filename"].unique()[0].split(".")[0]

        target.drop(columns=["ID"], inplace=True)
        target.drop(columns=["Frame"], inplace=True)
        target.drop(columns=["Filename"], inplace=True)
        target = np.array(target)
        # target = self.scaler.fit_transform(target.values)
        # (self.prediction_time, 38)

        target_labels = []

        for target_frame in target_frames:

            if int(target_frame) >= int(self.frame_label[target_filename][0]) and int(target_frame) <= int(
                self.frame_label[target_filename][1]
            ):
                target_labels.append(1)
            else:
                target_labels.append(0)

        target_labels = torch.LongTensor(target_labels)

        return torch.from_numpy(sequence).float(), torch.from_numpy(target).float(), target_labels

    def find_real_idx(self, idx):

        start = 0
        end = len(self.range_table) - 1
        while start <= end:
            mid = (start + end) // 2
            if self.range_table[mid] == idx:
                real_idx = idx + ((mid + 1) * (self.sequence_length + self.prediction_time - 1))
                return real_idx

            if self.range_table[mid] > idx:
                end = mid - 1
            else:
                start = mid + 1

        real_idx = idx + (start * (self.sequence_length + self.prediction_time - 1))

        return real_idx
