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

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # TODO: 한 영상에 start end 여러번 있는 경우 고려해서 코드 수정하기
        # 정답 frame 담은 dict 만들기
        self.frame_label = dd(lambda: dd(lambda: [-1, -1]))

        folder_list = os.listdir(label_root)

        for folder in folder_list:
            json_list = os.listdir(label_root + "/" + folder)

            for js in json_list:
                with open(label_root + "/" + folder + "/" + js, "r") as j:
                    json_dict = json.load(j)

                for dict in json_dict["annotations"]["track"]:
                    if dict["@label"].endswith("_start"):
                        cur_id = dict["@id"]
                        self.frame_label[js[:-5]][cur_id][0] = dict["box"][0]["@frame"]
                    elif dict["@label"].endswith("_end"):
                        cur_id = dict["@id"]
                        self.frame_label[js[:-5]][cur_id][1] = dict["box"][0]["@frame"]
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

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
            temp = 0
            for cur_id in self.frame_label[target_filename].keys():
                if int(target_frame) >= int(self.frame_label[target_filename][cur_id][0]) and int(
                    target_frame
                ) <= int(self.frame_label[target_filename][cur_id][1]):
                    temp = 1

            target_labels.append(temp)

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


class NormalVMAE(Dataset):
    """
    is_train = 1 <- train, 0 <- test
    """

    def __init__(
        self,
        # is_train=1,
        model_size="small",
        root="/data/ephemeral/home/level2-3-cv-finalproject-cv-06/datapreprocess/npy/normal",
        # label_root="/data/ephemeral/home/level2-3-cv-finalproject-cv-06/datapreprocess/json/abnormal",
    ):
        super().__init__()
        # self.is_train = is_train
        # normal의 경우 torch.utils.data.random_split 함수로 train/val 나눔

        self.path = root

        folder_list = os.listdir(self.path)
        folder_list.sort()

        self.data_list = []

        for folder_name in folder_list:
            print(f"==>> {folder_name} 폴더 데이터 로딩 시작")
            if folder_name.endswith("_base") and model_size == "small":
                continue
            elif not folder_name.endswith("_base") and model_size != "small":
                continue

            folder_path = folder_name + "/"
            data_list = os.listdir(self.path + "/" + folder_path)
            data_list.sort()
            data_list = [folder_path + name for name in data_list]
            self.data_list.extend(data_list)
            print(f"==>> {folder_name} 폴더 데이터 로딩 완료")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_name = self.data_list[idx]

        feature = np.load(self.path + "/" + file_name)
        # 정상 영상 feature는 (57,710) 또는 (38,710)
        if feature.shape[0] % 12 == 0:
            feature_npy = feature
        else:
            count = (feature.shape[0] // 12) + 1

            feature_npy = np.zeros((count * 12, 710))
            # 12로 나눌 수 있도록 (60, 710) or (48, 710) 준비

            feature_npy[: feature.shape[0]] = feature
            # np.load로 불러온 정상영상 feature는 (57, 710) 또는 (38,710)

            feature_npy[feature.shape[0] :] = [feature_npy[-1]] * (count * 12 - feature.shape[0])
            # 정상영상 feature의 마지막 부분으로 빈 자리 채우기

        feature_npy = feature_npy.reshape(12, -1, 710)
        # (12, 5, 710) or (12, 4, 710)
        feature_npy = np.mean(feature_npy, axis=1)
        # 이상행동 영상 feature의 (12,710)과 같아지도록 평균으로 조절

        gts = np.zeros(12)
        # 정상영상은 전부 정답이 0

        return torch.from_numpy(feature_npy).float(), torch.from_numpy(gts).float()


class AbnormalVMAE(Dataset):
    """
    is_train = 1 <- train, 0 <- test
    """

    def __init__(
        self,
        is_train=1,
        model_size="small",
        root="/data/ephemeral/home/level2-3-cv-finalproject-cv-06/datapreprocess/npy/abnormal",
        label_root="/data/ephemeral/home/level2-3-cv-finalproject-cv-06/datapreprocess/json/abnormal",
    ):
        super().__init__()
        self.is_train = is_train

        if self.is_train == 1:
            self.path = root + "/train/"
            self.label_path = label_root + "/train/"
        else:
            self.path = root + "/val/"
            self.label_path = label_root + "/val/"

        label_folder_list = os.listdir(self.label_path)
        label_folder_list.sort()

        self.data_list = []
        self.label_list = dd(lambda: dd(lambda: [-1, -1]))

        for folder_name in label_folder_list:
            print(f"==>> {folder_name} 폴더 데이터 로딩 시작")
            folder_path = (folder_name + "/") if model_size == "small" else (folder_name + "_base/")
            label_folder_path = self.label_path + folder_name + "/"

            label_list = os.listdir(label_folder_path)
            label_list.sort()
            data_list = [folder_path + name[:-4] + "mp4.npy" for name in label_list]
            self.data_list.extend(data_list)

            print(f"==>> {folder_name} 폴더 데이터 JSON 로딩 중")
            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            # TODO: 시간이 엄청 오래걸리는 JSON 로딩부분을 지우고 UCF-CRIME MIL 깃헙처럼 text파일에 미리 frame정보 저장하기
            for js in label_list:
                with open(label_folder_path + js, "r") as j:
                    json_dict = json.load(j)

                for dict in json_dict["annotations"]["track"]:
                    if dict["@label"].endswith("_start"):
                        cur_id = dict["@id"]
                        self.label_list[js[:-5]][cur_id][0] = dict["box"][0]["@frame"]
                    elif dict["@label"].endswith("_end"):
                        cur_id = dict["@id"]
                        self.label_list[js[:-5]][cur_id][1] = dict["box"][0]["@frame"]
            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            print(f"==>> {folder_name} 폴더 데이터 로딩 완료")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_name = self.data_list[idx]

        feature_npy = np.load(self.path + file_name)
        # feature_npy.shape: (12, 710)

        file_name = file_name.split("/")[-1].split(".")[0]

        frame_label = self.label_list[file_name]

        gts = np.zeros(192)
        # 이상행동 영상 180 프레임 => 12 * 16 = 192 가 되도록 길이 연장

        for key, (start, end) in frame_label.items():
            gts[int(start) - 1 : int(end)] = 1

        for i in range(12):
            gts[180 + i] = gts[179]
            # @@ feature extraction할때 마지막 조각에서 frame 개수가 16개가 안되면 마지막 frame을 복사해서 추가함

        if self.is_train:
            gts = gts.reshape(12, 16)
            # (192) => (12, 16)로 변경
            gts = np.mean(gts, axis=1)
            # 평균 내서 (12)로 변경

        # @@ validation일때는 평균내지 않고 (192) numpy array 그대로 반환

        return torch.from_numpy(feature_npy).float(), torch.from_numpy(gts).float()
