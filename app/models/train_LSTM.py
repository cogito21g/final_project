import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split

# from torch.utils.data import DataLoader
# from torch.utils.data import random_split

from sklearn.preprocessing import MinMaxScaler

from argparse import ArgumentParser

import os
import os.path as osp

from tqdm import tqdm


class NormalDataset(Dataset):

    def __init__(
        self,
        sequence_length=20,
        prediction_time=1,
        root="/data/ephemeral/home/level2-3-cv-finalproject-cv-06/datapreprocess/csv",
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.prediction_time = prediction_time

        self.scaler = MinMaxScaler()
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

        return torch.from_numpy(sequence).float(), torch.from_numpy(target).float()

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


class LSTMAutoencoder(nn.Module):
    def __init__(self, sequence_length, n_features, prediction_time):
        super(LSTMAutoencoder, self).__init__()

        self.sequence_length = sequence_length
        self.n_features = n_features
        self.prediction_time = prediction_time

        # Encoder
        self.encoder = nn.LSTM(input_size=n_features, hidden_size=100, batch_first=True)
        self.encoder2 = nn.LSTM(input_size=100, hidden_size=50, batch_first=True)

        # Repeat vector for prediction_time
        self.repeat_vector = nn.Sequential(
            nn.ReplicationPad1d(padding=(0, prediction_time - 1)),
            nn.ReplicationPad1d(padding=(0, 0)),  # Adjusted padding
        )

        # Decoder
        self.decoder = nn.LSTM(input_size=50, hidden_size=100, batch_first=True)
        self.decoder2 = nn.LSTM(input_size=100, hidden_size=n_features, batch_first=True)

    def forward(self, x):
        # Encoder
        # _, (x, _) = self.encoder(x)
        x, (_, _) = self.encoder(x)
        # output, (hn, cn) = rnn(x)
        x, (_, _) = self.encoder2(x)

        # Repeat vector for prediction_time
        x = self.repeat_vector(x)

        # Decoder
        x, (_, _) = self.decoder(x)
        x, (_, _) = self.decoder2(x)

        return x[:, -1, :].unsqueeze(dim=1)


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument(
        "--root_dir",
        type=str,
        default=os.environ.get(
            "SM_CHANNEL_TRAIN_CSV", "/data/ephemeral/home/level2-3-cv-finalproject-cv-06/datapreprocess/csv"
        ),
    )
    # 학습 데이터 경로
    parser.add_argument(
        "--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/data/ephemeral/home/pths")
    )
    # pth 파일 저장 경로

    parser.add_argument("--model_name", type=str, default="LSTM")
    # import_module로 불러올 model name

    parser.add_argument("--resume_name", type=str, default="")
    # resume 파일 이름

    parser.add_argument("--seed", type=int, default=666)
    # random seed

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument("--val_num_workers", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--max_epoch", type=int, default=20)

    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--val_interval", type=int, default=1)

    parser.add_argument("--patience", type=int, default=10)

    # parser.add_argument("--mp", action="store_false")
    # https://stackoverflow.com/questions/60999816/argparse-not-parsing-boolean-arguments
    # mixed precision 사용할 지 여부

    parser.add_argument("--wandb_mode", type=str, default="online")
    # parser.add_argument("--wandb_mode", type=str, default="disabled")
    # wandb mode
    parser.add_argument("--wandb_run_name", type=str, default="LSTM")
    # wandb run name

    args = parser.parse_args()

    return args


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def train(
    root_dir,
    model_dir,
    model_name,
    device,
    num_workers,
    batch_size,
    val_num_workers,
    val_batch_size,
    learning_rate,
    max_epoch,
    val_interval,
    save_interval,
    patience,
    resume_name,
    seed,
    # mp,
    wandb_mode,
    wandb_run_name,
):

    time_start = datetime.now()

    train_start = time_start.strftime("%Y%m%d_%H%M%S")

    set_seed(seed)

    if not osp.exists(model_dir):
        os.makedirs(model_dir)

    # Define parameters
    sequence_length = 20  # Adjust as needed
    prediction_time = 1  # Adjust as needed
    n_features = 38  # Number of features to predict

    batch_size = batch_size
    val_batch_size = val_batch_size

    # -- early stopping flag
    patience = patience
    counter = 0

    # 데이터셋
    dataset = NormalDataset(
        root=root_dir,
    )

    valid_data_size = len(dataset) // 10

    train_data_size = len(dataset) - valid_data_size

    train_dataset, valid_dataset = random_split(dataset, lengths=[train_data_size, valid_data_size])

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    valid_loader = DataLoader(
        dataset=valid_dataset, batch_size=val_batch_size, shuffle=False, num_workers=val_num_workers
    )

    data_load_end = datetime.now()
    data_load_time = data_load_end - time_start
    data_load_time = str(data_load_time).split(".")[0]
    print(f"==>> data_load_time: {data_load_time}")

    # Initialize the LSTM autoencoder model
    model = LSTMAutoencoder(sequence_length, n_features, prediction_time)

    # load_dict = None

    # if resume_name:
    #     load_dict = torch.load(osp.join(model_dir, f"{resume_name}.pth"), map_location="cpu")
    #     model.load_state_dict(load_dict["model_state_dict"])

    model.load_state_dict(
        torch.load(
            "/data/ephemeral/home/level2-3-cv-finalproject-cv-06/app/models/pytorch_model.pth",
            map_location="cpu",
        )
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)

    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)

    # if resume_name:
    #     optimizer.load_state_dict(load_dict["optimizer_state_dict"])
    #     scheduler.load_state_dict(load_dict["scheduler_state_dict"])
    #     scaler.load_state_dict(load_dict["scaler_state_dict"])

    criterion = nn.MSELoss()

    print(f"Start training..")

    # wandb.init(
    #     project="VAD",
    #     entity="pao-kim-si-woong",
    #     config={
    #         "lr": learning_rate,
    #         "dataset": "무인매장",
    #         "n_epochs": max_epoch,
    #         "loss": "MSE",
    #         "notes": "VAD 실험",
    #     },
    #     name=wandb_run_name + "_" + train_start,
    #     mode=wandb_mode,
    # )

    # wandb.watch((model,))

    best_loss = np.inf

    total_batches = len(train_loader)

    for epoch in range(max_epoch):
        model.train()

        epoch_start = datetime.now()

        epoch_loss = 0

        for step, (x, y) in tqdm(enumerate(train_loader), total=total_batches):

            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            pred = model(x)

            loss = criterion(pred, y)

            loss.backward()
            optimizer.step()

            epoch_loss += loss

        epoch_mean_loss = (epoch_loss / total_batches).item()

        train_end = datetime.now()
        train_time = train_end - epoch_start
        train_time = str(train_time).split(".")[0]
        print(f"==>> epoch {epoch+1} train_time: {train_time}\nloss: {round(epoch_mean_loss,4)}")

        if (epoch + 1) % save_interval == 0:

            ckpt_fpath = osp.join(model_dir, f"{model_name}_{train_start}_latest.pth")

            states = {
                "epoch": epoch,
                "model_name": model_name,
                "model_state_dict": model.state_dict(),  # 모델의 state_dict 저장
                "optimizer_state_dict": optimizer.state_dict(),
                # "scheduler_state_dict": scheduler.state_dict(),
                # "scaler_state_dict": scaler.state_dict(),
            }

            torch.save(states, ckpt_fpath)

        # validation 주기에 따라 loss를 출력하고 best model을 저장합니다.
        if (epoch + 1) % val_interval == 0:

            print(f"Start validation #{epoch+1:2d}")
            model.eval()

            with torch.no_grad():
                total_loss = 0

                for step, (x, y) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                    x, y = x.to(device), y.to(device)

                    pred = model(x)

                    val_loss = criterion(pred, y)

                    total_loss += val_loss

                val_mean_loss = (total_loss / len(valid_loader)).item()

            if best_loss > val_mean_loss:
                print(f"Best performance at epoch: {epoch + 1}, {best_loss:.4f} -> {val_mean_loss:.4f}")
                print(f"Save model in {model_dir}")
                states = {
                    "epoch": epoch,
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),  # 모델의 state_dict 저장
                    # "optimizer_state_dict": optimizer.state_dict(),
                    # "scheduler_state_dict": scheduler.state_dict(),
                    # "scaler_state_dict": scaler.state_dict(),
                    # best.pth는 inference에서만 쓰기?
                }

                best_ckpt_fpath = osp.join(model_dir, f"{model_name}_{train_start}_best.pth")
                torch.save(states, best_ckpt_fpath)
                best_loss = val_mean_loss
                counter = 0
            else:
                counter += 1

        # new_wandb_metric_dict = {
        #     "valid_dice": avg_dice,
        #     "valid_loss": val_mean_loss,
        #     "valid_f_loss": val_mean_f_loss,
        #     "train_bce_loss": epoch_mean_b_loss,
        #     "train_focal_loss": epoch_mean_f_loss,
        #     "train_dice_loss": epoch_mean_d_loss,
        #     # "train_loss": epoch_mean_loss,
        #     "train_loss": epoch_mean_b_loss + epoch_mean_d_loss,
        #     "learning_rate": scheduler.get_lr()[0],
        # }

        # wandb.log(new_wandb_metric_dict)

        # scheduler.step()

        epoch_end = datetime.now()
        epoch_time = epoch_end - epoch_start
        epoch_time = str(epoch_time).split(".")[0]
        print(f"==>> epoch {epoch+1} time: {epoch_time}\nvalid_loss: {round(val_mean_loss,4)}")
        if counter > patience:
            print("Early Stopping...")
            break

    time_end = datetime.now()
    total_time = time_end - time_start
    total_time = str(total_time).split(".")[0]
    print(f"==>> total time: {total_time}")


def main(args):
    train(**args.__dict__)


if __name__ == "__main__":
    args = parse_args()

    main(args)
