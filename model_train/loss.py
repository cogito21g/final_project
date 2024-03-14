import torch
import torch.nn.functional as F


def MIL(y_pred, batch_size, is_transformer=0):
    loss = torch.tensor(0.0).cuda()
    loss_intra = torch.tensor(0.0).cuda()
    sparsity = torch.tensor(0.0).cuda()
    smooth = torch.tensor(0.0).cuda()
    if is_transformer == 0:
        y_pred = y_pred.view(batch_size, -1)
        # (30*24, 1)을 (30, 24)로 다시 변경
        # dim=1 24 = 이상12 + 정상12
    else:
        y_pred = torch.sigmoid(y_pred)

    # print(f"==>> y_pred.shape: {y_pred.shape}")

    for i in range(batch_size):
        # anomaly_index = torch.randperm(12).cuda()
        # print(f"==>> anomaly_index: {anomaly_index}")
        # normal_index = torch.randperm(12).cuda()
        # print(f"==>> normal_index: {normal_index}")

        # print(f"==>> y_pred[i, :12].shape: {y_pred[i, :12].shape}")
        # print(f"==>> y_pred[i, 12:].shape: {y_pred[i, 12:].shape}")

        y_anomaly = y_pred[i, :12]
        # y_anomaly = y_pred[i, :12][anomaly_index]
        # print(f"==>> y_anomaly.shape: {y_anomaly.shape}")
        # MIL 논문의 segment 개수 32와 다르게 무인매장 데이터셋 feature는 12 segment
        y_normal = y_pred[i, 12:]
        # y_normal = y_pred[i, 12:][normal_index]
        # print(f"==>> y_normal.shape: {y_normal.shape}")

        y_anomaly_max = torch.max(y_anomaly)  # anomaly
        y_anomaly_min = torch.min(y_anomaly)

        y_normal_max = torch.max(y_normal)  # normal
        y_normal_min = torch.min(y_normal)

        loss += F.relu(1.0 - y_anomaly_max + y_normal_max)

        sparsity += torch.sum(y_anomaly) * 0.00008
        smooth += torch.sum((y_pred[i, :11] - y_pred[i, 1:12]) ** 2) * 0.00008
    loss = (loss + sparsity + smooth) / batch_size

    return loss
