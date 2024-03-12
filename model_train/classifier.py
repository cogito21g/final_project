import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMAutoencoder(nn.Module):
    def __init__(self, sequence_length, n_features, prediction_time):
        super().__init__()

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
        self.decoder = nn.LSTM(input_size=50 + prediction_time - 1, hidden_size=100, batch_first=True)
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

        if self.prediction_time == 1:
            return x[:, -1, :].unsqueeze(dim=1)
        else:
            return x[:, -(self.prediction_time) :, :]


class ClassifierVMAE(nn.Module):
    def __init__(self, input_dim=710, drop_p=0.0):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.drop_p = drop_p
        self.weight_init()

    def weight_init(self):
        for layer in self.classifier:
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        x = self.classifier(x)

        return x
