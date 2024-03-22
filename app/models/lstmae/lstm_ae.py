import torch
import torch.nn as nn

'''
LSTM output
- N : number of batches
- L : sequence lengh
- Q : input dim
- K : number of layers
- D : LSTM feature dimension

Y,(hn,cn) = LSTM(X)

- X : [N x L x Q] - `N` input sequnce of length `L` with `Q` dim.
- Y : [N x L x D] - `N` output sequnce of length `L` with `D` feature dim.
- hn : [K x N x D] - `K` (per each layer) of `N` final hidden state with  `D` feature dim.
- cn : [K x N x D] - `K` (per each layer) of `N` final hidden state with  `D` cell dim.
'''

class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = (
            embedding_dim, 2 * embedding_dim
        )
        
        self.rnn1 = nn.LSTM(
          input_size=n_features,
          hidden_size=self.hidden_dim,
          num_layers=1,
          batch_first=True
        )
        
        self.rnn2 = nn.LSTM(
          input_size=self.hidden_dim,
          hidden_size=embedding_dim,
          num_layers=1,
          batch_first=True
        )
        
    def forward(self, x):
        x, (hidden_n, cell_n) = self.rnn1(x) # x: [batch_size, seq_len, hidden_dim]
        x, (hidden_n, cell_n) = self.rnn2(x) # x: [batch_size, seq_len, embedding_dim]
        # decoder 에 가는 것은 마지막 sequence => [1, 1, 64]
        return  x[:,-1,:]
    

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        # Squash batch_size and seq_len into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (batch_size * seq_len, input_size) => (20, 128)
        y = self.module(x_reshape) # linear layer, output: n_features(38)
        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (batch_size, seq_len, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (seq_len, batch_size, output_size)
        return y
    
    
class Decoder(nn.Module):
    def __init__(self, prediction_time=1, input_dim=64, n_features=1):
        super(Decoder, self).__init__()
        # input 은 encoder 에서 나온 embedding_dim
        self.seq_len, self.input_dim = prediction_time, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features
        
        self.rnn1 = nn.LSTM(
          input_size=input_dim,
          hidden_size=input_dim,
          num_layers=1,
          batch_first=True
        )
        
        self.rnn2 = nn.LSTM(
          input_size=input_dim,
          hidden_size=self.hidden_dim,
          num_layers=1,
          batch_first=True
        )
        
        # time_distributed
        # linear 로 1차원으로 복원 후 각 time 에서 출력된 아웃풋을 linear layer 와 연결
        self.output_layer = torch.nn.Linear(self.hidden_dim, n_features)
        self.timedist = TimeDistributed(self.output_layer)
        
    def forward(self, x):
        x=x.reshape(-1,1,self.input_dim).repeat(1,self.seq_len,1) # [batch * last seq_len, 1 * seq_len, embedding_dim] => [1, 20, 64]
        x, (hidden_n, cell_n) = self.rnn1(x) # input [1,20,64]
        x, (hidden_n, cell_n) = self.rnn2(x) # input [1,20,128]
        return self.timedist(x)
    
    
class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, prediction_time, n_features, embedding_dim=64):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = Encoder(seq_len, n_features, embedding_dim)
        self.decoder = Decoder(prediction_time, embedding_dim, n_features)
    def forward(self, x):
        # x: [1, seq_len, n_features]
        x_e = self.encoder(x) # x_e: [batch_size, last seq_len, embedding_dim]
        x_d = self.decoder(x_e) 
        # final output(x_d): [1, seq_len, n_features]
        return x_d