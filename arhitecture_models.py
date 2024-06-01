import torch
import torch.nn as nn
import torchvision.models as models

class CNN_Encoder(nn.Module):
    def __init__(self,cnn_model,cnn_output_size, embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.cnn = cnn_model
        self.embedding_dim = embedding_dim
        self.cnn_output_size = cnn_output_size
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.cnn_output_size ,self.embedding_dim)
    def forward(self, x):
        features= self.cnn(x)
        features = self.fc(features)
        return features
    

class LSTM_Deconder(nn.Module):
    def __init__(self, vocab_dim, embedding_dim, hidden_dim, n_layers, drop_prob=0.):
        super(LSTM_Deconder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_dim)
        self.dropout = nn.Dropout(drop_prob)
        self.prelu = nn.PReLU()
    def forward(self, x, h, c=None):
        emb = self.embedding(x)
        out, (h, c) = self.lstm(emb, (h, c))
        out =self.fc1(out)
        out = self.prelu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out, h, c
    '''-
    def init_hidden(self, batch_size):
        " Initialize the hidden state of the RNN to zeros"
        weight = next(self.parameters()).data
        if self.rnn_cell == 'LSTM': # in LSTM we have a cell state and a hidden state
            return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(), weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
        else:                       # in GRU and RNN we only have a hidden state
            return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(), None
    '''


class CNNLSTMModel(nn.Module):
    def __init__(self, cnn_model, lstm_decoder):
        super(CNNLSTMModel, self).__init__()
        self.cnn = cnn_model
        self.lstm_decoder = lstm_decoder
        
    def forward(self, x):
        features = self.cnn(x[0])
        decoder_features=torch.reshape(features,(1,features.shape[0],-1))
        out = self.lstm_decoder(x[1],decoder_features,decoder_features)
        
        return out
