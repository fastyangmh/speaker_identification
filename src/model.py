# import
import torch
import torch.nn as nn

# class


class DCNN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, n_hidden, dropout, out_channels=13, kernel_size=3, padding=0, stride=1):
        super(DCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=39, out_channels=out_channels,
                               kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool1 = nn.MaxPool1d(
            int((in_dim - kernel_size+2*padding) / stride+1))
        in_sizes = [out_channels]+[hidden_dim]*(n_hidden-1)
        out_sizes = [hidden_dim]*n_hidden
        self.layers = nn.ModuleList([nn.Linear(in_size, out_size) for (
            in_size, out_size) in zip(in_sizes, out_sizes)])
        self.last_layer = nn.Linear(hidden_dim, out_dim)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool1(self.conv1(x)).squeeze(2)
        for layer in self.layers:
            x = self.dropout(self.leakyrelu(layer(x)))
        return self.softmax(self.last_layer(x))
