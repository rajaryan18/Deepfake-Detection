import torch
from torch import nn

class Convolution3d(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Convolution3d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=8, kernel_size=3, stride=2),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.BatchNorm3d(num_features=8),
            nn.Conv3d(in_channels=8, out_channels=64, kernel_size=3, stride=2),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.BatchNorm3d(num_features=64),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=2),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.BatchNorm3d(num_features=64),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=2),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.BatchNorm3d(num_features=64),
            nn.Conv3d(in_channels=64, out_channels=8, kernel_size=3, stride=2),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.BatchNorm3d(num_features=8),
            nn.Conv3d(in_channels=8, out_channels=1, kernel_size=3, stride=1),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.BatchNorm3d(num_features=1)
        )

        self.flatten = nn.Flatten()
        self.head = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size*2),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size*2, out_features=hidden_size*2),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size*2, out_features=hidden_size//2),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size//2, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, video: torch.Tensor):
        return self.head(self.flatten(self.conv(video)))


class ResnetLstm(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(ResnetLstm, self).__init__()
        self.flatten = nn.Flatten()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=3, dropout=0.2, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size//2),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=hidden_size//2, out_features=hidden_size//4),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=hidden_size//4, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, sequence: torch.Tensor):
        sequence = self.flatten(sequence)
        result, _ = self.lstm(sequence)
        output = self.head(result)
        return output