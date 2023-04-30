import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class GlossModel(nn.Module):
    def __init__(self, input_size: int, class_no: int):
        super().__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.lstm1 = nn.LSTM(
            input_size, 128, device=self.device, batch_first=True)
        self.dropout1 = nn.Dropout(p=0.15)
        self.lstm2 = nn.LSTM(128, 64, device=self.device, batch_first=True)
        self.dropout2 = nn.Dropout(p=0.15)
        self.fc1 = nn.Linear(64, 32, device=self.device)
        self.dropout3 = nn.Dropout(p=0.15)
        self.fc2 = nn.Linear(32, class_no, device=self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm1_out = self.dropout1(self.sigmoid(self.lstm1(x)[0]))
        lstm2_out = self.dropout2(self.sigmoid(self.lstm2(lstm1_out)[0]))
        fc1_out = self.dropout3(self.sigmoid(self.fc1(lstm2_out)))
        out = self.softmax(self.fc2(fc1_out))
        return out
