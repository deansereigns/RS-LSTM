import torch
import torch.nn as nn

class ForecastLSTM(nn.Module):
    def __init__(self, inputDim=8,hiddenDim=128,outputDim=7,numLayers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=inputDim,hidden_size=hiddenDim,num_layers=numLayers, batch_first=True)
        self.fc = nn.Linear(hiddenDim,outputDim)

    def forward(self,x):
        """
        Args:
            x: (batch,seq_len,inputDim)
        Returns:
            preds: (batch,seq_len,outputDim)
        """

        out,_ = self.lstm(x)
        preds = self.fc(out)
        return preds
    
class ForecastFCN(nn.Module):
    def __init__(self, inputDim=2, hiddenDims=(128, 256, 128), outputDim=7):
        super().__init__()
        self.fc1 = nn.Linear(inputDim, hiddenDims[0])
        self.tanh1 = nn.Tanh()
        self.fc2 = nn.Linear(hiddenDims[0], hiddenDims[1])
        self.tanh2 = nn.Tanh()
        self.fc3 = nn.Linear(hiddenDims[1], hiddenDims[2])
        self.tanh3 = nn.Tanh()
        self.fc4 = nn.Linear(hiddenDims[2], outputDim)

    def forward(self, x):
        """
        Args:
            x: tensor of shape (batch_size, inputDim)
        Returns:
            preds: tensor of shape (batch_size, outputDim)
        """
        out = self.fc1(x)
        out = self.tanh1(out)
        out = self.fc2(out)
        out = self.tanh2(out)
        out = self.fc3(out)
        out = self.tanh3(out)
        out = self.fc4(out)
        return out