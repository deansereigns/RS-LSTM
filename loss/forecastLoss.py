import torch.nn as nn

def dataLoss(yTrue,yPred):
    mseL = nn.MSELoss()
    return mseL(yTrue,yPred)
