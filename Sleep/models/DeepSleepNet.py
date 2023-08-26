import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import pywt
class BiLSTM(nn.Module):
    def __init__(self,input_size,hidden_size,num_layer):
        super(BiLSTM, self).__init__()
        self.hidden_size=hidden_size
        self.num_layer=num_layer
        self.lstm=nn.LSTM(input_size,hidden_size,num_layer,batch_first=True,dropout=0.5,bidirectional=True)
    def forward(self,x):

        h0=torch.zeros(self.num_layer*2,x.size(0),self.hidden_size).cuda()
        c0=torch.zeros(self.num_layer*2,x.size(0),self.hidden_size).cuda()

        out,_=self.lstm(x,(h0,c0))
        return out


class WaveletTransLayer(nn.Module):
    """
    小波变换层
    """
    def __init__(self):
        super(WaveletTransLayer,self).__init__()

    def forward(self,x):
        # print(x)
        output=self.waveletfunction(x)
        return output

    def waveletfunction(self,input):
        wavename = 'db5'
        # print(input)
        input1=input.flatten(1,2).cpu().numpy()
        cA, cD = pywt.dwt(input1, wavename)
        ya = pywt.idwt(cA, None, wavename, 'smooth')  # approximated component dipin
        yd = pywt.idwt(None, cD, wavename, 'smooth')  # detailed component gaopin

        ya=torch.from_numpy(ya).cuda().unsqueeze(1)
        return ya
        # return outputs

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        # (N,in_channel,x)->(N,out_channels,x_)
        # 卷积核大小为kernel_size*in_channels
        self.feature1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=50,stride=6),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Dropout(0.5),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8,stride=1,padding="same"),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8,padding="same"),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8,padding="same"),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),

        )
        self.feature2 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=400, stride=50),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(0.5),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=6,padding="same"),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=6,padding="same"),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=6,padding="same"),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.features_seq=nn.Sequential(
            BiLSTM(128*21,512,2)
        )
        self.res=nn.Linear(128*21,1024)
        self.reclassify=nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024,5)
        )
        self.prelayer=nn.Sequential(
            WaveletTransLayer()
        )
    def forward(self,x):
        # x=self.prelayer(x)
        # x.unsqueeze(1)
        x1=self.feature1(x) #[bs,128,15]
        x2=self.feature2(x) #[bs,128,6]
        out1=torch.cat((x1,x2),dim=2) # weidu [bs,128,61]
        out1=out1.flatten(1,2) # shape [bs,128*61=7808]
        x_seq=out1.unsqueeze(1) #shape [bs,1,7808]
        x_blstm=self.features_seq(x_seq)
        x_blstm=torch.squeeze(x_blstm,1)
        x_res=self.res(out1)
        x=torch.mul(x_res,x_blstm)
        y=self.reclassify(x)

        return y


if __name__ == "__main__":
    input = np.ones((10, 1, 3000))
    # input = torch.tensor(input, dtype=torch.float32)
    # # (batch,1,3000)->(batch,64,53)
    # conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=400, stride=50)
    # maxpool1=nn.MaxPool1d(kernel_size=8, stride=8)
    # conv2=nn.Conv1d(in_channels=64, out_channels=128, kernel_size=6,stride=1,padding="same")
    # output = conv1(input)
    # output1 = maxpool1(output)
    # output2=conv2(output1)
    model=DeepSleepNet()
    model=model.cuda()

    input=torch.tensor(input,dtype=torch.float32).cuda()
    output=model(input)

    # input.flatten(1,2)
    print(output)
    # print(model(input))
    # writer =SummaryWriter("../log")
    # writer.add_graph(model,x_3)
    # writer.close()
