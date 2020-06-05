import torch


class Stage2(torch.nn.Module):
    def __init__(self):
        super(Stage2, self).__init__()
        self.layer1 = torch.nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer2 = torch.nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer3 = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer4 = torch.nn.ReLU(inplace=True)
        self.layer5 = torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer6 = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer7 = torch.nn.ReLU(inplace=True)
        self.layer8 = torch.nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer9 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    

    def forward(self, input1):
        out0 = input1.clone()
        out1 = self.layer1(out0)
        out2 = self.layer2(out0)
        out3 = self.layer3(out1)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        out8 = self.layer8(out7)
        out9 = self.layer9(out2)
        return (out8, out9)
