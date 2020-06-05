import torch


class Stage1(torch.nn.Module):
    def __init__(self):
        super(Stage1, self).__init__()
        self.layer3 = torch.nn.ReLU(inplace=True)
        self.layer4 = torch.nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer5 = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer6 = torch.nn.ReLU(inplace=True)
        self.layer7 = torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer8 = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer9 = torch.nn.ReLU(inplace=True)
        self.layer10 = torch.nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer11 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer13 = torch.nn.ReLU(inplace=True)
        self.layer14 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer15 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer16 = torch.nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer17 = torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer18 = torch.nn.ReLU(inplace=True)
        self.layer19 = torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer20 = torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer21 = torch.nn.ReLU(inplace=True)
        self.layer22 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer23 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer25 = torch.nn.ReLU(inplace=True)
        self.layer26 = torch.nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer27 = torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer28 = torch.nn.ReLU(inplace=True)
        self.layer29 = torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer30 = torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer31 = torch.nn.ReLU(inplace=True)
        self.layer32 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer33 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer35 = torch.nn.ReLU(inplace=True)
        self.layer36 = torch.nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer37 = torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer38 = torch.nn.ReLU(inplace=True)
        self.layer39 = torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer40 = torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer41 = torch.nn.ReLU(inplace=True)
        self.layer42 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer43 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer45 = torch.nn.ReLU(inplace=True)
        self.layer46 = torch.nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer47 = torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer48 = torch.nn.ReLU(inplace=True)
        self.layer49 = torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer50 = torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer51 = torch.nn.ReLU(inplace=True)
        self.layer52 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer53 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer55 = torch.nn.ReLU(inplace=True)
        self.layer56 = torch.nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer57 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer58 = torch.nn.ReLU(inplace=True)
        self.layer59 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer60 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer61 = torch.nn.ReLU(inplace=True)
        self.layer62 = torch.nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer63 = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer64 = torch.nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer65 = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer67 = torch.nn.ReLU(inplace=True)
        self.layer68 = torch.nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer69 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer70 = torch.nn.ReLU(inplace=True)
        self.layer71 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer72 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer73 = torch.nn.ReLU(inplace=True)
        self.layer74 = torch.nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)

    

    def forward(self, input1, input0):
        out0 = input0.clone()
        out1 = input1.clone()
        out0 = out0 + out1
        out3 = self.layer3(out0)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        out8 = self.layer8(out7)
        out9 = self.layer9(out8)
        out10 = self.layer10(out9)
        out11 = self.layer11(out10)
        out11 = out11 + out3
        out13 = self.layer13(out11)
        out14 = self.layer14(out13)
        out15 = self.layer15(out14)
        out16 = self.layer16(out13)
        out17 = self.layer17(out16)
        out18 = self.layer18(out17)
        out19 = self.layer19(out18)
        out20 = self.layer20(out19)
        out21 = self.layer21(out20)
        out22 = self.layer22(out21)
        out23 = self.layer23(out22)
        out23 = out23 + out15
        out25 = self.layer25(out23)
        out26 = self.layer26(out25)
        out27 = self.layer27(out26)
        out28 = self.layer28(out27)
        out29 = self.layer29(out28)
        out30 = self.layer30(out29)
        out31 = self.layer31(out30)
        out32 = self.layer32(out31)
        out33 = self.layer33(out32)
        out33 = out33 + out25
        out35 = self.layer35(out33)
        out36 = self.layer36(out35)
        out37 = self.layer37(out36)
        out38 = self.layer38(out37)
        out39 = self.layer39(out38)
        out40 = self.layer40(out39)
        out41 = self.layer41(out40)
        out42 = self.layer42(out41)
        out43 = self.layer43(out42)
        out43 = out43 + out35
        out45 = self.layer45(out43)
        out46 = self.layer46(out45)
        out47 = self.layer47(out46)
        out48 = self.layer48(out47)
        out49 = self.layer49(out48)
        out50 = self.layer50(out49)
        out51 = self.layer51(out50)
        out52 = self.layer52(out51)
        out53 = self.layer53(out52)
        out53 = out53 + out45
        out55 = self.layer55(out53)
        out56 = self.layer56(out55)
        out57 = self.layer57(out56)
        out58 = self.layer58(out57)
        out59 = self.layer59(out58)
        out60 = self.layer60(out59)
        out61 = self.layer61(out60)
        out62 = self.layer62(out61)
        out63 = self.layer63(out62)
        out64 = self.layer64(out55)
        out65 = self.layer65(out64)
        out63 = out63 + out65
        out67 = self.layer67(out63)
        out68 = self.layer68(out67)
        out69 = self.layer69(out68)
        out70 = self.layer70(out69)
        out71 = self.layer71(out70)
        out72 = self.layer72(out71)
        out73 = self.layer73(out72)
        out74 = self.layer74(out73)
        return (out67, out74)
