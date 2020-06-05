import torch
from .stage0 import Stage0
from .stage1 import Stage1
from .stage2 import Stage2
from .stage3 import Stage3

class resnet101(torch.nn.Module):
    def __init__(self):
        super(resnet101, self).__init__()
        self.stage0 = Stage0()
        self.stage1 = Stage1()
        self.stage2 = Stage2()
        self.stage3 = Stage3()

    

    def forward(self, input0):
        out0 = self.stage0(input0)
        out2 = self.stage1(out0)
        (out3, out4) = self.stage2(out2)
        out5 = self.stage3(out3, out4)
        return out5
