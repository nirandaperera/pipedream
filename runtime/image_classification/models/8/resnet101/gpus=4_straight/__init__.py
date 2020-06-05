from .resnet101 import resnet101
from .stage0 import Stage0
from .stage1 import Stage1
from .stage2 import Stage2
from .stage3 import Stage3

def arch():
    return "resnet101"

def model(criterion):
    return [
        (Stage0(), ["input0"], ["out0", "out1"]),
        (Stage1(), ["out0", "out1"], ["out3"]),
        (Stage2(), ["out3"], ["out4", "out5"]),
        (Stage3(), ["out4", "out5"], ["out6"]),
        (criterion, ["out6"], ["loss"])
    ]

def full_model():
    return resnet101()
