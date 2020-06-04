from .resnet50 import Resnet50Straight
from .stage0 import Stage0
from .stage1 import Stage1
from .stage2 import Stage2
from .stage3 import Stage3

def arch():
    return "resnet50"

def model(criterion):
    return [
        (Stage0(), ["input0"], ["out1", "out0"]),
        (Stage1(), ["out1", "out0"], ["out2"]),
        (Stage2(), ["out2"], ["out4", "out3"]),
        (Stage3(), ["out4", "out3"], ["out5"]),
        (criterion, ["out5"], ["loss"])
    ]

def full_model():
    return Resnet50Straight()
