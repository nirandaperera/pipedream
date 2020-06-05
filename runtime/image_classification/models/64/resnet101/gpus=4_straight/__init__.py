from .resnet101 import resnet101
from .stage0 import Stage0
from .stage1 import Stage1
from .stage2 import Stage2
from .stage3 import Stage3

def arch():
    return "resnet101"

def model(criterion):
    return [
        (Stage0(), ["input0"], ["out0"]),
        (Stage1(), ["out0"], ["out2"]),
        (Stage2(), ["out2"], ["out3", "out4"]),
        (Stage3(), ["out3", "out4"], ["out5"]),
        (criterion, ["out5"], ["loss"])
    ]

def full_model():
    return resnet101()
