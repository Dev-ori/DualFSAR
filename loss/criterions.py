from torch import nn
from .build import loss_register

@loss_register("cross entropy")
def CrossEntropyLoss():
    return nn.CrossEntropyLoss()


@loss_register("l1")
def L1Loss():
    return nn.L1Loss()


@loss_register("smooth l1")
def SmoothL1Loss():
    return nn.SmoothL1Loss()


@loss_register("bce with logit")
def BCEWithLogitsLoss():
    return nn.BCEWithLogitsLoss()


