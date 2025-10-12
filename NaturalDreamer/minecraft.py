import torch
from torch import nn



class MinecraftSegmentationHead(nn.Module):

    def __init__(self, inputSize, config):
        super().__init__()
        self.config = config
        self.pow = config

    #Input:

    # Output:
    # B - batch size
    # X, Y, Z - grid dimensions
    # B_c - number of block clases
    # number of block states (
    #predicrs grid B,X,Y,Z,B_c,16
    pass

class TestBlockPositionDecoding:

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fov = config.minecraft.config
        H, W = 64,64 #TODO get from config

    def compute(self, info):
        input = info["depth"]
        XPos, YPos, ZPos, Yaw, Pitch = info["xpos"], info["ypos"], info["zpos"], info["yaw"], info["pitch"]
        YPos += 1.62
        assert input.shape == (self.H, self.W, 1)

