from typing import Tuple, List, Any, Union

import escnn.nn as enn
from escnn import gspaces
from escnn.nn import init,GeometricTensor,FieldType,EquivariantModule
from escnn.gspaces import *

import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

#rom argparse import ArgumentParser

def conv7x7(in_type: enn.FieldType, out_type: enn.FieldType, stride=1, padding=3,
            dilation=1, bias=False):
    """7x7 convolution with padding"""
    return enn.R2Conv(in_type, out_type, 7,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=bias,
                      sigma=None,
                      frequencies_cutoff=lambda r: 3*r,
                      )

def conv3x3(in_type: enn.FieldType, out_type: enn.FieldType, stride=1, padding=1,
            dilation=1, bias=False):
    """3x3 convolution with padding"""
    return enn.R2Conv(in_type, out_type, 3,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=bias,
                      sigma=None,
                      frequencies_cutoff=lambda r: 3*r,
                      )

def conv1x1(in_type: enn.FieldType, out_type: enn.FieldType, stride=1, padding=0,
            dilation=1, bias=False):
    """1x1 convolution with padding"""
    return enn.R2Conv(in_type, out_type, 1,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=bias,
                      sigma=None,
                      frequencies_cutoff=lambda r: 3*r,
                      )

def regular_feature_type(gspace: gspaces.GSpace, planes: int, fixparams: bool = True):
    """ build a regular feature map with the specified number of channels"""
    assert gspace.fibergroup.order() > 0

    N = gspace.fibergroup.order()

    if fixparams:
        planes *= math.sqrt(N)

    planes = planes / N
    planes = int(planes)

    return enn.FieldType(gspace, [gspace.regular_repr] * planes)


def trivial_feature_type(gspace: gspaces.GSpace, planes: int, fixparams: bool = True):
    """ build a trivial feature map with the specified number of channels"""

    if fixparams:
        planes *= math.sqrt(gspace.fibergroup.order())

    planes = int(planes)
    return enn.FieldType(gspace, [gspace.trivial_repr] * planes)



FIELD_TYPE = {
    "trivial": trivial_feature_type,
    "regular": regular_feature_type,
}       

class BasicBlock(enn.EquivariantModule):

    def __init__(self,
                 in_type: enn.FieldType,
                 inner_type: enn.FieldType,
                 dropout_rate: float,
                 stride: int = 1,
                 out_type: enn.FieldType = None,
                 ):
        super(BasicBlock, self).__init__()

        if out_type is None:
            out_type = in_type

        self.in_type = in_type
        inner_type = inner_type
        self.out_type = out_type

        conv = conv3x3

        self.bn1 = enn.InnerBatchNorm(self.in_type)
        self.relu1 = enn.ReLU(self.in_type, inplace=True)
        self.conv1 = conv(self.in_type, inner_type)

        self.bn2 = enn.InnerBatchNorm(inner_type)
        self.relu2 = enn.ReLU(inner_type, inplace=True)

        self.dropout = enn.PointwiseDropout(inner_type, p=dropout_rate)

        self.conv2 = conv(inner_type, self.out_type, stride=stride)

        self.shortcut = None
        if stride != 1 or self.in_type != self.out_type:
            self.shortcut = conv1x1(self.in_type, self.out_type, stride=stride, bias=False)

    def forward(self, x):
        x_n = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(x_n)))
        out = self.dropout(out)
        out = self.conv2(out)

        if self.shortcut is not None:
            out += self.shortcut(x_n)
        else:
            out += x

        return out

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        if self.shortcut is not None:
            return self.shortcut.evaluate_output_shape(input_shape)
        else:
            return input_shape
        
class Bottleneck(enn.EquivariantModule): #MIGHT WORK?
    expansion: int = 4

    def __init__(
        self,
        in_type: enn.FieldType,
        inner_type: enn.FieldType,
        dropout_rate: float,
        stride: int = 1,
        out_type: enn.FieldType = None,
        )-> None:
        super().__init__()

        if out_type is None:
            out_type = in_type

        self.in_type = in_type
        inner_type = inner_type
        self.out_type = out_type
        
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.bn1 = enn.InnerBatchNorm(self.in_type)
        self.relu1 = enn.ReLU(self.in_type,inplace=True)
        self.conv1 = conv1x1(self.in_type, inner_type)

        self.bn2 = enn.InnerBatchNorm(inner_type)
        self.relu2 = enn.ReLU(inner_type,inplace=True)
        self.conv2 = conv3x3(inner_type, inner_type, stride=stride) #I HAVE NO IDEA WHAT SHOULD BE IN AND INNER?

        #self.conv3 = conv1x1(inner_type, self.out_type)
        self.bn3 = enn.InnerBatchNorm(inner_type)
        self.relu3 = enn.ReLU(inner_type,inplace=True)
        self.conv3 = conv1x1(inner_type, self.out_type)

        self.stride = stride
        self.shortcut = None
        if stride != 1 or self.in_type != self.out_type:
            #self.shortcut = conv1x1(self.in_type, self.out_type, stride=stride)
            self.shortcut = conv1x1(self.in_type, self.out_type, stride=stride)

    def forward(self, x):
        out = self.bn1(x)
        x_n = self.relu1(out)
        out = self.conv1(x_n)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
    
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv3(out)

        if self.shortcut is not None:
            out += self.shortcut(x_n)
        else:
            out += x    

        return out
    
    def evaluate_output_shape(self, input_shape: Tuple): #NEEDED AS THIS IS ABSTRACT IN ENN MODULE?
        assert input_shape[1] == self.in_type.size
        if self.shortcut is not None:
            return self.shortcut.evaluate_output_shape(input_shape)
        else:
            return input_shape


class G_ResNet(torch.nn.Module):
    def __init__(self,
                 block: BasicBlock,
                 layers,
                 dropout_rate: float=0, 
                 num_classes:int =3,
                 num_channels: int=3,
                 N: int = 8,
                 r: int = 0,
                 f: bool = False,
                 deltaorth: bool = False,
                 fixparams: bool = True,
                 initial_stride: int = 1,
                 NOISY: bool = False,
                 custom_predict: bool = False, #Use Jia et al predict function
                 ):
        r"""
        
        Build and equivariant ResNet-18.
        
        The parameter ``N`` controls rotation equivariance and the parameter ``f`` reflection equivariance.
        
        More precisely, ``N`` is the number of discrete rotations the model is initially equivariant to.
        ``N = 1`` means the model is only reflection equivariant from the beginning.
        
        ``f`` is a boolean flag specifying whether the model should be reflection equivariant or not.
        If it is ``False``, the model is not reflection equivariant.
        
        ``r`` is the restriction level:
        
        - ``0``: no restriction. The model is equivariant to ``N`` rotations from the input to the output
        - ``1``: restriction before the last block. The model is equivariant to ``N`` rotations before the last block
               (i.e. in the first 2 blocks). Then it is restricted to ``N/2`` rotations until the output.
        
        - ``2``: restriction after the first block. The model is equivariant to ``N`` rotations in the first block.
               Then it is restricted to ``N/2`` rotations until the output (i.e. in the last 3 blocks).
               
        - ``3``: restriction after the first and the second block. The model is equivariant to ``N`` rotations in the first
               block. It is restricted to ``N/2`` rotations before the second block and to ``1`` rotations before the last
               block.
        
        NOTICE: if restriction to ``N/2`` is performed, ``N`` needs to be even!
        
        """
        super(G_ResNet, self).__init__()

        self.custom_predict = custom_predict
        if custom_predict:
            num_classes = 2

        nStages = [64, 64, 128, 256, 512]
        #nStages = [16,16,32,64,128]

        self._fixparams = fixparams

        self._layer = 0

        # number of discrete rotations to be equivariant to
        self._N = N

        # if the model is [F]lip equivariant
        self._f = f
        if self._f:
            if N != 1:
                self.gspace = gspaces.flipRot2dOnR2(N) # change to escnn
            else:
                self.gspace = gspaces.flip2dOnR2()
        else:
            if N != 1:
                self.gspace = gspaces.rot2dOnR2(N)
            else:
                self.gspace = gspaces.trivialOnR2()

        # level of [R]estriction:
        #   r = 0: never do restriction, i.e. initial group (either DN or CN) preserved for the whole network
        #   r = 1: restrict before the last block, i.e. initial group (either DN or CN) preserved for the first
        #          2 blocks, then restrict to N/2 rotations (either D{N/2} or C{N/2}) in the last block
        #   r = 2: restrict after the first block, i.e. initial group (either DN or CN) preserved for the first
        #          block, then restrict to N/2 rotations (either D{N/2} or C{N/2}) in the last 2 blocks
        #   r = 3: restrict after each block. Initial group (either DN or CN) preserved for the first
        #          block, then restrict to N/2 rotations (either D{N/2} or C{N/2}) in the second block and to 1 rotation
        #          in the last one (D1 or C1)
        assert r in [0, 1, 2, 3]
        self._r = r

        # the input has 3 color channels (RGB).
        # Color channels are trivial fields and don't transform when the input is rotated or flipped
        r1 = enn.FieldType(self.gspace, [self.gspace.trivial_repr] * num_channels)

        # input field type of the model
        self.in_type = r1

        # in the first layer we always scale up the output channels to allow for enough independent filters
        r2 = FIELD_TYPE["regular"](self.gspace, nStages[0], fixparams=self._fixparams)

        # dummy attribute keeping track of the output field type of the last submodule built, i.e. the input field type of
        # the next submodule to build
        self._in_type = r2

        self.conv1 = conv7x7(r1, r2, stride=2)
        self.layer1 = self.basicLayer(block, nStages[1], layers[0], dropout_rate, stride=1, NOISY=NOISY)
        self.layer2 = self.basicLayer(block, nStages[2], layers[1], dropout_rate, stride=2, NOISY=NOISY)
        self.layer3 = self.basicLayer(block, nStages[3], layers[2], dropout_rate, stride=2, NOISY=NOISY)
        # last layer maps to a trivial (invariant) feature map
        self.layer4 = self.basicLayer(block, nStages[4], layers[3], dropout_rate, stride=2, totrivial=True, NOISY=NOISY)

        self.bn = enn.InnerBatchNorm(self.layer4.out_type, momentum=0.9)
        self.relu = enn.ReLU(self.bn.out_type, inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.gpool = enn.GroupPooling(self.bn.out_type)
        self.linear = torch.nn.Linear(self.gpool.out_type.size, num_classes)

        for name, module in self.named_modules():
            if isinstance(module, enn.R2Conv):
                if deltaorth:
                    init.deltaorthonormal_init(module.weights, module.basisexpansion)
            elif isinstance(module, torch.nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, torch.nn.Linear):
                module.bias.data.zero_()
        if NOISY:
            print("MODEL TOPOLOGY:")
            for i, (name, mod) in enumerate(self.named_modules()):
                print(f"\t{i} - {name}")

    def basicLayer(self, block, planes: int, num_blocks: int, dropout_rate: float, stride: int, NOISY: bool,
                    totrivial: bool = False
                    ) -> enn.SequentialModule:

        self._layer += 1
        if NOISY:
            print("start building", self._layer)
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        main_type = FIELD_TYPE["regular"](self.gspace, planes, fixparams=self._fixparams)
        inner_type = FIELD_TYPE["regular"](self.gspace, planes, fixparams=self._fixparams)

        if totrivial:
            out_type = FIELD_TYPE["trivial"](self.gspace, planes, fixparams=self._fixparams)
        else:
            out_type = FIELD_TYPE["regular"](self.gspace, planes, fixparams=self._fixparams)

        for b, stride in enumerate(strides):
            if b == num_blocks - 1:
                out_f = out_type
            else:
                out_f = main_type
            layers.append(block(self._in_type, inner_type, dropout_rate, stride, out_type=out_f))
            self._in_type = out_f
        if NOISY:
            print("layer", self._layer, "built")
        return enn.SequentialModule(*layers)

    def features(self, x):

        x = enn.GeometricTensor(x, self.in_type)

        out = self.conv1(x)

        x1 = self.layer1(out)

        x2 = self.layer2(x1)

        x3 = self.layer3(x2)

        x4 = self.layer4(x3)

        return x1, x2, x3, x4

    def forward(self, x):

        # wrap the input tensor in a GeometricTensor
        x = enn.GeometricTensor(x, self.in_type)
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.gpool(out)

        # extract the tensor from the GeometricTensor to use the common Pytorch operations
        out = out.tensor
        gpool_out = out

        b, c, w, h = out.shape
        out = F.avg_pool2d(out, (w, h))

        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out #, gpool_out

    def state_dict(self,*args, **kwargs):
        #Issue with saving state dictionary if eval is not called
        in_train_mode = self.training
        self.eval()
        state_dictionary = super().state_dict(*args, **kwargs)

        if in_train_mode: #If model was previously in train mode, return to train mode
            self.train()
        return state_dictionary


    def predict(self, x): #Override predict
        if self.custom_predict:
            x_i = torch.flip(x, (-1,))
            a = super().__call__(x)
            a_i = super().__call__(x_i)
            return torch.cat((a[..., 0:1], a_i[..., 0:1], 0.5 * (a[..., 1:2] + a_i[..., 1:2])), dim=-1)
        else:
            a = super().__call__(x)
            return a

    def __call__(self, *args, **kwargs):
         return self.predict(*args, **kwargs)
    
def _resnet(
    block,
    layers: List[int],
    weights: None,
    progress: bool,
    **kwargs: Any,
) -> G_ResNet:

    model = G_ResNet(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model


def G_ResNet50(*, weights= None, progress: bool = True, **kwargs: Any) -> G_ResNet:
 
    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)


def G_ResNet18(*, weights = None, progress: bool = True, **kwargs: Any) -> G_ResNet:

    return _resnet(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)