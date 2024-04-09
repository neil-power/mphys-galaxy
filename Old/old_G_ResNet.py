from typing import Any, Callable, List, Optional, Type, Union,Tuple

import torch
import torch.nn as nn
from torch import Tensor
import torchvision.models.resnet as rn
from escnn import nn as enn
from escnn import gspaces

#Define convolutional layers -----------------------------------------------------------------------------------------------------

def conv3x3(in_planes: enn.FieldType, out_planes: enn.FieldType, stride: int = 1, groups: int = 1, dilation: int = 1) -> enn.R2Conv:
    """3x3 convolution with padding"""
    return enn.R2Conv(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        dilation=dilation,
    )


def conv1x1(in_planes: enn.FieldType, out_planes: enn.FieldType, stride: int = 1) -> enn.R2Conv:
    """1x1 convolution"""
    return enn.R2Conv(in_planes, out_planes, kernel_size=1, stride=stride)

#Define blocks layers -----------------------------------------------------------------------------------------------------

class BasicBlock(enn.EquivariantModule):

    def __init__(self,
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

        # self.conv1 = conv3x3(inplanes, inner_type, stride)
        # self.bn1 = norm_layer(self.in_type)
        # self.relu = nn.ReLU(inplace=True)
        
        # self.conv2 = conv3x3(planes, planes)
        # self.bn2 = norm_layer(planes)
        # self.downsample = downsample
        # self.stride = stride

        self.bn1 = enn.InnerBatchNorm(self.in_type)
        self.relu1 = enn.ReLU(self.in_type, inplace=True)
        self.conv1 = conv3x3(self.in_type, inner_type)

        self.bn2 = enn.InnerBatchNorm(inner_type)
        self.relu2 = enn.ReLU(inner_type, inplace=True)

        #self.dropout = enn.PointwiseDropout(inner_type, p=dropout_rate)

        self.conv2 = conv3x3(inner_type, self.out_type, stride=stride)

        self.shortcut = None
        if stride != 1 or self.in_type != self.out_type:
            self.shortcut = conv1x1(self.in_type, self.out_type, stride=stride)

    def forward(self, x):
        out = self.bn1(x) #ORDER IS DIFFERENT TO DEFAULT
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        #out = self.dropout(out)
        out = self.conv2(out)

        if self.shortcut is not None:
            out += self.shortcut(out)
        else:
            out += x  

        return out
    

    # def forward(self, x: Tensor) -> Tensor:
    #     identity = x

    #     out = self.conv1(x)
    #     out = self.bn1(out)
    #     out = self.relu1(out)

    #     out = self.conv2(out)
    #     out = self.bn2(out)

    #     out += identity
    #     out = self.relu2(out)

    #     return out
    def evaluate_output_shape(self, input_shape: Tuple): #NEEDED AS THIS IS ABSTRACT IN ENN MODULE?
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
        self.conv1 = conv1x1(self.in_type, inner_type)
        self.bn1 = enn.InnerBatchNorm(self.in_type)
        self.relu1 = enn.ReLU(self.in_type,inplace=True)

        self.conv2 = conv3x3(inner_type, inner_type) #I HAVE NO IDEA WHAT SHOULD BE IN AND INNER?
        self.bn2 = enn.InnerBatchNorm(inner_type)
        self.relu2 = enn.ReLU(inner_type,inplace=True)

        #self.conv3 = conv1x1(inner_type, self.out_type)
        self.bn3 = enn.InnerBatchNorm(inner_type)
        self.relu3 = enn.ReLU(inner_type,inplace=True)
        self.conv3 = conv1x1(inner_type, self.out_type)

        self.stride = stride

        self.shortcut = None
        if stride != 1 or self.in_type != self.out_type:
            self.shortcut = conv1x1(self.in_type, self.out_type, stride=stride)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
    
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv3(out)   

        if self.shortcut is not None:
            out += self.shortcut(out)
        else:
            out += x    

        return out
    
    def evaluate_output_shape(self, input_shape: Tuple): #NEEDED AS THIS IS ABSTRACT IN ENN MODULE?
        assert input_shape[1] == self.in_type.size
        if self.shortcut is not None:
            return self.shortcut.evaluate_output_shape(input_shape)
        else:
            return input_shape

class G_ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 3,
        num_channels: int=3,
        groups: int = 1,
        width_per_group: int = 64,
        N: int = 8,
        r: int = 0,
        f: bool = False,
    ) -> None:
        """``N`` is the number of discrete rotations the model is initially equivariant to.
        ``N = 1`` means the model is only reflection equivariant from the beginning.
        
        ``f`` is a boolean flag specifying whether the model should be reflection equivariant (T) or not (F).
        
        ``r`` is the restriction level:
        
        - ``0``: no restriction. The model is equivariant to ``N`` rotations from the input to the output
        - ``1``: restriction before the last block. The model is equivariant to ``N`` rotations before the last block
               (i.e. in the first 2 blocks). Then it is restricted to ``N/2`` rotations until the output.
        
        - ``2``: restriction after the first block. The model is equivariant to ``N`` rotations in the first block.
               Then it is restricted to ``N/2`` rotations until the output (i.e. in the last 3 blocks).
               
        - ``3``: restriction after the first and the second block. The model is equivariant to ``N`` rotations in the first
               block. It is restricted to ``N/2`` rotations before the second block and to ``1`` rotations before the last
               block."""
        super().__init__()

        #SET UP FIELD TYPES
        self._N = N
        self._f = f
        self._r = r
        
        if self._f:
            if N != 1:
                self.gspace = gspaces.flipRot2dOnR2(N)
            else:
                self.gspace = gspaces.flip2dOnR2()
        else:
            if N != 1:
                self.gspace = gspaces.rot2dOnR2(N)
            else:
                self.gspace = gspaces.trivialOnR2()


        in_type = enn.FieldType(self.gspace, [self.gspace.trivial_repr] * num_channels)
        self.in_type = in_type
        #NOT SURE WHY THERE ARE 16 PLANES HERE IN G_RESNET ORIG- MIGHT BE 64?
        out_type = enn.FieldType(self.gspace, [self.gspace.regular_repr] * 64)

        self._in_type = out_type
       
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group

        #NOTES:
        #conv takes an in_type, out_type
        #relu is just out_type
        #pools just take out_type

        self.conv1 = enn.R2Conv(in_type, out_type, kernel_size=7, stride=2, padding=3)
        self.bn1 = enn.InnerBatchNorm(in_type)
        self.relu = enn.ReLU(out_type,inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2) #or stride=1 maybe???
        self.layer2 = self._make_layer(block, 128, layers[1],stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2],stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3],stride=2,totrivial=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.gpool = enn.GroupPooling(self.bn1.out_type) #ADDED
        self.fc = nn.Linear(self.gpool.out_type.size, num_classes) #ADDED

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # if isinstance(module, enn.R2Conv): #IN ORIG
            #     if deltaorth:
            #         init.deltaorthonormal_init(module.weights, module.basisexpansion)
            # elif isinstance(module, torch.nn.BatchNorm2d):
            #     module.weight.data.fill_(1)
            #     module.bias.data.zero_()
            # elif isinstance(module, torch.nn.Linear):
            #     module.bias.data.zero_()

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        totrivial: bool = False,
        dropout_rate: float=0,
    ) -> enn.SequentialModule:
        # if dilate:
        #     self.dilation *= stride
        #     stride = 1
        # self.shortcut = None
        # if stride != 1 or self.in_type != self._out_type:
        #     self.shortcut = conv1x1(self.in_type, self.out_type, stride=stride)



        main_type = enn.FieldType(self.gspace, [self.gspace.regular_repr]*planes)
        inner_type = enn.FieldType(self.gspace, [self.gspace.regular_repr]*planes)

        if totrivial:
            out_type = enn.FieldType(self.gspace, [self.gspace.trivial_repr]*planes)
        else:
            out_type = enn.FieldType(self.gspace, [self.gspace.regular_repr]*planes)

        layers = []
        #self.inplanes = planes * block.expansion
        for b in range(1, blocks):
            if b == blocks - 1: #If last block
                out_f = out_type
            else:
                out_f = main_type
            layers.append(
            block(
                self._in_type, inner_type, stride, dropout_rate,out_type=out_f
            )
        )
            self._in_type = out_f

        return enn.SequentialModule(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = enn.GeometricTensor(x,self.in_type)
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)

        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)

        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.gpool(out)

        out = out.tensor

        b, c, w, h = out.shape #NEW
        out = nn.functional.avg_pool2d(out, (w, h))
        out = out.view(out.size(0), -1)

        x = self.fc(out)

        # if self.shortcut is not None:
        #     out += self.shortcut(x_n)
        # else:
        #     out += x

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    
    def state_dict(self,*args, **kwargs):
        #Issue with saving state dictionary if eval is not called
        in_train_mode = self.training
        self.eval()
        state_dictionary = super().state_dict(*args, **kwargs)

        if in_train_mode: #If model was previously in train mode, return to train mode
            self.train()
        return state_dictionary


def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: None,
    progress: bool,
    **kwargs: Any,
) -> G_ResNet:

    model = G_ResNet(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model


def G_ResNet50(*, weights= None, progress: bool = True, **kwargs: Any) -> rn.ResNet:
 
    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)


def G_ResNet18(*, weights = None, progress: bool = True, **kwargs: Any) -> rn.ResNet:

    return _resnet(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)