import torch
from torch import Tensor
import numpy as np
from typing import Type, Any, Callable, Union, List, Optional, Tuple
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.resnet import BasicBlock, Bottleneck

class CE_Resnet(models.resnet.ResNet):
    def __init__(self,
        block: Type[Union[BasicBlock, Bottleneck]],
        use_max_pool: bool = True,
        use_avg_pool: bool = True,
        num_classes: int = 2,
        avg_pool_size: Tuple[int] = (1, 1),
        add_fc: Optional[List[int]] = None, *args, **kwargs):
        
        super().__init__(block, *args, **kwargs)

        self.avgpool = nn.AdaptiveAvgPool2d(avg_pool_size)
        pool_expansion = 1
        if not use_avg_pool:
            pool_expansion = 16 if use_max_pool else 64
        else:
            pool_expansion = np.prod(avg_pool_size) 

        self.fc = self._make_fc(512 * block.expansion * pool_expansion, num_classes, add_fc)

        self.use_max_pool = use_max_pool
        self.use_avg_pool = use_avg_pool

    def _make_fc(self, in_features: int, out_features: int, add_fc: Optional[List[int]]):
        if add_fc is None:
            return nn.Linear(in_features, out_features)
        else:
            add_fc.insert(0, in_features)
            add_fc.append(out_features)
            fc_layers = []
            for i in range(len(add_fc) - 1):
                fc_layers.append(nn.Linear(add_fc[i], add_fc[i + 1]))
                if i != len(add_fc) - 2:
                    fc_layers.append(nn.Tanh())
            return nn.Sequential(*fc_layers)
        
    def _forward_impl(self, x: Tensor) -> Tensor: #Override forward
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.use_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.use_avg_pool:
            x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
 
    def predict(self, x: Tensor) -> Tensor: #Override predict
        x_i = torch.flip(x, (-1,))
        a = super().__call__(x)
        a_i = super().__call__(x_i)
        return torch.cat((a[..., 0:1], a_i[..., 0:1], 0.5 * (a[..., 1:2] + a_i[..., 1:2])), dim=-1)

    def __call__(self, *args, **kwargs):
         return self.predict(*args, **kwargs)


def CE_Resnet50(**kwargs: Any) -> CE_Resnet:
    model = CE_Resnet(block=Bottleneck, layers=[3, 4, 6, 3], use_max_pool=True,
     use_avg_pool=True, avg_pool_size=(1, 1), add_fc=[512, 512, 64, 64],**kwargs)
    return model

def CEResnet18(**kwargs: Any) -> CE_Resnet:
    model = CE_Resnet(block=BasicBlock, layers=[2, 2, 2, 2], use_max_pool=True,
     use_avg_pool=True, avg_pool_size=(1, 1), add_fc=[512, 512, 64, 64],**kwargs)
    return model