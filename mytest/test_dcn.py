from mmcv.ops import DeformConv2d
from mmcv.cnn import ConvModule
import torch.nn as nn
import torch

cls_convs = nn.ModuleList()
cls_convs.append(
                    ConvModule(
                        3,
                        64,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=None,
                        norm_cfg=None))
cls_convs.append(
                        DeformConv2d(
                            64,
                            64,
                            3,
                            stride=1,
                            padding=1,
                            groups=1,
                            bias=False))

input = torch.rand(1, 3, 224, 224)

out1 = cls_convs[0](input) # [1, 64, 224, 224]

init_t = torch.Tensor(out1.size(0), 1, out1.size(-2), out1.size(-1))  # [1, 1, 224, 224]
item = torch.ones_like(init_t)*3  #  [1, 1, 224, 224] 全3
zeros = torch.zeros_like(item)    #  [1, 1, 224, 224] 全0
sampling_loc = torch.cat((-item,-item,-item,zeros,-item,item,zeros,-item, zeros, zeros, 
                                      zeros,item,item,-item,item,zeros,item,item), dim=1) # [1, 18, 224. 224] 

out2 = cls_convs[1](out1, sampling_loc)

print(out2.shape)





