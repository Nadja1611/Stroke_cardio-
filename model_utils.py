# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 11:10:58 2024

@author: nadja
"""
import numpy as np
import torch.nn.functional as F
import torch

def define_inputs(self, features):
  
    features[:, 0:1, :, :, 0] = self.scale_to_zero_one(features[:, 0:1, :, :, 0])
    features[:, 1:2, :, :, 0] = self.scale_to_zero_one(features[:, 1:2, :, :, 0])

    # dwi_gt3 = F.avg_pool2d(dwi_input, (2,2))
    # dwi_gt2 = F.avg_pool2d(dwi_gt3, (2,2))
    # dwi_gt1 = F.avg_pool2d(dwi_gt2, (2,2))
    # adc_input3 = F.avg_pool2d(adc_input, (2,2))
    # adc_input2 = F.avg_pool2d(adc_input3, (2,2))
    # adc_input1 = F.avg_pool2d(adc_input2, (2,2))
    segmentation_mask = features[:, 2:, :, :, 0].to(self.device)
    # seg3 = F.max_pool2d(segmentation_mask, (2,2))
    # seg2 = F.max_pool2d(seg3, (2,2))
    # seg1 = F.max_pool2d(seg2, (2,2))
    input_x = torch.cat([features[:, 0:1, :, :, 0], features[:, 1:2, :, :, 0],
                        features[:, :1, :, :, 0]*features[:, 1:2, :, :, 0]], axis=1)
    return input_x
    
    