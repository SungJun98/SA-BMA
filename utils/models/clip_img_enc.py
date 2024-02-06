
import torch
import torch.nn as nn

import clip

class resnet50_clip(nn.Module):
    def __init__(self, num_classes):
        super(resnet50_clip, self).__init__()
        clip_model, _ = clip.load("RN50")
        self.base = clip_model.visual
        self.base.attnpool = None
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.base(x)
        x = self.avgpool(x)
        return self.fc(x)
    
    

class resnet101_clip(nn.Module):
    def __init__(self, num_classes):
        super(resnet101_clip, self).__init__()
        clip_model, _ = clip.load("RN101")
        self.base = clip_model.visual
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.base(x)
        x = self.avgpool(x)
        return self.fc(x)
    
    
    
class vitb16_clip(nn.Module):
    def __init__(self, num_classes):
        super(vitb16_clip, self).__init__()
        clip_model, _ = clip.load("ViT-B/16")
        self.base = clip_model.visual
        self.fc_norm = nn.Identity()
        self.head_drop = nn.Dropout(p=0.0, inplace=False)
        self.head = nn.Linear(in_features=768, out_features=num_classes, ias=True)

    def forward(self, x):
        x = self.base(x)
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return self.head(x)