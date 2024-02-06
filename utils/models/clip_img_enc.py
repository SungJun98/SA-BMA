
import torch
import torch.nn as nn

import clip


class resnet50_clip(nn.Module):
    def __init__(self, num_classes):
        super(resnet50_clip, self).__init__()
        clip_model, _ = clip.load("RN50")
        self.base = clip_model.visual
        self.fc = nn.Linear(in_features=1024, out_features=num_classes, bias=True).to(dtype=torch.float16)
        

    def forward(self, x):
        x = self.base(x)        
        x = torch.flatten(x, 1)
        return self.fc(x)
    
    

class resnet101_clip(nn.Module):
    def __init__(self, num_classes):
        super(resnet101_clip, self).__init__()
        clip_model, _ = clip.load("RN101")
        self.base = clip_model.visual
        self.fc = nn.Linear(in_features=1024, out_features=num_classes, bias=True).to(dtype=torch.float16)

    def forward(self, x):
        x = self.base(x)        
        x = torch.flatten(x, 1)
        return self.fc(x)
    
    
    
class vitb16_clip(nn.Module):
    def __init__(self, num_classes):
        super(vitb16_clip, self).__init__()
        clip_model, _ = clip.load("ViT-B/16")
        self.base = clip_model.visual.float()
        self.base.proj = None
        self.head_drop = nn.Dropout(p=0.0, inplace=False) .float()
        self.head = nn.Linear(in_features=768, out_features=num_classes, bias=True).float()

    def forward(self, x):
        x = self.base(x)
        x = self.head_drop(x)
        return self.head(x)