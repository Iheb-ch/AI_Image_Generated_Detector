import torch
import torch.nn.functional as F
from torch import nn
import timm
from torch.nn.parameter import Parameter

class Backbone(nn.Module) :
    def __init__(self,name,pretrained) :
        super(Backbone,self).__init__()
        self.net = timm.create_model(name,pretrained=pretrained)
        self.out_features = self.net.get_classifier().in_features
    def forward(self,x) :
        x = self.net.forward_features(x)
        return x

class CustomModel(nn.Module) :
    def __init__(self) :
        super(CustomModel,self).__init__()
        self.backbone = Backbone("tf_efficientnetv2_s",False)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.backbone.out_features,1)
    def forward(self,x) :
        x = self.backbone(x)
        x = self.pooling(x).squeeze()
        target = self.head(x)
        output = {}
        output['label'] = target
        return output
