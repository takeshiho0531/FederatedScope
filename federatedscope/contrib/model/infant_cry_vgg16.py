from torchvision import models
import torch.nn as nn
from federatedscope.register import register_model

def call_my_net(model_config, local_data):
    if model_config.type == "mynet":
        use_pretrained = True
        net = models.vgg16(pretrained=use_pretrained)
        net.classifier[6] = nn.Linear(in_features=4096, out_features=5)
        return net

register_model("mynet", call_my_net)
