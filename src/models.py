"""TSN-style clip classifier with three torchvision backbones.

Input (B, T, 3, S, S) -> backbone per frame -> mean over T -> Linear(d, C).

Backbone feature = output of the convolutional trunk + global average pool;
the ImageNet classification head is removed:
  mobilenet_v3_large : features + avgpool -> d = 960
                       (the pretrained 960->1280 classifier layer is dropped)
  efficientnet_b0    : features + avgpool -> d = 1280
  resnet18           : conv1..layer4 + avgpool -> d = 512

Weights are pinned to IMAGENET1K_V1 for reproducibility.

Transfer strategies:
  scratch : random init, all weights trained
  frozen  : backbone frozen, only the linear head trained
  partial : backbone frozen except its final stage (PARTIAL below) + head
  full    : all weights trained
BatchNorm layers whose parameters are all frozen are kept in eval mode, so
their running statistics stay at the pretrained values.
"""
import torch
import torch.nn as nn
from torchvision import models as tvm

# final-stage modules trained in the "partial" strategy
PARTIAL = {
    "mobilenet_v3_large": "features[13:]",   # last 160-channel stage + final 1x1 conv (960)
    "efficientnet_b0": "features[7:]",       # stage 7 (320 ch) + final 1x1 conv (1280)
    "resnet18": "layer4",
}


def build_backbone(name, pretrained):
    if name == "mobilenet_v3_large":
        w = tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        m = tvm.mobilenet_v3_large(weights=w)
        body = nn.Sequential(m.features, m.avgpool, nn.Flatten(1))
        partial = list(m.features[13:])
        dim = 960
    elif name == "efficientnet_b0":
        w = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = tvm.efficientnet_b0(weights=w)
        body = nn.Sequential(m.features, m.avgpool, nn.Flatten(1))
        partial = list(m.features[7:])
        dim = 1280
    elif name == "resnet18":
        w = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = tvm.resnet18(weights=w)
        body = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool,
                             m.layer1, m.layer2, m.layer3, m.layer4, m.avgpool, nn.Flatten(1))
        partial = [m.layer4]
        dim = 512
    else:
        raise ValueError(f"unknown backbone {name}")
    return body, dim, partial


class TSN(nn.Module):
    def __init__(self, backbone, n_classes, pretrained=True, dropout=0.0):
        super().__init__()
        self.body, self.dim, self._partial = build_backbone(backbone, pretrained)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(self.dim, n_classes)

    def forward(self, x):
        b, t = x.shape[:2]
        f = self.body(x.flatten(0, 1))           # (B*T, d)
        f = f.view(b, t, -1).mean(dim=1)         # temporal average pooling
        return self.head(self.drop(f))


def build_model(cfg, n_classes):
    mc = cfg["model"]
    transfer = mc["transfer"]
    pretrained = mc["pretrained"] and transfer != "scratch"
    model = TSN(mc["backbone"], n_classes, pretrained=pretrained, dropout=mc["head_dropout"])
    if transfer in ("frozen", "partial"):
        for p in model.body.parameters():
            p.requires_grad = False
        if transfer == "partial":
            for mod in model._partial:
                for p in mod.parameters():
                    p.requires_grad = True
    elif transfer not in ("scratch", "full"):
        raise ValueError(f"unknown transfer strategy {transfer}")
    return model


def set_train_mode(model, freeze_bn_in_frozen=True):
    """model.train(), but keep fully frozen BatchNorm layers in eval mode."""
    model.train()
    if freeze_bn_in_frozen:
        for m in model.body.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                params = list(m.parameters())
                if params and not any(p.requires_grad for p in params):
                    m.eval()


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
