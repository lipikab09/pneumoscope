import torch, torch.nn as nn
from torchvision import models
from .config import OUT

def build(name="densenet121", pretrained=True):
    w = "DEFAULT" if pretrained else None
    if name == "densenet121":      # CheXNet backbone
        m = models.densenet121(weights=w); m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1024, 1))
        head, layer = m.classifier, m.features.denseblock4
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=w); m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, 1))
        head, layer = m.classifier, m.features[-1]
    elif name == "resnet50":
        m = models.resnet50(weights=w); m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(2048, 1))
        head, layer = m.fc, m.layer4
    else: raise ValueError(name)
    return m, head, layer

def set_backbone_grad(model, head, flag):
    hp = {id(p) for p in head.parameters()}
    for p in model.parameters(): p.requires_grad = flag or id(p) in hp

def load(ckpt=OUT / "best.pt", device="cpu"):
    c = torch.load(ckpt, map_location=device)
    m, _, layer = build(c["name"], pretrained=False); m.load_state_dict(c["state"])
    return m.to(device).eval(), layer, c
