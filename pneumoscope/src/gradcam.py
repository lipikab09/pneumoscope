import numpy as np, torch, torch.nn.functional as F
from matplotlib import colormaps

class GradCAM:
    """Grad-CAM (Selvaraju et al. 2017) for a single-logit binary classifier."""
    def __init__(s, model, layer):
        s.m, s.a, s.g = model, None, None
        layer.register_forward_hook(s._fwd)
    def _fwd(s, _, __, out):
        s.a = out
        if out.requires_grad: out.register_hook(lambda g: setattr(s, "g", g))
    def __call__(s, x):
        s.m.zero_grad(); x = x.clone().requires_grad_(True)
        logit = s.m(x); logit.sum().backward()
        w = s.g.mean((2, 3), keepdim=True)
        cam = F.relu((w * s.a).sum(1, keepdim=True))
        cam = F.interpolate(cam, x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        mn, mx = cam.amin((1, 2), keepdim=True), cam.amax((1, 2), keepdim=True)
        cam = (cam - mn) / (mx - mn + 1e-8)
        return torch.sigmoid(logit).detach().squeeze(1).cpu().numpy(), cam.detach().cpu().numpy()

def overlay(img_rgb, cam, alpha=0.45, cmap="jet"):
    """img_rgb: HxWx3 uint8, cam: HxW in [0,1] -> blended uint8 image."""
    heat = (colormaps[cmap](cam)[..., :3] * 255).astype(np.float32)
    a = alpha * cam[..., None]  # transparent where model ignores the region
    return (img_rgb * (1 - a) + heat * a).clip(0, 255).astype(np.uint8)
