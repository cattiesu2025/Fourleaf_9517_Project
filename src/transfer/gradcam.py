"""
src/transfer/gradcam.py
------------------------
From-scratch Grad-CAM (Selvaraju et al., ICCV 2017).

Usage:
    cam = GradCAM(model, target_layer=model.layer4[-1])
    heatmap, pred_class = cam.generate(input_tensor, class_idx=None)
    cam.remove_hooks()
"""

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.fwd_handle = self.target_layer.register_forward_hook(forward_hook)
        self.bwd_handle = self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        """
        input_tensor: (1, C, H, W). Returns (heatmap [H,W] in [0,1], class_idx used).
        """
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())

        score = output[0, class_idx]
        score.backward()

        activations = self.activations[0]  # (K, h, w)
        gradients = self.gradients[0]  # (K, h, w)
        weights = gradients.mean(dim=(1, 2))  # (K,)

        cam = torch.zeros(activations.shape[1:], dtype=activations.dtype, device=activations.device)
        for k in range(activations.shape[0]):
            cam += weights[k] * activations[k]
        cam = F.relu(cam)

        H, W = input_tensor.shape[2], input_tensor.shape[3]
        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam, class_idx

    def remove_hooks(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()


def overlay_heatmap(image_rgb_uint8, heatmap, alpha=0.4):
    import matplotlib.cm as cm

    colormap = cm.get_cmap("jet")
    heatmap_colored = (colormap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    overlaid = (image_rgb_uint8 * (1 - alpha) + heatmap_colored * alpha).astype(np.uint8)
    return overlaid


def denormalize_image(tensor_chw, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Undo ImageNet normalization for display. tensor_chw: (3,H,W) torch tensor."""
    img = tensor_chw.detach().cpu().numpy().transpose(1, 2, 0)
    img = (img * np.array(std) + np.array(mean)).clip(0, 1)
    return (img * 255).astype(np.uint8)
