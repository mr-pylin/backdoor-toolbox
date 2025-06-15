import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor:
    """Utility class to extract intermediate feature maps and layer weights from a PyTorch model."""

    def __init__(self, model: nn.Module):
        """
        Initialize the FeatureExtractor.

        Args:
            model (nn.Module): The model from which to extract features.
        """
        self.model = model
        self._features = {}
        self._hooks = []

    def _hook_fn(self, name: str) -> callable:
        def fn(module, inp, out):
            self._features[name] = out.detach().cpu()

        return fn

    def register_hooks(self, layer_names: list[str]) -> None:
        """
        Register forward hooks on specified layers.

        Args:
            layer_names (list[str]): List of hierarchical layer names.

        Raises:
            ValueError: If any layer name is not found in the model.
        """
        for name in layer_names:
            try:
                submod = self.model.get_submodule(name)
            except AttributeError:
                raise ValueError(f"Layer '{name}' not found in model")
            handle = submod.register_forward_hook(self._hook_fn(name))
            self._hooks.append(handle)

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def extract_feature_maps(self, x: torch.Tensor, layer_names: list[str]) -> dict[str, torch.Tensor]:
        """
        Extract intermediate feature maps from specified layers.

        Args:
            x (torch.Tensor): Input tensor.
            layer_names (list[str]): List of layers to extract from.

        Returns:
            dict[str, torch.Tensor]: Feature maps keyed by layer name.

        Raises:
            RuntimeError: If any requested feature map is missing.
        """
        self._features.clear()
        self.register_hooks(layer_names)
        _ = self.model(x)  # forward
        self.remove_hooks()
        output = {name: self._features[name] for name in layer_names if name in self._features}
        missing = set(layer_names) - output.keys()
        if missing:
            raise RuntimeError(f"Missing feature maps for layers: {missing}")
        return output

    def get_layer_weights(self, layer_name: str) -> torch.Tensor:
        """
        Retrieve the weights of a specified layer.

        Args:
            layer_name (str): Hierarchical name of the layer.

        Returns:
            torch.Tensor: Detached CPU tensor of weights.

        Raises:
            ValueError: If the layer is not found or lacks a weight attribute.
        """
        try:
            submod = self.model.get_submodule(layer_name)
        except AttributeError:
            raise ValueError(f"Layer '{layer_name}' not found in model")
        if not hasattr(submod, "weight"):
            raise ValueError(f"Layer '{layer_name}' has no 'weight' attribute")
        return submod.weight.detach().cpu()


class GradCAM:
    """Computes Grad-CAM heatmaps for specified layers in a neural network."""

    def __init__(self, model: nn.Module, target_layers: list[str]):
        """
        Initialize the GradCAM module.

        Args:
            model (nn.Module): The model to inspect.
            target_layers (list[str]): Names of layers to register for Grad-CAM.
        """
        self.model = model
        self.target_layers = target_layers
        self.named_modules = dict(model.named_modules())
        self.activations = {}
        self.gradients = {}
        self._forward_hooks = []
        self._backward_hooks = []

    def _make_hooks(self, layer_name: str) -> None:
        def forward_hook(module, inp, out):
            self.activations[layer_name] = out

        def backward_hook(module, grad_in, grad_out):
            self.gradients[layer_name] = grad_out[0]

        try:
            layer = self.model.get_submodule(layer_name)
        except AttributeError:
            raise ValueError(f"Layer '{layer_name}' not found in model")

        self._forward_hooks.append(layer.register_forward_hook(forward_hook))
        self._backward_hooks.append(layer.register_full_backward_hook(backward_hook))

    def __enter__(self):
        for name in self.target_layers:
            self._make_hooks(name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for h in self._forward_hooks + self._backward_hooks:
            h.remove()
        self._forward_hooks.clear()
        self._backward_hooks.clear()

    def generate(self, x: torch.Tensor, target_class: int | list[int] | None = None) -> dict[str, torch.Tensor]:
        """
        Generate Grad-CAM heatmaps for each target layer.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W] or [C, H, W].
            target_class (int | list[int] | None): Target class or classes for which gradients are computed.

        Returns:
            dict[str, torch.Tensor]: Heatmaps per layer with shape [B, H, W].
        """
        is_single = x.dim() == 3
        if is_single:
            x = x.unsqueeze(0)

        B = x.size(0)
        logits = self.model(x)

        if target_class is None:
            target_indices = logits.argmax(dim=1)
        elif isinstance(target_class, int):
            target_indices = torch.full((B,), target_class, dtype=torch.long, device=x.device)
        else:
            target_indices = torch.tensor(target_class, device=x.device)

        heatmaps = {layer: [] for layer in self.target_layers}

        for i in range(B):
            self.model.zero_grad(set_to_none=True)
            logits[i, target_indices[i]].backward(retain_graph=True)

            for layer in self.target_layers:
                A = self.activations[layer][i]  # [C,H,W]
                G = self.gradients[layer][i]  # [C,H,W]
                w = G.mean(dim=(1, 2), keepdim=True)  # [C,1,1]
                cam = torch.relu((w * A).sum(dim=0, keepdim=True))  # [1,H,W]
                cam = F.interpolate(cam.unsqueeze(0), size=x.shape[2:], mode="bilinear", align_corners=False)
                cam = cam.squeeze()
                if cam.max() > 0:
                    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                heatmaps[layer].append(cam.cpu().unsqueeze(0))  # [1,H,W]

        for layer in self.target_layers:
            heatmaps[layer] = torch.cat(heatmaps[layer], dim=0)  # [B,H,W]

        return heatmaps

    def overlay_heatmaps(
        self,
        orig_imgs: torch.Tensor,
        heatmaps: dict[str, torch.Tensor],
        alpha: float = 0.4,
    ) -> dict[str, torch.Tensor]:
        """
        Overlay Grad-CAM heatmaps onto original images.

        Args:
            orig_imgs (torch.Tensor): Images of shape [B, 3, H, W], values in [0, 1] or [0, 255].
            heatmaps (dict[str, torch.Tensor]): Dict of [B, H, W] heatmaps per layer.
            alpha (float): Transparency factor for heatmap overlay.

        Returns:
            dict[str, torch.Tensor]: Images with heatmaps overlaid, shape [B, 3, H, W] per layer.
        """
        if orig_imgs.max() > 1:
            orig_imgs = orig_imgs / 255.0

        B = orig_imgs.size(0)
        overlays = {}

        for layer, cams in heatmaps.items():
            batch_overlay = []
            for i in range(B):
                img = orig_imgs[i]
                cam = cams[i]

                cam_np = cam.detach().cpu().numpy()
                color_map = cm.jet(cam_np)[..., :3]  # [H, W, 3]
                heatmap_tensor = torch.from_numpy(color_map).permute(2, 0, 1).to(img.device).float()  # [3, H, W]

                overlay = alpha * heatmap_tensor + (1 - alpha) * img
                overlay = torch.clamp(overlay, 0, 1)
                batch_overlay.append(overlay.unsqueeze(0))  # [1, 3, H, W]

            overlays[layer] = torch.cat(batch_overlay, dim=0)  # [B, 3, H, W]

        return overlays


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from backdoor_toolbox.models.cnn.resnet_wrapper import CustomResNet

    # instantiate or load a model
    model = CustomResNet("resnet18", 3, 10, None, "cpu", True)
    model.eval()

    # prepare dummy input batches (batch_size=3, 3×32×32)
    x = torch.randn(8, 3, 32, 32)

    # model inspector
    inspector = FeatureExtractor(model)

    # extract a feature map from the chosen layer
    feat_maps = inspector.extract_feature_maps(x, layer_names=["layer1", "layer2", "layer3"])

    # log
    print(f"Feature maps: ")
    print(f"{feat_maps.keys()}")
    print(f"{list(feat_maps.values())[0].shape}")

    # gradcam
    grad_cam = GradCAM(model, target_layers=["layer1", "layer2", "layer3"])

    # compute a Grad-CAM heatmap for the same input
    with grad_cam as gc:
        heatmaps = gc.generate(x)

    # log
    print(f"GradCAM: ")
    print(f"{heatmaps.keys()}")
    print(f"{list(heatmaps.values())[0].shape}")

    # (optional) sanity check: ensure values are non-negative
    print(
        f"Heatmap min/max: {list(heatmaps.values())[0][0].min().item():.4f} / {list(heatmaps.values())[0][0].max().item():.4f}"
    )

    # overlay
    overlay = gc.overlay_heatmaps(x, heatmaps, alpha=0.4)
    print(f"{overlay.keys()}")
    print(f"{list(overlay.values())[0].shape}")

    for k, v in overlay.items():
        print(k)
        print(type(v))
        print(v.dtype)
        print(v.shape)
    plt.imshow(list(overlay.values())[0][0].permute(1, 2, 0))
    plt.axis("off")
    plt.show()
