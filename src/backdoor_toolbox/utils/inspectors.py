import torch
import torch.nn as nn


class ModelInspector:
    """
    Utility for extracting intermediate feature maps and layer weights from a nn.Module.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._features = {}
        self._hooks = []

    def _hook_fn(self, name):
        def fn(module, inp, out):
            self._features[name] = out.detach().cpu()

        return fn

    def register_hooks(self, layer_names: list[str]):
        """
        Attach forward hooks to multiple layers.
        """
        modules = dict(self.model.named_modules())
        for name in layer_names:
            submod = modules.get(name, None)
            if submod is None:
                raise ValueError(f"Layer '{name}' not found in model")
            handle = submod.register_forward_hook(self._hook_fn(name))
            self._hooks.append(handle)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def extract_feature_maps(self, x: torch.Tensor, layer_names: list[str]) -> dict[str, torch.Tensor]:
        """
        Runs a forward pass on x, returns feature maps from the specified layers.
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
        Returns the weight tensor of layer_name (e.g., conv kernels).
        """
        submod = dict(self.model.named_modules()).get(layer_name, None)
        if submod is None:
            raise ValueError(f"Layer '{layer_name}' not found in model")
        if not hasattr(submod, "weight"):
            raise ValueError(f"Layer '{layer_name}' has no 'weight' attribute")
        return submod.weight.detach().cpu()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from backdoor_toolbox.models.cnn.resnet_wrapper import CustomResNet

    # instantiate or load a model
    model = CustomResNet("resnet18", 3, 10, None, "cpu", True)
    model.eval()

    # prepare dummy input batches (batch_size=3, 3×32×32)
    x = torch.randn(3, 3, 32, 32)

    # model inspector
    inspector = ModelInspector(model)

    # extract a feature map from the chosen layer
    feat_maps = inspector.extract_feature_maps(x, layer_names=["model.layer1", "model.layer2", "model.layer3"])

    # log
    print(f"Feature maps: ")
    print(f"{feat_maps.keys()}")
    print(f"{list(feat_maps.values())[0].shape}")
