"""
The two models of the proposal, built so that they can be compared fairly.

  U-Net     (segmentation_models_pytorch, encoder ResNet34 pretrained on ImageNet)
            A CNN encoder-decoder: the encoder is the ResNet of lab 2, the decoder is
            the up-sampling half of lab 3's autoencoder, and SKIP CONNECTIONS copy the
            encoder's high-resolution feature maps straight into the decoder.
  SegFormer (encoder MiT-B2 = 'Mix Transformer', pretrained on ImageNet by NVIDIA)
            A transformer encoder (self-attention between image patches at 4 scales)
            with a very light all-MLP decoder.

Both get the same wrapper (SegModel): ImageNet normalisation as the first layer (as in
lab 2), padding to a multiple of 32, and the output cut back to the input size. So
every model takes (B, 3, H, W) in [0, 1] and returns (B, 1, H, W) LOGITS, one per pixel.

Fair comparison: ResNet34-U-Net and MiT-B2-SegFormer both have ~24-25 M parameters.

Run `python models.py` to see the parameter counts and output shapes (no download needed).

Note on the proposal: it says 'SegFormer (Hugging Face)'. segmentation_models_pytorch
contains the same SegFormer architecture (weights hosted on the Hugging Face hub), and
using one library for both models means one training loop for both. If your TA insists
on the `transformers` implementation, use name="segformer_hf": same idea, same wrapper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import IMAGENET_MEAN, IMAGENET_STD, count_parameters

DEFAULT_ENCODER = {"unet": "resnet34", "segformer": "mit_b2", "segformer_hf": "nvidia/mit-b2"}


class SegModel(nn.Module):
    """normalise -> pad to a multiple of 32 -> network -> logits at input size."""

    def __init__(self, net: nn.Module, hf: bool = False, multiple: int = 32):
        super().__init__()
        self.net = net
        self.hf = hf
        self.multiple = multiple
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        h, w = x.shape[-2:]
        x = (x - self.mean) / self.std
        ph, pw = (-h) % self.multiple, (-w) % self.multiple
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        y = self.net(pixel_values=x).logits if self.hf else self.net(x)
        if y.shape[-2:] != x.shape[-2:]:  # SegFormer's decoder works at 1/4 resolution
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return y[..., :h, :w]

    def encoder_parameters(self):
        if self.hf:
            return self.net.segformer.encoder.parameters()
        return self.net.encoder.parameters()


def build_model(name: str = "unet", encoder: str | None = None, pretrained: bool = True,
                freeze_encoder: bool = False) -> SegModel:
    """name: "unet", "segformer" (smp) or "segformer_hf" (Hugging Face transformers).
    pretrained=True downloads ImageNet weights the first time (do that on a LOGIN node,
    see setup_vsc.sh). freeze_encoder=True trains only the decoder (lab 2's frozen ResNet)."""
    encoder = encoder or DEFAULT_ENCODER[name]
    if name in ("unet", "segformer"):
        import segmentation_models_pytorch as smp
        cls = smp.Unet if name == "unet" else smp.Segformer
        net = cls(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None,
                  in_channels=3, classes=1)
        model = SegModel(net)
    elif name == "segformer_hf":
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
        if pretrained:
            net = SegformerForSemanticSegmentation.from_pretrained(encoder, num_labels=1)
        else:
            net = SegformerForSemanticSegmentation(SegformerConfig.from_pretrained(encoder, num_labels=1)) \
                if "/" in encoder else SegformerForSemanticSegmentation(SegformerConfig(num_labels=1))
        model = SegModel(net, hf=True)
    else:
        raise ValueError(f"unknown model {name!r}")
    if freeze_encoder:
        for p in model.encoder_parameters():
            p.requires_grad = False
    return model


if __name__ == "__main__":
    x = torch.rand(2, 3, 400, 400)  # two BDAPPV-sized images; 400 is NOT a multiple of 32: the wrapper pads
    for name in ("unet", "segformer"):
        m = build_model(name, pretrained=False).eval()
        enc = sum(p.numel() for p in m.encoder_parameters())
        with torch.no_grad():
            y = m(x)
        print(f"{name:10s} encoder {DEFAULT_ENCODER[name]:9s} | total {count_parameters(m, False) / 1e6:5.1f} M "
              f"parameters, of which encoder {enc / 1e6:5.1f} M | in {tuple(x.shape)} -> out {tuple(y.shape)}")
    m = build_model("unet", pretrained=False, freeze_encoder=True)
    print(f"unet with frozen encoder: {count_parameters(m) / 1e6:.1f} M trainable parameters")
