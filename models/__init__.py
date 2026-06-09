"""Model construction utilities for INSID3."""

from __future__ import annotations

import torch

from models.insid3 import INSID3
from models.onnx_encoder import ONNXEncoder

_HUB_NAMES = {
    "small": "dinov3_vits16",
    "base": "dinov3_vitb16",
    "large": "dinov3_vitl16",
}

_WEIGHTS = {
    "small": "pretrain/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "base": "pretrain/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "large": "pretrain/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
}

_ONNX_DEFAULT = "pretrain/onnx_weights/onnx/model_quantized.onnx"


def _build_encoder(model_size: str = "large", onnx_path: str | None = None):
    if onnx_path is not None:
        return ONNXEncoder(onnx_path)

    encoder = torch.hub.load(
        "facebookresearch/dinov3",
        _HUB_NAMES[model_size],
        weights=_WEIGHTS[model_size],
    )
    return torch.compile(encoder)


def build_insid3(
    *,
    model_size: str = "large",
    image_size: int = 1024,
    svd_components: int = 500,
    tau: float = 0.6,
    merge_threshold: float = 0.2,
    mask_refiner: str = "bilinear",
    resize_to_orig_size: bool = True,
    device: str = "cuda",
    onnx_path: str | None = None,
):
    encoder = _build_encoder(model_size, onnx_path=onnx_path)
    model = INSID3(
        encoder=encoder,
        image_size=image_size,
        svd_components=svd_components,
        tau=tau,
        merge_threshold=merge_threshold,
        mask_refiner=mask_refiner,
        resize_to_orig_size=resize_to_orig_size,
        device=device,
    )
    for param in model.parameters():
        param.requires_grad = False
    return model


def build_insid3_from_args(args):
    return build_insid3(
        model_size=args.model_size,
        image_size=args.image_size,
        svd_components=int(args.svd_comps),
        tau=args.tau,
        merge_threshold=args.merge_thresh,
        mask_refiner='crf' if getattr(args, 'crf_mask_refinement', False) else 'bilinear',
        resize_to_orig_size=False,
        device=args.device,
        onnx_path=getattr(args, 'onnx_model', None),
    )