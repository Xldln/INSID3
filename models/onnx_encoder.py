"""ONNX DINOv3 encoder wrapper.

Wraps an ONNX Runtime session to mimic the ``get_intermediate_layers``
interface of ``facebookresearch/dinov3``, so it can be dropped into the
INSID3 pipeline as a drop-in replacement for the PyTorch encoder.

The quantized model path is expected (default: ``model_quantized.onnx``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class ONNXEncoder:
    """DINOv3 encoder backed by ONNX Runtime.

    Attributes:
        patch_size: DINOv3 patch size (always 16).
        num_register_tokens: Number of register tokens (4 for DINOv3-register).
    """

    def __init__(self, onnx_path: str, device: str = "cpu") -> None:
        import onnxruntime

        onnx_path = str(Path(onnx_path).resolve())

        # Pick providers: prefer CUDA if available and requested, else CPU
        available = onnxruntime.get_available_providers()
        providers = []
        if "cuda" in device and "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        sess_opts = onnxruntime.SessionOptions()
        sess_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = onnxruntime.InferenceSession(
            onnx_path, sess_opts, providers=providers
        )
        self._device = device
        self.patch_size = 16
        self.num_register_tokens = 4
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    # ──────── Public API (mimics facebookresearch/dinov3) ────────

    def get_intermediate_layers(
        self, x: torch.Tensor, n: int = 1, reshape: bool = True
    ) -> list[torch.Tensor]:
        """Run the encoder and return intermediate features.

        Args:
            x: ``(B, 3, H, W)`` torch tensor, already normalised with
                ImageNet stats.  H and W must be multiples of 16.
            n: Ignored — always returns the last layer (ONNX exports only
                the final block).
            reshape: If True, strip CLS/register tokens and return
                ``(B, C, H_p, W_p)`` feature maps.  If False, return
                raw ``(B, N, D)`` token sequences.

        Returns:
            List of 1 feature tensor.
        """
        B, C, H, W = x.shape
        ort_inputs = {self.input_name: x.cpu().numpy().astype(np.float32)}
        outputs = self.session.run(self.output_names, ort_inputs)
        hidden = torch.from_numpy(outputs[0])  # (B, N, D)

        if not reshape:
            return [hidden]

        # Strip CLS (first) and register tokens
        n_reg = self.num_register_tokens
        patch_tokens = hidden[:, 1 + n_reg :, :]  # (B, H_p * W_p, D)
        h_p = H // self.patch_size
        w_p = W // self.patch_size
        feat_map = patch_tokens.permute(0, 2, 1).reshape(B, -1, h_p, w_p)
        return [feat_map]

    def to(self, device: str) -> ONNXEncoder:
        """No-op — ONNX session device is fixed at construction."""
        return self
