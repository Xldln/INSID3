"""Real-time inference with INSID3 using a RealSense camera.

Usage:
    python realtime_inference.py

Controls:
    q - quit
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from models import build_insid3


def _overlay_mask(frame_bgr: np.ndarray, mask: np.ndarray,
                  color: tuple[int, int, int] = (0, 220, 80),
                  alpha: float = 0.35) -> np.ndarray:
    """Blend a binary mask onto a BGR frame."""
    out = frame_bgr.astype(np.float32)
    out[mask] = (1.0 - alpha) * out[mask] + alpha * np.array(color, dtype=np.float32)
    return out.clip(0, 255).astype(np.uint8)


def main() -> None:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Build model and set reference (cat) ──
    model = build_insid3(device=device)
    model.set_reference("assets/pencilbag_ref.jpg", "assets/pencilbag_ref_mask.png")
    print("Reference loaded.")

    # ── RealSense ──
    import pyrealsense2 as rs
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(cfg)
    print("RealSense started. Press 'q' to quit.")

    fps_smooth = 0.0
    alpha = 0.1  # EMA smoothing factor

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Raw frame (HWC BGR uint8)
            frame = np.asanyarray(color_frame.get_data())

            # ── Inference ──
            tick = cv2.getTickCount()

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            model.set_target(Image.fromarray(frame_rgb))
            pred_mask = model.predict_mask(
                model._ref_images, model._ref_masks, model._tgt_image
            )

            elapsed = (cv2.getTickCount() - tick) / cv2.getTickFrequency()
            fps = 1.0 / max(elapsed, 1e-8)
            fps_smooth = alpha * fps + (1.0 - alpha) * fps_smooth

            # ── Visualisation ──
            mask = pred_mask.cpu().numpy()  # (H, W) bool, same size as frame

            display = frame.copy()

            if mask.any():
                # Semi-transparent overlay
                display = _overlay_mask(display, mask)

                # Bounding box around the largest component
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                )
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest)
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 220, 80), 2)
                    label = f"cat  {int(cv2.contourArea(largest))}px"
                    cv2.putText(display, label, (x, y - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 80), 2)

            # FPS overlay
            cv2.putText(display, f"FPS: {fps_smooth:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("INSID3 Real-time Inference", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
