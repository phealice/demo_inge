"""
MambaVision ONNX inference.

Assumes a valid ONNX model on disk (produced by export/export_onnx.py).
No dependency on PyTorch or ROS — pure ORT + numpy.
"""

import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .constants import (
    ONNX_INPUT_NAME,
    ONNX_OUTPUT_NAME,
    ORT_PROVIDERS_CPU,
    ORT_PROVIDERS_GPU,
)
from .preprocess import preprocess, preprocess_from_path

log = logging.getLogger(__name__)


class MambaVisionInference:
    """
    Wraps an ONNX MambaVision model for single-image or batch inference.

    Parameters
    ----------
    model_path : Path
        Path to the .onnx file.
    use_gpu : bool
        If True, tries CUDAExecutionProvider first, falls back to CPU.
    """

    def __init__(self, model_path: Path, use_gpu: bool = True) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        providers = ORT_PROVIDERS_GPU if use_gpu else ORT_PROVIDERS_CPU
        log.info("Loading ONNX model from %s (providers: %s)", model_path, providers)

        self._session = ort.InferenceSession(str(model_path), providers=providers)
        self._active_provider = self._session.get_providers()[0]
        log.info("ORT session ready — active provider: %s", self._active_provider)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Run inference on a single BGR uint8 image (HxWxC, OpenCV format).

        Returns
        -------
        logits : np.ndarray of shape (num_classes,) — raw unnormalized scores.
        """
        tensor = preprocess(image_bgr)          # (1, C, H, W) float32
        return self._run_tensor(tensor)[0]      # squeeze batch dim

    def run_from_path(self, path: Path) -> np.ndarray:
        """Convenience: load image from disk + inference."""
        tensor = preprocess_from_path(Path(path))
        return self._run_tensor(tensor)[0]

    def top_k(self, image_bgr: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
        """
        Returns the top-k (class_index, score) pairs sorted by descending score.
        Scores are raw logits (not softmax).
        """
        logits  = self.run(image_bgr)
        indices = np.argsort(logits)[::-1][:k]
        return [(int(i), float(logits[i])) for i in indices]

    def probabilities(self, image_bgr: np.ndarray) -> np.ndarray:
        """Run inference and return softmax probabilities."""
        logits = self.run(image_bgr)
        exp    = np.exp(logits - logits.max())   # numerically stable softmax
        return exp / exp.sum()

    @property
    def active_provider(self) -> str:
        return self._active_provider

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run_tensor(self, tensor: np.ndarray) -> np.ndarray:
        """
        Raw ORT call on a (B, C, H, W) float32 numpy array.
        Returns (B, num_classes) logits.
        """
        return self._session.run(
            [ONNX_OUTPUT_NAME],
            {ONNX_INPUT_NAME: tensor},
        )[0]
    

if __name__ == "__main__":
    model = MambaVisionInference('/home/alice/code/demo_inge/export/artifacts/mambavision_t_1k.onnx', True)
    image_test = np.random.rand(224, 224, 3)
    model.run(image_test)
    print("Simple sanity check pass !")