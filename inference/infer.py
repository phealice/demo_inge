"""
MambaVision ONNX inference.

Assumes a valid ONNX model on disk (produced by export/export_onnx.py).
No dependency on PyTorch or ROS — pure ORT + numpy.
"""

import logging
from pathlib import Path

import numpy as np
import json
import onnxruntime as ort

from .constants import (
    ONNX_INPUT_NAME,
    ONNX_OUTPUT_NAME,
    ORT_PROVIDERS_CPU,
    ORT_PROVIDERS_GPU,
    IMAGENET_PATH,
)
from .preprocess import preprocess, preprocess_from_path

log = logging.getLogger(__name__)

def _load_imagenet_labels(labels_path: Path | None) -> dict[int, str] | None:
    labels_path = Path(labels_path)
    if not labels_path.exists():
        log.warning("Labels file not found: %s — falling back to indices", labels_path)
        return None
    with labels_path.open() as f:
        raw = json.load(f)
    return {int(k): v[1] for k, v in raw.items()}

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

    def __init__(self, model_path: Path, labels_path:Path, use_gpu: bool = True) -> None:
        model_path = Path(model_path)
        labels_path = Path(labels_path)
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        providers = ORT_PROVIDERS_GPU if use_gpu else ORT_PROVIDERS_CPU
        log.info("Loading ONNX model from %s (providers: %s)", model_path, providers)

        self._labels = _load_imagenet_labels(labels_path) #image net labels 
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

    def label(self, class_index: int) -> str:
        """
        Resolve a class index to its ImageNet label string.
        Returns the index as string if labels are not loaded.
        """
        if self._labels is None:
            return str(class_index)
        return self._labels.get(class_index, str(class_index))
    
    def top_k(self, image_bgr: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
        """
        Returns the top-k (class_index, score) pairs sorted by descending score.
        Scores are raw logits (not softmax).
        """
        logits  = self.run(image_bgr)
        exp = np.exp(logits - logits.max())
        logits = exp / exp.sum()
        indices = np.argsort(logits)[::-1][:k]
        return [(self.label(int(i)), float(logits[i])) for i in indices]

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
    from .preprocess import load_image_bgr
    model = MambaVisionInference('/home/alice/code/demo_inge/export/artifacts/mambavision_t_1k.onnx', IMAGENET_PATH, True)
    path_image = "./tests/bear.jpg"
    print(Path(path_image))
    image = load_image_bgr(Path(path_image))
    print(model.top_k(image))
    print("Simple sanity check pass !")