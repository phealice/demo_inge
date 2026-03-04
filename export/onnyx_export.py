import argparse
import logging
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from transformers import AutoModelForImageClassification

# ── Patch selective_scan ──────────────────────────────────────────────────────
# selective_scan_fn uses a pybind11 CUDA kernel that torch.export cannot trace.
# We replace it with the pure-PyTorch reference implementation from mamba_ssm
# BEFORE the model is loaded, so every call site picks up the patched version.
import mamba_ssm.ops.selective_scan_interface as _ssi
_ssi.selective_scan_fn = _ssi.selective_scan_ref
# -----------------------------------------------------------------------------

# ── Constants ────────────────────────────────────────────────────────────────

MODEL_ID       = "nvidia/MambaVision-T-1K"
INPUT_H        = 224
INPUT_W        = 224
INPUT_C        = 3
DEFAULT_OPSET  = 17
DEFAULT_OUTPUT = Path("artifacts/mambavision_t_1k.onnx")

# Tolerances for PyTorch vs ONNX comparison.
# FP32 inference with SSM ops can accumulate small numerical differences.
ATOL = 1e-3
RTOL = 1e-2
# Number of random inputs used for the numerical validation.
N_VALIDATION_SAMPLES = 8

SEED = 42

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_model(device: torch.device) -> torch.nn.Module:
    log.info("Loading model %s ...", MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_ID, trust_remote_code=True
    )
    model.to(device).eval()
    log.info("Model loaded on %s", device)
    return model


def torch_inference(model: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
    """Run PyTorch forward pass and return logits as numpy array."""
    with torch.no_grad():
        out = model(x)
    return out["logits"].cpu().numpy()


def export(
    model: torch.nn.Module,
    output_path: Path,
    opset: int,
    batch_size: int,
    device: torch.device,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(batch_size, INPUT_C, INPUT_H, INPUT_W, device=device)

    log.info("Exporting to ONNX (opset %d) → %s", opset, output_path)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        opset_version=opset,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input":  {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        do_constant_folding=True,
    )
    log.info("Export done.")


def check_onnx_model(path: Path) -> None:
    """Run the ONNX graph checker."""
    log.info("Checking ONNX model integrity ...")
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    log.info("ONNX model is valid. IR version: %d, opset: %d",
             model.ir_version,
             model.opset_import[0].version)


def build_ort_session(path: Path) -> ort.InferenceSession:
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if ort.get_device() == "GPU"
        else ["CPUExecutionProvider"]
    )
    log.info("ORT providers: %s", providers)
    return ort.InferenceSession(str(path), providers=providers)


def validate(
    model: torch.nn.Module,
    session: ort.InferenceSession,
    batch_size: int,
    device: torch.device,
) -> None:
    """
    Run N_VALIDATION_SAMPLES forward passes and compare PyTorch vs ORT outputs.
    Raises AssertionError if any sample exceeds tolerance.
    """
    log.info(
        "Validating %d samples (atol=%.0e, rtol=%.0e) ...",
        N_VALIDATION_SAMPLES, ATOL, RTOL,
    )

    rng = np.random.default_rng(SEED)
    max_abs_diff = 0.0
    max_rel_diff = 0.0

    for i in range(N_VALIDATION_SAMPLES):
        x_np = rng.standard_normal(
            (batch_size, INPUT_C, INPUT_H, INPUT_W)
        ).astype(np.float32)

        # PyTorch
        x_torch = torch.from_numpy(x_np).to(device)
        pt_out = torch_inference(model, x_torch)  # (B, num_classes)

        # ORT
        ort_out = session.run(["logits"], {"input": x_np})[0]  # (B, num_classes)

        abs_diff = np.abs(pt_out - ort_out)
        rel_diff = abs_diff / (np.abs(pt_out) + 1e-8)

        max_abs_diff = max(max_abs_diff, abs_diff.max())
        max_rel_diff = max(max_rel_diff, rel_diff.max())

        try:
            np.testing.assert_allclose(pt_out, ort_out, atol=ATOL, rtol=RTOL)
        except AssertionError as e:
            log.error("Sample %d FAILED numerical check", i)
            raise e

        log.debug(
            "Sample %d OK | max abs diff: %.2e | max rel diff: %.2e",
            i, abs_diff.max(), rel_diff.max(),
        )

    log.info(
        "All %d samples passed. | overall max abs diff: %.2e | max rel diff: %.2e",
        N_VALIDATION_SAMPLES, max_abs_diff, max_rel_diff,
    )


def check_topk_consistency(
    model: torch.nn.Module,
    session: ort.InferenceSession,
    device: torch.device,
    k: int = 5,
) -> None:
    """
    Sanity check at the prediction level: top-k classes must match exactly
    between PyTorch and ORT on a fixed input.
    """
    rng = np.random.default_rng(SEED + 1)
    x_np = rng.standard_normal((1, INPUT_C, INPUT_H, INPUT_W)).astype(np.float32)

    x_torch = torch.from_numpy(x_np).to(device)
    pt_logits = torch_inference(model, x_torch)
    ort_logits = session.run(["logits"], {"input": x_np})[0]

    pt_topk  = np.argsort(pt_logits[0])[::-1][:k].tolist()
    ort_topk = np.argsort(ort_logits[0])[::-1][:k].tolist()

    if pt_topk != ort_topk:
        raise AssertionError(
            f"Top-{k} mismatch!\n  PyTorch: {pt_topk}\n  ORT:     {ort_topk}"
        )

    log.info("Top-%d classes match: %s", k, pt_topk)


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MambaVision to ONNX")
    parser.add_argument("--output",     type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--opset",      type=int,  default=DEFAULT_OPSET)
    parser.add_argument("--batch-size", type=int,  default=1)
    parser.add_argument("--cpu",        action="store_true",
                        help="Force CPU even if CUDA is available")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    log.info("Using device: %s", device)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model   = load_model(device)
    export(model, args.output, args.opset, args.batch_size, device)

    check_onnx_model(args.output)

    session = build_ort_session(args.output)

    validate(model, session, args.batch_size, device)
    check_topk_consistency(model, session, device)

    log.info("Export and validation successful → %s", args.output)


if __name__ == "__main__":
    main()