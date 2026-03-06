# =============================================================================
# MambaVision ROS2 Inference — x86 GPU (cloud / workstation)
# =============================================================================
#
# Target hardware : any x86_64 machine with an NVIDIA GPU
# CUDA provider  : onnxruntime-gpu (bundles CUDA 12.x + cuDNN 9.x)
# TRT             : built at first container start via trtexec
#                   requires --gpus all at docker run
#
# BUILD:
#   docker build \
#     --build-arg ONNX_MODEL_PATH=/path/to/custom.onnx \
#     -t mambavision-ros:latest \
#     -f docker/Dockerfile .
#
# RUN EXAMPLES:
#   # GPU
#   docker run --gpus all mambavision-ros:latest
#   docker run ----runtime nvidia mambavision-ros:latest 
#
#   # CPU only
#   docker run mambavision-ros:latest


# =============================================================================

# -----------------------------------------------------------------------------
# WHY osrf/ros:humble-ros-base-jammy
# -----------------------------------------------------------------------------
# - Official OSRF image, Ubuntu 22.04 LTS, ROS2 Humble LTS (EOL 2027)
# - "ros-base" = core ROS2 only, no GUI tools → smaller image (~400MB vs ~2GB
#   for desktop)
# - Python 3.10 native on Jammy → no version conflicts with our deps
# - We do NOT use nvidia/cuda as base: onnxruntime-gpu >= 1.18 bundles its own
#   CUDA/cuDNN libraries in the pip wheel. The NVIDIA Container Runtime
#   (--gpus all) injects host GPU drivers at runtime — no CUDA base needed.
# - TensorRT IS needed for trtexec. We install it via apt from the NVIDIA
#   TensorRT repo added below. This is the only reason we need NVIDIA packages.
# -----------------------------------------------------------------------------
FROM ros:jazzy-ros-base AS build_base

# -----------------------------------------------------------------------------
# BUILD ARGS
# -----------------------------------------------------------------------------
ARG ONNX_MODEL_PATH

# Fail the build immediately if the ONNX file is missing.
# Without this, COPY would silently produce a broken image.
RUN --mount=type=bind,source=.,target=/ctx \
    test -f /ctx/${ONNX_MODEL_PATH} || \
    (echo "ERROR: ONNX model not found at ${ONNX_MODEL_PATH}" && exit 1)

# -----------------------------------------------------------------------------
# SYSTEM DEPENDENCIES
# -----------------------------------------------------------------------------
# libgl1 + libglib2.0-0 : OpenCV headless runtime (no display needed)
# python3-pip            : pip for Python deps
# ros-humble-cv-bridge   : ROS2 ↔ OpenCV image bridge (used by the ROS node)
# apt-get clean + rm -rf /var/lib/apt/lists/* reduces layer size by ~50MB
# by removing the apt package index that is not needed at runtime.
# -----------------------------------------------------------------------------

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        python3-pip \
        ros-jazzy-cv-bridge \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# # -----------------------------------------------------------------------------
# # PYTHON DEPENDENCIES
# # -----------------------------------------------------------------------------
# # requirements.txt is copied BEFORE application code intentionally.
# # Docker caches each layer. If requirements.txt hasn't changed, the pip install
# # layer is fully reused on rebuild even if inference/ code changed. This is the single most impactful Docker caching optimization.
# #
# # --no-cache-dir : do not store the pip download cache inside the image
# # --upgrade pip  : avoids pip resolver bugs in older versions
# #
# # onnxruntime-gpu: provides CUDAExecutionProvider + TensorrtExecutionProvider.
# # Do NOT install both onnxruntime and onnxruntime-gpu — they conflict.
# # The GPU package falls back to CPU automatically if no GPU is found.
# # -----------------------------------------------------------------------------

WORKDIR /app
COPY requirements.txt .

# ------------------------------------------------------------------------------

FROM build_base AS debug


RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt
# -----------------------------------------------------------------------------
# APPLICATION CODE + MODEL
# -----------------------------------------------------------------------------
# Separate COPY instructions so that:
#   - changing inference/ code does not invalidate the model layer
#   - changing the ONNX model does not invalidate the code layer
# Layers ordered from least-frequently-changed to most-frequently-changed.
# -----------------------------------------------------------------------------
COPY inference/ ./inference/
COPY ${ONNX_MODEL_PATH} ./models/mambavision.onnx

# ROS2 workspace skeleton (populated when ROS node is added later)
RUN mkdir -p /ros2_ws/src

# -----------------------------------------------------------------------------
# ENVIRONMENT VARIABLES
# -----------------------------------------------------------------------------
# PYTHONUNBUFFERED=1 : Python stdout/stderr are not buffered → logs appear
#                      immediately in docker logs / ROS2 console output.
# PYTHONDONTWRITEBYTECODE=1 : no .pyc files inside the container (cleaner).
# MODEL_PATH : makes the model path injectable at runtime via -e if needed.
# ORT_USE_GPU : read by infer.py to decide provider order (1=GPU, 0=CPU).
# -----------------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/models/mambavision.onnx \
    ORT_USE_GPU=1

# -----------------------------------------------------------------------------
# ENTRYPOINT + DEFAULT CMD
# -----------------------------------------------------------------------------
# The entrypoint sources ROS2, optionally builds the TRT engine, then execs
# whatever CMD is passed. See docker/entrypoint.sh for details.
#
# Default CMD is a smoke test: prints ORT version + available providers.
# Override at docker run to launch the ROS2 node:
#   docker run ... mambavision-ros:v1.0.0 ros2 run mambavision_node infer_node
# -----------------------------------------------------------------------------
COPY --chown=appuser:appuser docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

# Smoke test - check exists
# CMD ["python3", "-c", "\
# import onnxruntime as ort; \
# print('ORT version     :', ort.__version__); \
# print('Available providers:', ort.get_available_providers()); \
# print('Active device   :', ort.get_device()); \
# "]

CMD ["bash"]