IMAGE_H = 224
IMAGE_W = 224

# ImageNet normalization
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ONNX I/O names (must match export_onnx.py)
ONNX_INPUT_NAME  = "input"
ONNX_OUTPUT_NAME = "logits"

# ORT execution providers in priority order
#ORT_PROVIDERS_GPU = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
ORT_PROVIDERS_GPU = ["CUDAExecutionProvider", "CPUExecutionProvider"]
ORT_PROVIDERS_CPU = ["CPUExecutionProvider"]

# COCO 
IMAGENET_PATH = "inference/imagenet_class_index.json"
IMAGENET_LABELS_URL = (
    "https://storage.googleapis.com/download.tensorflow.org"
    "/data/imagenet_class_index.json"
)
