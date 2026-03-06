from inference import MambaVisionInference
import cv2


infer = MambaVisionInference('/app/models/mambavision_t_1k.onnx', '/app/inference/imagenet_class_index.json')
img = cv2.imread('/app/tests/bear.jpg')
print(infer.top_k(img, k=5))