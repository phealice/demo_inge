[![GitHub Actions Demo](https://github.com/phealice/demo_inge/actions/workflows/github-actions-demo.yml/badge.svg)](https://github.com/phealice/demo_inge/actions/workflows/github-actions-demo.yml)

## Overview
The goal is to write a short demo of a Mamba deployement through Docker-ROS image.
- show production grade deployment 
    - deterministic build
    - optimized Docker image
    - clear workflow
    - CI/CD with github actions
- separate dev and deployment of the robotic stack for hygrometer project 

============= 

## Installation / Requirements


## Project Structure


## TODO
- [x] Mamba vision to onnx
- [ ] Complete the Readme
    - [ ] Write the chunks


## Execution 
En local windows -> jetson (arm64)
docker buildx build \
  --platform linux/arm64 \
  --target debug \
  --build-arg ONNX_MODEL_PATH=export/artifacts/mambavision_t_1k.onnx \
  -t mambavision-ros:latest \
  -f docker/Dockerfile .

else 
docker build \
--target debug \
--build-arg ONNX_MODEL_PATH=export/artifacts/mambavision_t_1k.onnx \
-t mambavision-ros:latest \
-f docker/Dockerfile .
