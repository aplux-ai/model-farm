# APLUX Model Farm

## Overview

To accelerate evaluation AI model performance on target edge devices, APLUX builds the [Model Farm](https://aiot.aidlux.com/en/models). Model Farm contains hundreds of mainstream open-source models with different functions, optimized for different hardware platforms, and provides benchmark performance reference based on real testing. Developers can quickly finish evaluations according to their actual requirements without investing substantial costs and time costs.

At the same time, Model Farm also provides ready-to-run model inference example code, greatly reducing the difficulty and workload for developers to test model performance and develop AI application on edge devices, shortening the entire process time and accelerating solution deployment.

## Features

Specifically, Model Farm can help developers accomplish the following:

- Query AI model performance reference on specific Qualcomm chip
- Download optimized AI models (leveraging NPU for acceleration inference)
- Download pre/post-processing and model inference example code
- View model conversion & optimization steps, which help developers to optimize their own fine-tuned models quickly

**Models on Model Farm**

![](./imgs/model-farm-test-flowchat-en.png)

**Fine-tuned Models by User**

![](./imgs/model-farm-test-finetune-flowchat-en.png)

## Support Matrix

| QCS8550  | QCS6490 | QCS8625 | IQ9 |IQ8 |IQ10 |
| :---: | :---: | :---: | :---: |:---: |:---: |
|  ✔  |  ✔  |   ✔   | ✔  |✔  |✔  |

## New Model Request

Please submit adaptation requests for new models via GitHub Issues. We will collect these requests and regularly select popular models for adaptation.

## Recent updates

📣 **2026.7.23**
**GenAI**
- support Hy-MT2-1.8B
- support LocateAnything-3B (support MTP mode)
- support MiniCPM5-1B
**Robot**
- support BEVFormer-Tiny
- support GR00T-1.5
- Optimize π₀.₅
**General**
- support PP-OCRv6

📣 **2026.2.4**
- support MMS-TTS (Preview Section)

📣 **2026.1.16**

**GenAI**
- Support Qwen3-VL-4B (Preview Section)
- Support Qwen3-0.6B (Preview Section)
- Support Qwen3-1.7B (Preview Section)
- Support Qwen3-8B (Preview Section)

**Robot**
- Support FoundationPose (Preview Section)
- Support BEVFusion

**General**
- Support YOLOv11 Pose
- Support HRNetFace
- Support YOLO-R
- Support Detectron2
- Support DeepFilterNet (Preview Section)
- Support EdgeTAM
  
📣 **2025.10.10**
- Support Falcon3-7B-Instruct (Preview Section)

📣 **2025.09.30**
- Support Qwen3-4B (Preview Section)
- Support π0 (Preview Section)

📣 **2025.09.16**
- [Model Farm Preview](https://aiot.aidlux.com/en/models/preview) Section is Now Live

## Resources

- [Model Farm](https://aiot.aidlux.com/en/models)
- [Model Farm User Guide](https://rhinopi.docs.aidlux.com/en/software/model-farm/model_farm_guide)
- [AI Model Optimizer (AIMO)](https://rhinopi.docs.aidlux.com/en/software/aimo/aimo_guide)
- [Tutorials](https://rhinopi.docs.aidlux.com/en/software/tutorial/)

## Contact Us

- [contact APLUX](mailto:liuweibin@aidlux.com?cc=huangwenbo@aidlux.com) 
