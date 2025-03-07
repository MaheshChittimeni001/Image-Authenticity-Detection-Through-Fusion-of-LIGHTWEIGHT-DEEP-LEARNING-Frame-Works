# Image-Authenticity-Detection-Through-Fusion-of-LIGHTWEIGHT-DEEP-LEARNING-Frame-Works

Project Overview
Detects fake, manipulated, or deepfake images using lightweight deep learning models.
Aims to achieve high accuracy with minimal computational cost, making it suitable for mobile and edge devices.
Uses a fusion approach by combining multiple deep learning models to improve detection performance.

Methodology
Dataset: Uses benchmark datasets like FaceForensics++, DFDC, and CASIA for training.
Lightweight Models: Implements MobileNetV3, EfficientNet-Lite, XceptionNet, and SqueezeNet for detection.
Model Fusion: Combines predictions using ensemble learning (weighted averaging, feature fusion, stacking).
Optimization: Uses quantization, pruning, and transfer learning to improve efficiency.
Classification Output: Labels images as Authentic, Manipulated, or Deepfake.

Implementation Details
Tech Stack: Python, TensorFlow, PyTorch, OpenCV, Scikit-Learn.
Deployment: Converts models to ONNX/TFLite for real-time use on edge devices.

Applications
Fake news detection – Identifies manipulated images in media.
Cybersecurity – Prevents identity fraud through deepfake detection.
Social media monitoring – Detects forged images on platforms.
Forensic analysis – Assists law enforcement in verifying digital evidence.

Key Benefits
High accuracy with low computational cost.
Real-time detection capability.
Deployable on mobile and edge devices.
