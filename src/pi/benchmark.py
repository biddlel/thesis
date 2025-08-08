#!/usr/bin/env python3
"""
YOLOv11s ONNX Benchmark Script
-----------------------------
Measures the performance of YOLOv11s ONNX model on test images.
"""

import os
import time
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO
import torch

def load_test_images(image_dir, num_images=100):
    """Load test images from directory."""
    image_paths = sorted(Path(image_dir).glob('*.jpg'))[:num_images]
    if not image_paths:
        raise FileNotFoundError(f"No JPG images found in {image_dir}")
    
    images = []
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is not None:
            images.append(img)
    
    if not images:
        raise ValueError("No valid images could be loaded")
        
    return images

def benchmark_model(model_path, test_images, warmup=3, runs=50):
    """Benchmark the YOLO model."""
    # Load model
    model = YOLO(model_path, task="detect")
    
    # Warmup
    print("Warming up...")
    for _ in range(warmup):
        _ = model(test_images[0], verbose=False)
    
    # Benchmark
    print(f"Running benchmark with {len(test_images)} images for {runs} runs...")
    times = []
    detections = []
    
    for _ in range(runs):
        for img in test_images:
            start_time = time.perf_counter()
            results = model(img, verbose=False)
            end_time = time.perf_counter()
            
            times.append((end_time - start_time) * 1000)  # Convert to ms
            detections.append(len(results[0].boxes))
    
    # Calculate statistics
    times = np.array(times)
    detections = np.array(detections)
    
    stats = {
        'avg_inference_time_ms': np.mean(times),
        'std_inference_time_ms': np.std(times),
        'min_inference_time_ms': np.min(times),
        'max_inference_time_ms': np.max(times),
        'fps': 1000 / np.mean(times),
        'avg_detections': np.mean(detections),
        'total_runs': len(times),
        'device': str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    }
    
    return stats

def print_benchmark_results(stats):
    """Print benchmark results in a readable format."""
    print("\n" + "="*50)
    print("YOLOv11m ONNX Benchmark Results")
    print("="*50)
    print(f"Device: {stats['device']}")
    print(f"Total runs: {stats['total_runs']}")
    print("\nInference Time (ms):")
    print(f"  Average: {stats['avg_inference_time_ms']:.2f} ± {stats['std_inference_time_ms']:.2f}")
    print(f"  Min: {stats['min_inference_time_ms']:.2f}")
    print(f"  Max: {stats['max_inference_time_ms']:.2f}")
    print(f"\nThroughput: {stats['fps']:.2f} FPS")
    print(f"Average detections per frame: {stats['avg_detections']:.2f}")
    print("="*50 + "\n")

def main():
    # Configuration
    MODEL_PATH = "models/yolo11s_ncnn_model"  # Update this path if needed
    TEST_IMAGE_DIR = "test_images"  # Directory containing test images
    
    try:
        # Create test_images directory if it doesn't exist
        os.makedirs(TEST_IMAGE_DIR, exist_ok=True)
        
        # Check if test images exist, if not, create a sample image
        if not any(Path(TEST_IMAGE_DIR).glob('*.jpg')):
            print(f"No test images found in {TEST_IMAGE_DIR}, creating a sample image...")
            sample_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(TEST_IMAGE_DIR, "sample.jpg"), sample_img)
        
        # Load test images
        print("Loading test images...")
        test_images = load_test_images(TEST_IMAGE_DIR)
        print(f"Loaded {len(test_images)} test images")
        
        # Run benchmark
        print("Starting benchmark...")
        stats = benchmark_model(MODEL_PATH, test_images)
        
        # Print results
        print_benchmark_results(stats)
        
    except Exception as e:
        print(f"Error during benchmark: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
