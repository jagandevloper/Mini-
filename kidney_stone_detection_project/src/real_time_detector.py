#!/usr/bin/env python3
"""
Real-Time Kidney Stone Detection with CUDA
==========================================

This script provides real-time kidney stone detection using your trained YOLOv8 model
with CUDA acceleration. Perfect for medical imaging applications.

Author: AI Assistant
Date: 2024
"""

import cv2
import torch
import numpy as np
from pathlib import Path
import time
from ultralytics import YOLO
import argparse

class RealTimeKidneyStoneDetector:
    """Real-time kidney stone detection using CUDA-accelerated YOLOv8."""
    
    def __init__(self, model_path: str, device: str = 'cuda', conf_threshold: float = 0.25):
        """
        Initialize the detector.
        
        Args:
            model_path: Path to trained YOLOv8 model
            device: Device to use ('cuda' or 'cpu')
            conf_threshold: Confidence threshold for detections
        """
        self.device = device
        self.conf_threshold = conf_threshold
        
        # Load model
        print(f"Loading model from: {model_path}")
        self.model = YOLO(model_path)
        self.model.to(device)
        
        # Check CUDA availability
        if device == 'cuda' and torch.cuda.is_available():
            print(f"✅ CUDA available! Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ Using CPU (CUDA not available)")
        
        print(f"Model loaded successfully on {device}")
    
    def detect_from_image(self, image_path: str, save_result: bool = True):
        """
        Detect kidney stones in a single image.
        
        Args:
            image_path: Path to input image
            save_result: Whether to save result image
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return None
        
        # Run detection
        start_time = time.time()
        results = self.model(image, conf=self.conf_threshold, device=self.device)
        inference_time = time.time() - start_time
        
        # Process results
        result = results[0]
        detections = result.boxes
        
        print(f"\n📊 Detection Results for: {Path(image_path).name}")
        print(f"⏱️ Inference Time: {inference_time*1000:.1f}ms")
        
        if detections is not None and len(detections) > 0:
            print(f"🔍 Found {len(detections)} kidney stone(s):")
            for i, detection in enumerate(detections):
                conf = detection.conf.item()
                print(f"   Stone {i+1}: Confidence = {conf:.3f}")
        else:
            print("✅ No kidney stones detected")
        
        # Save result if requested
        if save_result:
            output_path = f"detection_result_{Path(image_path).stem}.jpg"
            result.save(output_path)
            print(f"💾 Result saved to: {output_path}")
        
        return result
    
    def detect_from_camera(self, camera_id: int = 0):
        """
        Real-time detection from camera feed.
        
        Args:
            camera_id: Camera device ID
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        print("🎥 Starting real-time detection...")
        print("Press 'q' to quit, 's' to save current frame")
        
        frame_count = 0
        fps_counter = 0
        fps_start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run detection every 5 frames for performance
            if frame_count % 5 == 0:
                start_time = time.time()
                results = self.model(frame, conf=self.conf_threshold, device=self.device)
                inference_time = time.time() - start_time
                
                # Draw results
                annotated_frame = results[0].plot()
                
                # Calculate FPS
                fps_counter += 1
                if fps_counter % 30 == 0:
                    fps = 30 / (time.time() - fps_start_time)
                    fps_start_time = time.time()
                    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"Inference: {inference_time*1000:.1f}ms", 
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Kidney Stone Detection', annotated_frame)
            else:
                cv2.imshow('Kidney Stone Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                filename = f"captured_frame_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Frame saved: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def batch_detect(self, input_folder: str, output_folder: str = "detection_results"):
        """
        Batch detection on multiple images.
        
        Args:
            input_folder: Folder containing input images
            output_folder: Folder to save results
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        print(f"🔍 Processing {len(image_files)} images...")
        
        total_time = 0
        detections_count = 0
        
        for i, image_file in enumerate(image_files):
            print(f"\nProcessing {i+1}/{len(image_files)}: {image_file.name}")
            
            # Run detection
            start_time = time.time()
            results = self.model(str(image_file), conf=self.conf_threshold, device=self.device)
            inference_time = time.time() - start_time
            total_time += inference_time
            
            # Count detections
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                detections_count += len(result.boxes)
                print(f"   Found {len(result.boxes)} stone(s) in {inference_time*1000:.1f}ms")
            else:
                print(f"   No stones detected in {inference_time*1000:.1f}ms")
            
            # Save result
            output_file = output_path / f"detected_{image_file.name}"
            result.save(str(output_file))
        
        # Summary
        avg_time = total_time / len(image_files) * 1000
        print(f"\n📊 Batch Processing Summary:")
        print(f"   Total images: {len(image_files)}")
        print(f"   Total detections: {detections_count}")
        print(f"   Average time per image: {avg_time:.1f}ms")
        print(f"   Results saved to: {output_path}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Real-Time Kidney Stone Detection')
    parser.add_argument('--model', type=str, 
                       default='runs/kidney_stone_cuda_test/weights/best.pt',
                       help='Path to trained model')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--mode', type=str, default='image',
                       choices=['image', 'camera', 'batch'],
                       help='Detection mode')
    parser.add_argument('--input', type=str, help='Input image/folder path')
    parser.add_argument('--output', type=str, default='detection_results',
                       help='Output folder for batch mode')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = RealTimeKidneyStoneDetector(
        model_path=args.model,
        device=args.device,
        conf_threshold=args.conf
    )
    
    # Run detection based on mode
    if args.mode == 'image':
        if not args.input:
            print("Error: --input required for image mode")
            return
        detector.detect_from_image(args.input)
    
    elif args.mode == 'camera':
        detector.detect_from_camera()
    
    elif args.mode == 'batch':
        if not args.input:
            print("Error: --input required for batch mode")
            return
        detector.batch_detect(args.input, args.output)


if __name__ == "__main__":
    main()

