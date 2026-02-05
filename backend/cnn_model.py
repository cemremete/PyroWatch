"""
PyroWatch CNN Model for Fire Detection
Uses MobileNetV2 pre-trained model fine-tuned for fire detection from satellite imagery.

Model Architecture:
- Base: MobileNetV2 (ImageNet pretrained)
- Custom classifier head for binary classification (fire/no-fire)
- Input: RGB images (224x224)
- Output: Fire probability (0-1)

Usage:
    from model.cnn_model import FireDetectionModel
    
    model = FireDetectionModel()
    model.load_weights('fire_detector.pth')
    
    # Predict on image
    probability = model.predict(image_tensor)
"""

import torch, torch.nn as nn, torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class FireDetectionModel(nn.Module):
    """
    Fire detection CNN using MobileNetV2 backbone.
    Target accuracy: 91%+
    """
    
    def __init__(self, pretrained: bool = True, num_classes: int = 2):
        super(FireDetectionModel, self).__init__()
        
        # Load MobileNetV2 pretrained on ImageNet
        self.backbone = models.mobilenet_v2(pretrained=pretrained)
        
        # Get the number of features from the last layer
        num_features = self.backbone.classifier[1].in_features
        
        # Replace classifier with custom head for fire detection
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(128, num_classes)
        )
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def forward(self, x):
        # Forward pass through the model
        return self.backbone(x)
    
    def predict(self, image: Image.Image) -> Tuple[float, str]:
        """
        Predict fire probability on a single image.
        
        Args:
            image: PIL Image
            
        Returns:
            Tuple of (probability, class_label)
        """
        self.eval()
        
        with torch.no_grad():
            # Preprocess image
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            tensor = self.transform(image).unsqueeze(0)
            
            # Get prediction
            outputs = self(tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get fire probability (assuming class 1 = fire)
            fire_prob = probabilities[0][1].item()
            confidence = fire_prob * 100
            
            # Class label based on threshold
            label = 'fire' if fire_prob > 0.5 else 'no_fire'
            
            return confidence, label
    
    def predict_batch(self, images: list) -> list:
        """
        Predict on multiple images at once.
        More efficient for batch processing.
        """
        self.eval()
        results = []
        
        with torch.no_grad():
            for img in images:
                try:
                    conf, label = self.predict(img)
                    results.append({
                        'confidence': conf,
                        'prediction': label
                    })
                except Exception as e:
                    logger.warning(f"Failed to process image: {e}")
                    results.append({
                        'confidence': 0.0,
                        'prediction': 'error'
                    })
        
        return results
    
    def load_weights(self, path: str):
        """Load model weights from file"""
        try:
            self.load_state_dict(torch.load(path, map_location='cpu'))
            logger.info(f"Loaded model weights from {path}")
        except Exception as e:
            logger.error(f"Failed to load weights: {e}")
            raise
    
    def save_weights(self, path: str):
        """Save model weights to file"""
        try:
            torch.save(self.state_dict(), path)
            logger.info(f"Saved model weights to {path}")
        except Exception as e:
            logger.error(f"Failed to save weights: {e}")
            raise


class FireInference:
    """
    High-level interface for fire detection inference.
    Handles image loading and preprocessing automatically.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = FireDetectionModel()
        if model_path:
            self.model.load_weights(model_path)
    
    def process_satellite_data(self, image_path: str) -> dict:
        """
        Process satellite image for fire detection.
        
        Args:
            image_path: Path to satellite image file
            
        Returns:
            Dictionary with detection results
        """
        try:
            # Load image
            image = Image.open(image_path)
            
            # Get prediction
            confidence, prediction = self.model.predict(image)
            
            # Additional metadata
            result = {
                'image_path': image_path,
                'fire_detected': prediction == 'fire',
                'confidence': confidence,
                'prediction': prediction,
                'timestamp': torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'
            }
            
            # Log high-confidence detections
            if confidence > 80:
                logger.info(f"High confidence fire detected: {confidence:.1f}%")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing satellite data: {e}")
            return {
                'image_path': image_path,
                'fire_detected': False,
                'confidence': 0.0,
                'prediction': 'error',
                'error': str(e)
            }
    
    def batch_process_directory(self, directory: str, image_extensions: list = None) -> list:
        """
        Process all images in a directory.
        
        Args:
            directory: Path to directory containing images
            image_extensions: List of allowed extensions (default: common image formats)
            
        Returns:
            List of detection results
        """
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        import os
        results = []
        
        # Find all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend([
                os.path.join(directory, f) 
                for f in os.listdir(directory) 
                if f.lower().endswith(ext.lower())
            ])
        
        print(f"Found {len(image_files)} images to process")
        
        # Process each image
        for img_path in image_files:
            result = self.process_satellite_data(img_path)
            results.append(result)
        
        # Summary statistics
        fire_count = sum(1 for r in results if r['fire_detected'])
        avg_confidence = sum(r['confidence'] for r in results) / len(results) if results else 0
        
        summary = {
            'total_images': len(results),
            'fires_detected': fire_count,
            'avg_confidence': avg_confidence,
            'results': results
        }
        
        logger.info(f"Processed {len(results)} images, {fire_count} fires detected")
        return summary


# Example usage and testing
if __name__ == "__main__":
    # Create inference engine
    inference = FireInference()
    
    # Test with sample data (if available)
    sample_data = {
        'image_path': 'test_satellite.jpg',
        'coordinates': (37.7749, -122.4194),  # San Francisco
        'timestamp': '2026-02-05T14:00:00Z'
    }
    
    print("Fire Detection Model Test")
    print("=" * 40)
    
    # Simulate processing
    result = inference.process_satellite_data('test_image.jpg')
    print(f"Test result: {result}")
    
    # Batch processing example
    # results = inference.batch_process_directory('./test_images/')
    # print(f"Batch results: {results['summary']}")
