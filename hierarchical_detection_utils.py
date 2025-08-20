#!/usr/bin/env python3
# hierarchical_detection_utils.py - Hierarchical Detection and Visualization Utils

import cv2
import numpy as np
import yaml
import json
from pathlib import Path

class HierarchicalDetectionVisualizer:
    """
    Hierarchical detection visualization for agricultural multi-task model
    Displays main category + specific type with color coding
    """
    
    def __init__(self, config_file="config_datasets.yaml"):
        self.config_file = config_file
        self.config = self._load_config()
        self.class_mapping = self.config.get('class_mapping', {})
        self.detection_settings = self.config.get('detection_settings', {})
        
        # Initialize detection mappings
        self.main_to_specific = self._build_specific_mappings()
        self.bbox_colors = self._build_color_mappings()
        
        print(f"✅ HierarchicalDetectionVisualizer initialized")
        print(f"📊 Main categories: {len(self.class_mapping)}")
        print(f"🎨 Visualization ready for hierarchical detection")
    
    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"⚠️  Config file not found: {self.config_file}")
            return {}
    
    def _build_specific_mappings(self):
        """Build mapping from specific classes to Turkish names"""
        mappings = {}
        
        for main_class, info in self.class_mapping.items():
            # Get specific mappings if available
            specific_items = info.get('specific_pests', {})
            specific_items.update(info.get('specific_deficiencies', {}))
            
            for specific_class, turkish_name in specific_items.items():
                mappings[specific_class.lower()] = {
                    'main_category': main_class,
                    'turkish_name': turkish_name,
                    'detection_label': info.get('detection_label', main_class.upper())
                }
        
        return mappings
    
    def _build_color_mappings(self):
        """Build color mappings for bounding boxes"""
        colors = {}
        
        for main_class, info in self.class_mapping.items():
            bbox_color = info.get('bbox_color', [128, 128, 128])  # Default gray
            colors[main_class] = tuple(bbox_color)
        
        return colors
    
    def get_hierarchical_label(self, class_name, confidence=None):
        """
        Get hierarchical label for detection
        Returns: (main_label, specific_label, full_label, color)
        """
        class_name_lower = class_name.lower()
        
        # Find main category
        main_category = None
        for main_class, info in self.class_mapping.items():
            sub_classes = [sc.lower() for sc in info.get('sub_classes', [])]
            if class_name_lower in sub_classes:
                main_category = main_class
                break
        
        if not main_category:
            # Default to unknown
            main_category = 'unknown'
        
        # Get category info
        category_info = self.class_mapping.get(main_category, {})
        main_label = category_info.get('detection_label', main_category.upper())
        color = self.bbox_colors.get(main_category, (128, 128, 128))
        
        # Get specific type if available
        specific_info = self.main_to_specific.get(class_name_lower, {})
        specific_label = specific_info.get('turkish_name', class_name)
        
        # Build full label
        confidence_text = f" ({confidence:.2f})" if confidence else ""
        
        if self.detection_settings.get('show_both', True):
            full_label = f"{main_label}: {specific_label}{confidence_text}"
        elif self.detection_settings.get('show_specific_type', True):
            full_label = f"{specific_label}{confidence_text}"
        else:
            full_label = f"{main_label}{confidence_text}"
        
        return main_label, specific_label, full_label, color
    
    def draw_hierarchical_detection(self, image, bbox, class_name, confidence=None):
        """
        Draw hierarchical detection on image
        
        Args:
            image: OpenCV image (BGR)
            bbox: [x1, y1, x2, y2] in pixel coordinates
            class_name: Detected class name
            confidence: Detection confidence (0-1)
        
        Returns:
            Modified image with hierarchical detection visualization
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Get hierarchical information
        main_label, specific_label, full_label, color = self.get_hierarchical_label(
            class_name, confidence
        )
        
        # Get visualization settings
        bbox_thickness = self.detection_settings.get('bbox_thickness', 3)
        text_thickness = self.detection_settings.get('text_thickness', 2)
        text_scale = self.detection_settings.get('text_scale', 0.8)
        text_color = tuple(self.detection_settings.get('text_color', [255, 255, 255]))
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, bbox_thickness)
        
        # Prepare text
        (text_width, text_height), baseline = cv2.getTextSize(
            full_label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness
        )
        
        # Text background
        if self.detection_settings.get('text_background', True):
            text_bg_alpha = self.detection_settings.get('text_background_alpha', 0.7)
            
            # Create text background
            text_bg = np.zeros_like(image[y1-text_height-10:y1, x1:x1+text_width+10])
            text_bg[:] = color
            
            # Blend with original image
            if y1-text_height-10 >= 0 and x1+text_width+10 <= image.shape[1]:
                roi = image[y1-text_height-10:y1, x1:x1+text_width+10]
                cv2.addWeighted(roi, 1-text_bg_alpha, text_bg, text_bg_alpha, 0, roi)
        
        # Draw text
        cv2.putText(image, full_label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                   text_scale, text_color, text_thickness)
        
        return image
    
    def process_yolo_results(self, image, results):
        """
        Process YOLO detection results and apply hierarchical visualization
        
        Args:
            image: Original image (BGR)
            results: YOLO detection results
            
        Returns:
            Annotated image with hierarchical detections
        """
        annotated_image = image.copy()
        
        if hasattr(results, 'boxes') and results.boxes is not None:
            boxes = results.boxes
            
            for i in range(len(boxes)):
                # Get detection info
                bbox = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                confidence = boxes.conf[i].cpu().numpy()
                class_id = int(boxes.cls[i].cpu().numpy())
                
                # Get class name (assumes you have class names list)
                if hasattr(results, 'names'):
                    class_name = results.names[class_id]
                else:
                    class_name = f"class_{class_id}"
                
                # Apply hierarchical visualization
                annotated_image = self.draw_hierarchical_detection(
                    annotated_image, bbox, class_name, confidence
                )
        
        return annotated_image
    
    def get_detection_summary(self, results):
        """
        Get detection summary with hierarchical information
        
        Returns:
            List of detection dictionaries with hierarchical info
        """
        detections = []
        
        if hasattr(results, 'boxes') and results.boxes is not None:
            boxes = results.boxes
            
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy()
                confidence = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                
                if hasattr(results, 'names'):
                    class_name = results.names[class_id]
                else:
                    class_name = f"class_{class_id}"
                
                # Get hierarchical info
                main_label, specific_label, full_label, color = self.get_hierarchical_label(
                    class_name, confidence
                )
                
                detection = {
                    'bbox': bbox.tolist(),
                    'confidence': confidence,
                    'original_class': class_name,
                    'main_category': main_label,
                    'specific_type': specific_label,
                    'full_label': full_label,
                    'color': color
                }
                
                detections.append(detection)
        
        return detections
    
    def save_detection_results(self, detections, output_file="detection_results.json"):
        """Save detection results to JSON file"""
        results_data = {
            'timestamp': np.datetime64('now').isoformat(),
            'total_detections': len(detections),
            'detections': detections,
            'categories_detected': list(set(d['main_category'] for d in detections)),
            'config_file': self.config_file
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Detection results saved to: {output_file}")
    
    def create_detection_report(self, detections):
        """Create a formatted detection report"""
        if not detections:
            return "No detections found."
        
        # Group by main category
        category_groups = {}
        for detection in detections:
            main_cat = detection['main_category']
            if main_cat not in category_groups:
                category_groups[main_cat] = []
            category_groups[main_cat].append(detection)
        
        # Build report
        report = f"🌱 Tarımsal Tespit Raporu\n"
        report += f"=" * 50 + "\n"
        report += f"📊 Toplam Tespit: {len(detections)}\n"
        report += f"🏷️  Ana Kategoriler: {len(category_groups)}\n\n"
        
        for main_category, group_detections in category_groups.items():
            report += f"📂 {main_category} ({len(group_detections)} tespit)\n"
            report += f"-" * 30 + "\n"
            
            for detection in group_detections:
                specific = detection['specific_type']
                confidence = detection['confidence']
                report += f"  • {specific} (Güven: {confidence:.2f})\n"
            
            report += "\n"
        
        return report

# Example usage and testing
if __name__ == "__main__":
    # Initialize visualizer
    visualizer = HierarchicalDetectionVisualizer("config_datasets.yaml")
    
    # Test hierarchical labeling
    test_classes = [
        "Spider Mite",
        "spider mite", 
        "Aphid",
        "nitrogen-deficiency",
        "Apple Scab Leaf",
        "healthy"
    ]
    
    print("\n🧪 Testing Hierarchical Labels:")
    for class_name in test_classes:
        main, specific, full, color = visualizer.get_hierarchical_label(class_name, 0.85)
        print(f"'{class_name}' → {full} {color}")
    
    print("\n✅ Hierarchical detection system ready!")
    print("🎯 Example output: 'ZARLI: Kırmızı Örümcek (0.85)'")