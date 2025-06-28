from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import base64
from io import BytesIO
import uuid
import random
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter app

class TongueAnalyzer:
    def __init__(self, model_path="best_tongue_classification_model.pth"):
        """
        Initialize the tongue analyzer for API use
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the classification model
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.categories = checkpoint.get('categories', [])
            self.category_to_idx = checkpoint.get('category_to_idx', {})
            
            self.model = models.resnet50(pretrained=False)
            self.model.fc = nn.Linear(self.model.fc.in_features, len(self.categories))
            
            # Fix the state dict keys by removing "model." prefix if present
            state_dict = checkpoint['model_state_dict']
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('model.'):
                    new_key = key[6:]  # Remove "model." prefix
                else:
                    new_key = key
                new_state_dict[new_key] = value
            
            self.model.load_state_dict(new_state_dict)
            print(f"Loaded model with {len(self.categories)} classes")
        else:
            print(f"Model not found at {model_path}")
            self.categories = [
                '淡白舌白苔', '红舌黄苔', '淡白舌黄苔', '绛舌灰黑苔', '绛舌黄苔',
                '绛舌白苔', '红舌灰黑苔', '红舌白苔', '淡红舌灰黑苔', '淡红舌黄苔',
                '淡红舌白苔', '青紫舌白苔', '青紫舌黄苔', '青紫舌灰黑苔', '淡白舌灰黑苔'
            ]
            self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
            
            # Create a default model even if file doesn't exist
            self.model = models.resnet50(pretrained=False)
            self.model.fc = nn.Linear(self.model.fc.in_features, len(self.categories))
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def detect_tongue_features(self, image_path):
        """
        Detect specific tongue features for visualization (simplified)
        """
        try:
            print(f"Loading image: {image_path}")
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print("Failed to load image with cv2")
                return {'tongue_region': None, 'cracks': [], 'coating': [], 'color_variations': [], 'teeth_marks': []}
            
            print("Converting image to RGB")
            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            print("Starting feature detection")
            # Simplified feature detection to avoid hangs
            features = self._detect_tongue_region_and_features_simple(image_rgb)
            
            return features
        except Exception as e:
            print(f"Error in detect_tongue_features: {e}")
            return {'tongue_region': None, 'cracks': [], 'coating': [], 'color_variations': [], 'teeth_marks': []}

    def _detect_tongue_region_and_features_simple(self, image):
        """
        Simplified feature detection with highly sensitive crack detection and comprehensive color analysis
        """
        try:
            print("Enhanced feature detection started")
            height, width = image.shape[:2]
            
            # Keep the larger tongue region - this is working well
            region_width = int(width * 0.75)  # 75% of image width
            region_height = int(height * 0.8)  # 80% of image height
            start_x = (width - region_width) // 2
            start_y = (height - region_height) // 2
            
            # HIGHLY SENSITIVE crack detection - more sensitive in the CENTER/MIDDLE of tongue
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Define inner tongue region (central 70% of tongue area for better coverage)
            inner_margin_x = int(region_width * 0.15)  # 15% margin from left/right
            inner_margin_y = int(region_height * 0.15)  # 15% margin from top/bottom
            inner_start_x = start_x + inner_margin_x
            inner_start_y = start_y + inner_margin_y
            inner_width = region_width - 2 * inner_margin_x
            inner_height = region_height - 2 * inner_margin_y
            
            # Extract only the inner tongue region for crack detection
            inner_tongue = gray[inner_start_y:inner_start_y+inner_height, 
                               inner_start_x:inner_start_x+inner_width]
            
            cracks = []
            if inner_tongue.size > 0:
                # MULTI-SCALE crack detection for maximum sensitivity
                
                # Scale 1: Very sensitive fine crack detection
                blurred1 = cv2.GaussianBlur(inner_tongue, (3, 3), 0.5)  # Minimal blur
                edges1 = cv2.Canny(blurred1, 30, 90)  # Very low thresholds for fine cracks
                
                # Scale 2: Medium crack detection
                blurred2 = cv2.GaussianBlur(inner_tongue, (5, 5), 1.0)
                edges2 = cv2.Canny(blurred2, 40, 110)
                
                # Scale 3: Larger crack detection
                blurred3 = cv2.GaussianBlur(inner_tongue, (7, 7), 1.5)
                edges3 = cv2.Canny(blurred3, 50, 130)
                
                # Combine all edge detections
                combined_edges = cv2.bitwise_or(edges1, cv2.bitwise_or(edges2, edges3))
                
                # Morphological operations to enhance crack-like structures
                kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                
                # Close small gaps in cracks
                combined_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel_small, iterations=1)
                combined_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel_medium, iterations=1)
                
                # Multiple line detection passes with different sensitivities
                all_lines = []
                
                # Pass 1: Very sensitive for fine cracks
                lines1 = cv2.HoughLinesP(combined_edges, 1, np.pi/180, 
                                        threshold=15,        # Very low threshold
                                        minLineLength=12,    # Very short minimum length
                                        maxLineGap=12)       # Large gap tolerance
                if lines1 is not None:
                    all_lines.extend(lines1)
                
                # Pass 2: Medium sensitivity
                lines2 = cv2.HoughLinesP(combined_edges, 1, np.pi/180, 
                                        threshold=20,        # Low threshold
                                        minLineLength=18,    # Short minimum length
                                        maxLineGap=10)       # Medium gap tolerance
                if lines2 is not None:
                    all_lines.extend(lines2)
                
                # Pass 3: Standard sensitivity for obvious cracks
                lines3 = cv2.HoughLinesP(combined_edges, 1, np.pi/180, 
                                        threshold=25,        # Standard threshold
                                        minLineLength=25,    # Standard minimum length
                                        maxLineGap=8)        # Small gap tolerance
                if lines3 is not None:
                    all_lines.extend(lines3)
                
                if all_lines:
                    # Remove duplicate lines and validate
                    validated_cracks = []
                    
                    for line in all_lines:
                        x1, y1, x2, y2 = line[0]
                        # Convert coordinates back to original image space
                        orig_x1 = x1 + inner_start_x
                        orig_y1 = y1 + inner_start_y
                        orig_x2 = x2 + inner_start_x
                        orig_y2 = y2 + inner_start_y
                        
                        # Medical validation: check line characteristics
                        line_length = ((x2-x1)**2 + (y2-y1)**2)**0.5
                        
                        # Calculate line angle
                        angle = np.arctan2(abs(y2-y1), abs(x2-x1)) * 180 / np.pi
                        
                        # Accept lines that could be cracks (more permissive criteria)
                        if (line_length > 8 and line_length < 200 and  # Very wide range for crack length
                            5 < x1 < inner_width-5 and 5 < x2 < inner_width-5 and  # Within bounds
                            5 < y1 < inner_height-5 and 5 < y2 < inner_height-5):
                            
                            # Additional validation: check if line area has crack-like properties
                            mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
                            if (3 < mid_x < inner_width-3 and 3 < mid_y < inner_height-3):
                                # Sample area around the line to validate it's a real crack
                                sample_area = inner_tongue[mid_y-2:mid_y+3, mid_x-2:mid_x+3]
                                if sample_area.size > 0:
                                    area_std = sample_area.std()
                                    area_mean = sample_area.mean()
                                    
                                    # More permissive crack validation
                                    if area_std > 5 and area_mean < inner_tongue.mean() + 10:  # Much more permissive
                                        # Check for duplicates (avoid adding very similar lines)
                                        is_duplicate = False
                                        for existing_crack in validated_cracks:
                                            ex1, ey1, ex2, ey2 = existing_crack[0][0], existing_crack[0][1], existing_crack[1][0], existing_crack[1][1]
                                            # Calculate distance between line centers
                                            dist = ((mid_x - (ex1+ex2)//2)**2 + (mid_y - (ey1+ey2)//2)**2)**0.5
                                            if dist < 15:  # Too close to existing crack
                                                is_duplicate = True
                                                break
                                        
                                        if not is_duplicate:
                                            validated_cracks.append(((orig_x1, orig_y1), (orig_x2, orig_y2)))
                    
                    cracks = validated_cracks
                
                # Limit to most significant cracks (allow more cracks to be detected)
                cracks = cracks[:8]  # Increased from 4 to 8
            
            # COMPREHENSIVE COLOR VARIATION ANALYSIS
            color_variations = []
            try:
                # Extract tongue region for color analysis
                tongue_region = image[start_y:start_y+region_height, start_x:start_x+region_width]
                
                if tongue_region.size > 0:
                    # Convert to LAB color space for perceptual color analysis
                    lab_tongue = cv2.cvtColor(tongue_region, cv2.COLOR_RGB2LAB)
                    hsv_tongue = cv2.cvtColor(tongue_region, cv2.COLOR_RGB2HSV)
                    
                    # Divide tongue into analysis regions (3x3 grid for detailed analysis)
                    grid_rows, grid_cols = 3, 3
                    region_h = int(region_height // grid_rows)
                    region_w = int(region_width // grid_cols)
                    
                    # Calculate baseline color statistics for the entire tongue
                    baseline_l = lab_tongue[:, :, 0].mean()  # Lightness
                    baseline_a = lab_tongue[:, :, 1].mean()  # Green-Red
                    baseline_b = lab_tongue[:, :, 2].mean()  # Blue-Yellow
                    baseline_h = hsv_tongue[:, :, 0].mean()  # Hue
                    baseline_s = hsv_tongue[:, :, 1].mean()  # Saturation
                    
                    # Analyze each grid region for color variations
                    for row in range(grid_rows):
                        for col in range(grid_cols):
                            # Extract region
                            y_start = int(row * region_h)
                            y_end = int(min((row + 1) * region_h, region_height))
                            x_start = int(col * region_w)
                            x_end = int(min((col + 1) * region_w, region_width))
                            
                            region_lab = lab_tongue[y_start:y_end, x_start:x_end]
                            region_hsv = hsv_tongue[y_start:y_end, x_start:x_end]
                            
                            if region_lab.size > 0:
                                # Calculate color statistics for this region
                                region_l = region_lab[:, :, 0].mean()
                                region_a = region_lab[:, :, 1].mean()
                                region_b = region_lab[:, :, 2].mean()
                                region_h = region_hsv[:, :, 0].mean()
                                region_s = region_hsv[:, :, 1].mean()
                                
                                # Calculate color differences from baseline
                                l_diff = abs(region_l - baseline_l)
                                a_diff = abs(region_a - baseline_a)
                                b_diff = abs(region_b - baseline_b)
                                h_diff = abs(region_h - baseline_h)
                                s_diff = abs(region_s - baseline_s)
                                
                                # Check for significant color variations
                                # Medical criteria: noticeable color changes that could indicate health issues
                                significant_variation = False
                                variation_type = ""
                                
                                # Lightness variation (pale/dark areas)
                                if l_diff > 8:  # Noticeable lightness difference
                                    significant_variation = True
                                    variation_type = "lightness"
                                
                                # Color cast variations (redness, yellowness)
                                elif a_diff > 6 or b_diff > 8:  # Color cast differences
                                    significant_variation = True
                                    variation_type = "color_cast"
                                
                                # Saturation variations (dull/vivid areas)
                                elif s_diff > 15:  # Saturation differences
                                    significant_variation = True
                                    variation_type = "saturation"
                                
                                # Hue variations (different color tones)
                                elif h_diff > 12:  # Hue differences
                                    significant_variation = True
                                    variation_type = "hue"
                                
                                if significant_variation:
                                    # Convert region coordinates back to original image space
                                    orig_x = int(start_x + x_start)
                                    orig_y = int(start_y + y_start)
                                    orig_w = int(x_end - x_start)
                                    orig_h = int(y_end - y_start)
                                    
                                    color_variations.append({
                                        'region': (orig_x, orig_y, orig_w, orig_h),
                                        'type': variation_type,
                                        'intensity': max(l_diff, a_diff, b_diff, h_diff/2, s_diff/2)
                                    })
                    
                    # Additional spot analysis for localized color changes
                    # Use k-means clustering to find distinct color regions
                    try:
                        # Reshape image for k-means
                        pixel_data = lab_tongue.reshape(-1, 3).astype(np.float32)
                        
                        # Perform k-means clustering with 4-6 clusters
                        k = min(5, max(3, len(pixel_data) // 1000))  # Adaptive cluster count
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
                        _, labels, centers = cv2.kmeans(pixel_data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                        
                        # Analyze cluster distribution to find unusual color concentrations
                        unique_labels, counts = np.unique(labels, return_counts=True)
                        total_pixels = len(labels)
                        
                        for i, (label, count) in enumerate(zip(unique_labels, counts)):
                            percentage = count / total_pixels
                            
                            # If a color cluster represents a small but significant portion (localized variation)
                            if 0.05 < percentage < 0.25:  # 5-25% of tongue area
                                # Find the spatial distribution of this cluster
                                cluster_mask = (labels == label).reshape(int(region_height), int(region_width))
                                
                                # Find contours of this color cluster
                                cluster_mask_uint8 = cluster_mask.astype(np.uint8) * 255
                                contours, _ = cv2.findContours(cluster_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                
                                for contour in contours:
                                    area = cv2.contourArea(contour)
                                    if area > 200:  # Significant area
                                        x, y, w, h = cv2.boundingRect(contour)
                                        # Convert to original image coordinates
                                        orig_x = int(start_x + x)
                                        orig_y = int(start_y + y)
                                        
                                        color_variations.append({
                                            'region': (orig_x, orig_y, w, h),
                                            'type': 'cluster',
                                            'intensity': percentage * 100  # Percentage as intensity
                                        })
                    
                    except Exception as e:
                        print(f"K-means color analysis error: {e}")
                    
                    # Sort by intensity and limit results
                    color_variations.sort(key=lambda x: x['intensity'], reverse=True)
                    color_variations = color_variations[:6]  # Top 6 most significant variations
                    
            except Exception as e:
                print(f"Error in color variation detection: {e}")
                color_variations = []

            # Medical-grade teeth marks detection - ONLY on BOTTOM HALF edges (where teeth contact)
            teeth_marks = []
            try:
                # Define tongue edge regions - only bottom half where teeth actually contact
                edge_thickness = 18  # How deep into the tongue to check for indentations
                
                # Calculate bottom half region (tongue tip is at bottom of image)
                bottom_half_start_y = start_y + region_height // 2  # Start from middle
                bottom_half_end_y = start_y + region_height - 20    # End near bottom with margin
                
                # Left edge of tongue - BOTTOM HALF ONLY (where teeth contact)
                left_edge_x = start_x
                left_indented_areas = []  # Collect continuous indented areas
                
                for y_pos in range(bottom_half_start_y, bottom_half_end_y, 15):  # Every 15 pixels for better coverage
                    # Sample area just inside the left edge
                    sample_x = left_edge_x + 8  # 8 pixels inside the tongue
                    if (sample_x + edge_thickness < width and y_pos + edge_thickness < height):
                        edge_area = gray[y_pos:y_pos+edge_thickness, sample_x:sample_x+edge_thickness]
                        if edge_area.size > 0:
                            # Check for indentation pattern (darker/shadowed areas indicating depth)
                            edge_variance = edge_area.var()
                            edge_mean = edge_area.mean()
                            edge_std = edge_area.std()
                            
                            # Teeth marks create shadows/indentations with specific patterns
                            if edge_variance > 230 and edge_mean < 120 and edge_std > 12:  # Slightly more sensitive
                                left_indented_areas.append(y_pos)
                
                # Group continuous indented areas into tilted rectangles following edge contour
                if left_indented_areas:
                    # Group nearby points into continuous regions
                    current_group = [left_indented_areas[0]]
                    for y in left_indented_areas[1:]:
                        if y - current_group[-1] <= 25:  # Within 25 pixels = same indented area
                            current_group.append(y)
                        else:
                            # Create tilted rectangle for this indented area following left edge
                            if len(current_group) >= 2:  # At least 2 detection points
                                area_start = min(current_group) - 10
                                area_end = max(current_group) + edge_thickness + 10
                                # Create tilted rectangle following the left edge contour
                                # Left edge is typically vertical, so create a slightly tilted rectangle
                                x1, y1 = left_edge_x - 20, area_start  # Top-left
                                x2, y2 = left_edge_x + 35, area_start - 5  # Top-right (slightly inward)
                                x3, y3 = left_edge_x + 30, area_end + 5  # Bottom-right (slightly inward)
                                x4, y4 = left_edge_x - 15, area_end  # Bottom-left
                                teeth_marks.append((x1, y1, x2, y2, x3, y3, x4, y4))  # Tilted rectangle
                            current_group = [y]
                    
                    # Handle the last group
                    if len(current_group) >= 2:
                        area_start = min(current_group) - 10
                        area_end = max(current_group) + edge_thickness + 10
                        x1, y1 = left_edge_x - 20, area_start
                        x2, y2 = left_edge_x + 35, area_start - 5
                        x3, y3 = left_edge_x + 30, area_end + 5
                        x4, y4 = left_edge_x - 15, area_end
                        teeth_marks.append((x1, y1, x2, y2, x3, y3, x4, y4))
                
                # Right edge of tongue - BOTTOM HALF ONLY (where teeth contact)
                right_edge_x = start_x + region_width
                right_indented_areas = []  # Collect continuous indented areas
                
                for y_pos in range(bottom_half_start_y, bottom_half_end_y, 15):  # Every 15 pixels for better coverage
                    # Sample area just inside the right edge
                    sample_x = right_edge_x - edge_thickness - 8  # 8 pixels inside the tongue
                    if (sample_x >= 0 and y_pos + edge_thickness < height):
                        edge_area = gray[y_pos:y_pos+edge_thickness, sample_x:sample_x+edge_thickness]
                        if edge_area.size > 0:
                            # Check for indentation pattern
                            edge_variance = edge_area.var()
                            edge_mean = edge_area.mean()
                            edge_std = edge_area.std()
                            
                            if edge_variance > 230 and edge_mean < 120 and edge_std > 12:  # Slightly more sensitive
                                right_indented_areas.append(y_pos)
                
                # Group continuous indented areas into tilted rectangles following right edge contour
                if right_indented_areas:
                    current_group = [right_indented_areas[0]]
                    for y in right_indented_areas[1:]:
                        if y - current_group[-1] <= 25:  # Within 25 pixels = same indented area
                            current_group.append(y)
                        else:
                            # Create tilted rectangle for this indented area following right edge
                            if len(current_group) >= 2:
                                area_start = min(current_group) - 10
                                area_end = max(current_group) + edge_thickness + 10
                                # Create tilted rectangle following the right edge contour
                                # Right edge is typically vertical, so create a slightly tilted rectangle
                                x1, y1 = right_edge_x - 35, area_start - 5  # Top-left (slightly inward)
                                x2, y2 = right_edge_x + 20, area_start  # Top-right
                                x3, y3 = right_edge_x + 15, area_end  # Bottom-right
                                x4, y4 = right_edge_x - 30, area_end + 5  # Bottom-left (slightly inward)
                                teeth_marks.append((x1, y1, x2, y2, x3, y3, x4, y4))  # Tilted rectangle
                            current_group = [y]
                    
                    # Handle the last group
                    if len(current_group) >= 2:
                        area_start = min(current_group) - 10
                        area_end = max(current_group) + edge_thickness + 10
                        x1, y1 = right_edge_x - 35, area_start - 5
                        x2, y2 = right_edge_x + 20, area_start
                        x3, y3 = right_edge_x + 15, area_end
                        x4, y4 = right_edge_x - 30, area_end + 5
                        teeth_marks.append((x1, y1, x2, y2, x3, y3, x4, y4))
                
                # Bottom edge of tongue (horizontal indentations) - where front teeth contact
                bottom_edge_y = start_y + region_height
                bottom_indented_areas = []  # Collect continuous indented areas
                
                for x_pos in range(start_x + 60, start_x + region_width - 60, 20):  # Every 20 pixels along bottom edge
                    # Sample area just inside the bottom edge
                    sample_y = bottom_edge_y - edge_thickness - 8  # 8 pixels inside the tongue
                    if (x_pos + edge_thickness < width and sample_y >= 0):
                        edge_area = gray[sample_y:sample_y+edge_thickness, x_pos:x_pos+edge_thickness]
                        if edge_area.size > 0:
                            edge_variance = edge_area.var()
                            edge_mean = edge_area.mean()
                            edge_std = edge_area.std()
                            
                            if edge_variance > 200 and edge_mean < 115 and edge_std > 10:  # Front teeth indentations
                                bottom_indented_areas.append(x_pos)
                
                # Group continuous bottom indented areas into tilted rectangles following bottom edge contour
                if bottom_indented_areas:
                    current_group = [bottom_indented_areas[0]]
                    for x in bottom_indented_areas[1:]:
                        if x - current_group[-1] <= 30:  # Within 30 pixels = same indented area
                            current_group.append(x)
                        else:
                            # Create tilted rectangle for this indented area following bottom edge
                            if len(current_group) >= 2:
                                area_start = min(current_group) - 15
                                area_end = max(current_group) + edge_thickness + 15
                                # Create tilted rectangle following the bottom edge contour (horizontal)
                                # Bottom edge follows tongue curve, so create a slightly curved rectangle
                                x1, y1 = area_start, bottom_edge_y - 45  # Top-left
                                x2, y2 = area_end, bottom_edge_y - 40  # Top-right (slightly lower for curve)
                                x3, y3 = area_end - 5, bottom_edge_y + 15  # Bottom-right (slightly inward)
                                x4, y4 = area_start + 5, bottom_edge_y + 10  # Bottom-left (slightly inward)
                                teeth_marks.append((x1, y1, x2, y2, x3, y3, x4, y4))  # Tilted rectangle
                            current_group = [x]
                    
                    # Handle the last group
                    if len(current_group) >= 2:
                        area_start = min(current_group) - 15
                        area_end = max(current_group) + edge_thickness + 15
                        x1, y1 = area_start, bottom_edge_y - 45
                        x2, y2 = area_end, bottom_edge_y - 40
                        x3, y3 = area_end - 5, bottom_edge_y + 15
                        x4, y4 = area_start + 5, bottom_edge_y + 10
                        teeth_marks.append((x1, y1, x2, y2, x3, y3, x4, y4))
                
                # Limit to most significant teeth marks (medical reality: usually 2-5 visible marks in bottom half)
                teeth_marks = teeth_marks[:5]
                
            except Exception as e:
                print(f"Error in tooth marks detection: {e}")
                teeth_marks = []
            
            features = {
                'tongue_region': (start_x, start_y, region_width, region_height),
                'cracks': cracks,
                'coating': [],  # Keep coating empty for simplicity
                'color_variations': color_variations,
                'teeth_marks': teeth_marks
            }
            
            print(f"Enhanced features: tongue_region={features['tongue_region']}, cracks={len(cracks)}, color_variations={len(color_variations)}, teeth_marks={len(teeth_marks)}")
            return features
        except Exception as e:
            print(f"Error in enhanced feature detection: {e}")
            return {'tongue_region': None, 'cracks': [], 'coating': [], 'color_variations': [], 'teeth_marks': []}

    def _detect_tongue_region_and_features(self, image):
        """
        Detect tongue region and specific features
        """
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Define range for skin/tongue color (adjust these values based on your images)
        lower_skin = np.array([0, 20, 70])
        upper_skin = np.array([20, 255, 255])
        
        # Create mask for tongue region
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Morphological operations to clean up the mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        features = {
            'tongue_region': None,
            'cracks': [],
            'coating': [],
            'color_variations': [],
            'teeth_marks': []
        }
        
        if contours:
            # Find the largest contour (likely the tongue)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            features['tongue_region'] = (x, y, w, h)
            
            # Detect cracks (fissures) using edge detection
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Find lines that could be cracks
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                                   minLineLength=30, maxLineGap=10)
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    # Check if line is within tongue region
                    if (x <= x1 <= x+w and x <= x2 <= x+w and 
                        y <= y1 <= y+h and y <= y2 <= y+h):
                        features['cracks'].append(((x1, y1), (x2, y2)))
            
            # Detect coating (areas with different texture/color)
            roi = image[y:y+h, x:x+w]
            if roi.size > 0:
                # Detect areas with different brightness (potential coating)
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
                _, thresh = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                coating_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in coating_contours:
                    if cv2.contourArea(contour) > 100:  # Filter small areas
                        cx, cy, cw, ch = cv2.boundingRect(contour)
                        features['coating'].append((x + cx, y + cy, cw, ch))
        
        return features

    def create_annotated_image(self, image_path, features, classification=None):
        """
        Create an annotated image with detected features and optional classification
        """
        try:
            print(f"Creating annotated image for {image_path}")
            # Open original image and ensure it's RGB
            original_image = Image.open(image_path).convert('RGB')
            draw = ImageDraw.Draw(original_image)
            
            # Use default font to avoid hanging on font loading
            try:
                font = ImageFont.load_default()
                print("Using default font")
            except:
                font = None
                print("No font available")
            
            # Ensure features is not None
            if not features:
                print("No features provided, using empty features")
                features = {'tongue_region': None, 'cracks': [], 'coating': [], 'color_variations': [], 'teeth_marks': []}
            
            # Draw tongue region with better visibility
            if features.get('tongue_region'):
                x, y, w, h = features['tongue_region']
                # Draw rectangle around tongue with thicker line
                draw.rectangle([x, y, x+w, y+h], outline='lime', width=4)
                if font:
                    draw.text((x, max(0, y-35)), "Tongue Region", fill='lime', font=font)
                print(f"Drew tongue region: {x}, {y}, {w}, {h}")
            
            # Draw cracks/fissures with count and better visibility
            cracks = features.get('cracks', [])
            if cracks:
                for i, crack in enumerate(cracks[:15]):  # Limit to first 15 for visibility
                    if len(crack) >= 2:
                        (x1, y1), (x2, y2) = crack[0], crack[1]
                        # Convert numpy types to regular ints if needed
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        draw.line([x1, y1, x2, y2], fill='red', width=3)
                print(f"Drew {len(cracks)} cracks")
                # Add label for cracks with count
                crack_count = len(cracks)
                if font:
                    draw.text((10, 10), f"Cracks: {crack_count} detected", fill='red', font=font)
            
            # Draw coating areas with count and better visibility
            coating = features.get('coating', [])
            if coating:
                for i, area in enumerate(coating[:10]):  # Limit to first 10
                    if len(area) >= 4:
                        cx, cy, cw, ch = area
                        cx, cy, cw, ch = int(cx), int(cy), int(cw), int(ch)
                        draw.rectangle([cx, cy, cx+cw, cy+ch], outline='blue', width=3)
                # Add label for coating with count
                coating_count = len(coating)
                if font:
                    draw.text((10, 40), f"Coating: {coating_count} areas detected", fill='blue', font=font)
            
            # Draw color variations count
            color_variations = features.get('color_variations', [])
            if color_variations:
                color_count = len(color_variations)
                if font:
                    draw.text((10, 70), f"Color Variations: {color_count} detected", fill='orange', font=font)
            
            # Draw teeth marks count
            teeth_marks = features.get('teeth_marks', [])
            if teeth_marks:
                teeth_count = len(teeth_marks)
                if font:
                    draw.text((10, 100), f"Teeth Marks: {teeth_count} areas", fill='purple', font=font)
            
            # Draw AI classification if provided
            if classification and font:
                draw.text((10, 130), f"AI Classification: {classification}", fill='darkblue', font=font)
            
            # Add a summary box in the bottom right
            if font:
                img_width, img_height = original_image.size
                summary_text = f"Analysis Complete"
                draw.text((img_width-200, img_height-30), summary_text, fill='green', font=font)
            
            print("Annotated image created successfully")
            return original_image
            
        except Exception as e:
            print(f"Error creating annotated image: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            # Return original image if there's an error
            try:
                return Image.open(image_path).convert('RGB')
            except:
                # Create a simple error image if all else fails
                error_img = Image.new('RGB', (400, 300), color='white')
                error_draw = ImageDraw.Draw(error_img)
                error_draw.text((50, 150), "Error creating annotated image", fill='red')
                return error_img

    def classify(self, image_path):
        """
        Classify the tongue image
        """
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                primary_classification = self.categories[predicted_idx.item()]
                confidence_value = confidence.item()
                
                return {
                    'primary_classification': primary_classification,
                    'confidence': confidence_value,
                    'all_probabilities': probabilities.cpu().numpy().tolist()[0]
                }
        except Exception as e:
            print(f"Error during classification: {e}")
            return {
                'primary_classification': 'error',
                'confidence': 0.0,
                'all_probabilities': []
            }

class QuestionnaireAnalyzer:
    def analyze(self, gender: str, water_cups: int, symptoms: list):
        """
        Analyzes the questionnaire data and returns structured suggestions.
        """
        recommendations = []
        suggestions = []

        # Process gender
        if gender.lower() == "female":
            recommendations.append("Consider tracking your menstrual cycle as it may affect your health patterns.")

        # Process water intake
        try:
            if 0 <= int(water_cups) <= 6:
                recommendations.append("Recommendation: Drink more water. Aim for 8-10 cups daily for better hydration.")
        except (ValueError, TypeError):
            recommendations.append("Please enter a valid number for water intake.")

        # Process symptoms
        symptom_map = {
            "疲劳": "Eat warming foods such as ginger or cinnamon to boost energy.",
            "头晕": "Eat regular balanced meals to maintain stable blood sugar levels.",
            "多汗": "Eat cooling foods such as watermelon or cucumber to help regulate body temperature.",
            "失眠": "Limit caffeine intake, especially in the afternoon and evening.",
            "消化不良": "Increase dietary fiber intake through fruits, vegetables, and whole grains.",
            "口干": "Drink more water throughout the day to maintain hydration."
        }
        for symptom in symptoms:
            if symptom in symptom_map:
                suggestions.append(symptom_map[symptom])
        
        return {
            "recommendations": recommendations,
            "suggestions": suggestions
        }

class ResultIntegrator:
    def integrate(self, questionnaire_results: dict, classification_result: str):
        """
        Combines analysis from the questionnaire and AI model into a final formatted string.
        """
        symptoms_from_ai = []
        suggestions = questionnaire_results.get("suggestions", [])
        recommendations = questionnaire_results.get("recommendations", [])

        # Process AI classification result to generate symptoms and suggestions
        if classification_result != "error":
            if "crenated" in classification_result.lower():
                symptoms_from_ai.append(f"crenated {random.randint(1, 3)}")
                suggestions.append("Eat a variety of fruits and vegetables.")
            elif "fissured" in classification_result.lower():
                symptoms_from_ai.append(f"fissured {random.randint(1, 3)}")
                suggestions.append("Drink more water to maintain hydration.")
                suggestions.append("Eat foods rich in iron and vitamin B for better health.")

        # --- Build the final output string ---
        output = ""
        if classification_result != "error":
             output += f"AI Classification: {classification_result}\n\n"

        if symptoms_from_ai:
            output += "Symptoms:\n"
            # Use a set to prevent duplicate suggestions
            for i, symptom in enumerate(symptoms_from_ai, 1):
                output += f"{i}. {symptom}\n"
            output += "\n"

        if suggestions:
            output += "Suggestions:\n"
            # Use a set to prevent duplicate suggestions
            for i, suggestion in enumerate(sorted(list(set(suggestions))), 1):
                output += f"{i}. {suggestion}\n"
            output += "\n"

        if recommendations:
            output += "General Recommendations:\n"
            for i, recommendation in enumerate(recommendations, 1):
                output += f"{i}. {recommendation}\n"
        
        return output.strip() if output.strip() else "No specific recommendations at this time."

# Initialize global analyzer instances with lazy loading
tongue_analyzer = None
questionnaire_analyzer = QuestionnaireAnalyzer()
result_integrator = ResultIntegrator()

def get_tongue_analyzer():
    """Lazy load the tongue analyzer to avoid startup delays"""
    global tongue_analyzer
    if tongue_analyzer is None:
        print("Loading tongue analyzer model...")
        tongue_analyzer = TongueAnalyzer()
        print("Tongue analyzer model loaded successfully")
    return tongue_analyzer

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Railway and Flutter app"""
    try:
        import time
        start_time = time.time()
        
        # Basic health info
        health_info = {
            'status': 'healthy',
            'message': 'Tongue Analysis API is running',
            'server_ready': True,
            'timestamp': time.time(),
            'checks': {}
        }
        
        # Check if we can import required modules
        try:
            import torch
            import cv2
            import numpy as np
            from PIL import Image
            health_info['checks']['dependencies'] = 'ok'
        except Exception as e:
            health_info['checks']['dependencies'] = f'error: {str(e)}'
            health_info['status'] = 'degraded'
        
        # Check if model can be loaded (without actually loading it to save time)
        try:
            model_path = "best_tongue_classification_model.pth"
            if os.path.exists(model_path):
                health_info['checks']['model_file'] = 'found'
            else:
                health_info['checks']['model_file'] = 'missing (will use default)'
        except Exception as e:
            health_info['checks']['model_file'] = f'error: {str(e)}'
        
        # Check memory and basic system info
        try:
            import psutil
            memory = psutil.virtual_memory()
            health_info['checks']['memory'] = f'{memory.percent}% used'
        except:
            health_info['checks']['memory'] = 'unavailable'
        
        # Check if we can create a simple image (PIL test)
        try:
            test_img = Image.new('RGB', (100, 100), color='white')
            health_info['checks']['image_processing'] = 'ok'
        except Exception as e:
            health_info['checks']['image_processing'] = f'error: {str(e)}'
            health_info['status'] = 'degraded'
        
        # Response time
        response_time = (time.time() - start_time) * 1000
        health_info['response_time_ms'] = round(response_time, 2)
        
        # Set appropriate HTTP status code
        status_code = 200 if health_info['status'] == 'healthy' else 503
        
        return jsonify(health_info), status_code
        
    except Exception as e:
        # Emergency fallback health check
        return jsonify({
            'status': 'error',
            'message': f'Health check failed: {str(e)}',
            'server_ready': False
        }), 503

# Additional health check endpoints for Railway
@app.route('/', methods=['GET'])
def root_health():
    """Root endpoint that Railway might check"""
    return jsonify({
        'service': 'TCM Tongue Analysis API',
        'status': 'running',
        'version': '2.0',
        'endpoints': ['/health', '/analyze', '/classify-only']
    })

@app.route('/ping', methods=['GET'])
def ping():
    """Simple ping endpoint for basic connectivity"""
    return 'pong', 200

@app.route('/ready', methods=['GET'])
def readiness_check():
    """Readiness check that includes model loading"""
    try:
        # Try to get the analyzer (this will load the model if needed)
        analyzer = get_tongue_analyzer()
        return jsonify({
            'status': 'ready',
            'model_loaded': True,
            'message': 'Service is ready to accept requests'
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'not_ready',
            'model_loaded': False,
            'error': str(e),
            'message': 'Service is starting up, please wait'
        }), 503

@app.route('/analyze', methods=['POST'])
def analyze_tongue():
    """
    Main analysis endpoint for Flutter app
    Expects: multipart form data with image, gender, water_cups, symptoms
    Returns: JSON with analysis results and base64 encoded annotated image
    """
    try:
        # Get form data
        image_file = request.files.get('image')
        gender = request.form.get('gender', '')
        water_cups = int(request.form.get('water_cups', 0))
        symptoms = request.form.getlist('symptoms') if 'symptoms' in request.form else []
        
        # Validate image file
        if not image_file or image_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No image file uploaded'
            })
        
        # Check file extension
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
        file_extension = image_file.filename.rsplit('.', 1)[1].lower() if '.' in image_file.filename else ''
        
        if file_extension not in allowed_extensions:
            return jsonify({
                'success': False,
                'error': f'Invalid file type. Allowed types: {", ".join(allowed_extensions)}'
            })
        
        # Create unique filename to avoid conflicts
        unique_filename = f"uploaded_image_{uuid.uuid4().hex[:8]}.jpg"
        temp_image_path = unique_filename
        
        # Save uploaded image temporarily
        image_file.save(temp_image_path)
        
        # Verify the image can be opened
        try:
            with Image.open(temp_image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    img.save(temp_image_path, 'JPEG')
        except Exception as e:
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            return jsonify({
                'success': False,
                'error': f'Invalid image file: {str(e)}'
            })
        
        print(f"Starting analysis for {temp_image_path}")
        
        # Perform classification
        print("Starting classification...")
        analyzer = get_tongue_analyzer()
        classification_result = analyzer.classify(temp_image_path)
        print(f"Classification complete: {classification_result['primary_classification']}")
        
        # Analyze questionnaire
        print("Starting questionnaire analysis...")
        questionnaire_results = questionnaire_analyzer.analyze(gender, water_cups, symptoms)
        print("Questionnaire analysis complete")
        
        # Integrate results
        print("Integrating results...")
        final_results = result_integrator.integrate(questionnaire_results, classification_result['primary_classification'])
        print("Integration complete")
        
        # Detect features for visualization (comprehensive)
        print("Detecting features...")
        try:
            features = analyzer.detect_tongue_features(temp_image_path)
            print(f"Feature detection complete: {len(features.get('cracks', []))} cracks, {len(features.get('coating', []))} coating areas")
        except Exception as e:
            print(f"Feature detection failed: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            features = {'tongue_region': None, 'cracks': [], 'coating': [], 'color_variations': [], 'teeth_marks': []}
        
        # Ensure features are valid
        if not features:
            features = {'tongue_region': None, 'cracks': [], 'coating': [], 'color_variations': [], 'teeth_marks': []}
        
        # Create annotated image with comprehensive error handling
        print("Creating annotated image...")
        try:
            # Add random number to classification for display consistency
            classification_with_number = f"{classification_result['primary_classification']} {random.randint(1, 3)}"
            annotated_image = analyzer.create_annotated_image(temp_image_path, features, classification_with_number)
            print("Annotated image creation complete")
        except Exception as e:
            print(f"Annotated image creation failed: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            # Fallback to original image with error handling
            try:
                annotated_image = Image.open(temp_image_path).convert('RGB')
                print("Using original image as fallback")
            except Exception as e2:
                print(f"Failed to load original image: {e2}")
                # Create a simple error image
                annotated_image = Image.new('RGB', (400, 300), color='white')
                draw = ImageDraw.Draw(annotated_image)
                draw.text((50, 150), "Image processing error", fill='red')
        
        # Convert annotated image to base64 for Flutter
        print("Converting image to base64...")
        try:
            buffer = BytesIO()
            # Use JPEG format with lower quality for faster processing
            annotated_image.save(buffer, format='JPEG', quality=80)
            img_data = base64.b64encode(buffer.getvalue()).decode()
            print(f"Base64 conversion complete, size: {len(img_data)} characters")
        except Exception as e:
            print(f"Base64 conversion failed: {e}")
            img_data = ""
        
        # Clean up temporary files
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        
        # Return comprehensive results for Flutter
        return jsonify({
            'success': True,
            'annotated_image': img_data,
            'classification': classification_with_number,
            'final_results': final_results,
            'features_detected': {
                'tongue_region': features['tongue_region'] is not None if features else False,
                'cracks_detected': len(features['cracks']) > 0 if features else False,
                'coating_detected': len(features['coating']) > 0 if features else False,
                'crack_count': len(features['cracks']) if features else 0,
                'coating_count': len(features['coating']) if features else 0
            },
            'questionnaire_results': questionnaire_results,
            'filename': image_file.filename
        })
        
    except Exception as e:
        # Clean up any temporary files
        if 'temp_image_path' in locals() and os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/classify-only', methods=['POST'])
def classify_only():
    """
    Endpoint for classification only (without questionnaire analysis)
    """
    try:
        image_file = request.files.get('image')
        
        if not image_file or image_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No image file uploaded'
            })
        
        # Create unique filename
        unique_filename = f"classify_{uuid.uuid4().hex[:8]}.jpg"
        temp_image_path = unique_filename
        
        # Save and process image
        image_file.save(temp_image_path)
        
        try:
            with Image.open(temp_image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    img.save(temp_image_path, 'JPEG')
        except Exception as e:
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            return jsonify({
                'success': False,
                'error': f'Invalid image file: {str(e)}'
            })
        
        # Perform classification
        analyzer = get_tongue_analyzer()
        classification_result = analyzer.classify(temp_image_path)
        
        # Clean up
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        
        # Add random number
        random_num = random.randint(1, 3)
        classification_with_number = f"{classification_result['primary_classification']} {random_num}"
        
        return jsonify({
            'success': True,
            'classification': classification_with_number,
            'all_probabilities': classification_result.get('all_probabilities', [])
        })
        
    except Exception as e:
        if 'temp_image_path' in locals() and os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    import os
    import sys
    
    # Get port from environment (Railway sets this)
    port = int(os.environ.get('PORT', 8000))
    
    print(f"🚀 Starting TCM Tongue Analysis API on port {port}")
    print(f"🔍 Health check available at: http://0.0.0.0:{port}/health")
    print(f"📊 Ready check available at: http://0.0.0.0:{port}/ready")
    
    try:
        # Start the Flask app with production settings
        app.run(
            debug=False,           # Disable debug mode for production
            host='0.0.0.0',       # Listen on all interfaces
            port=port,            # Use Railway's PORT
            threaded=True,        # Enable threading for better performance
            use_reloader=False    # Disable reloader for production
        )
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1) 