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
        self.model_loaded = False
        
        # Define categories with improved mapping
        self.categories = [
            '淡白舌白苔', '红舌黄苔', '淡白舌黄苔', '绛舌灰黑苔', '绛舌黄苔',
            '绛舌白苔', '红舌灰黑苔', '红舌白苔', '淡红舌灰黑苔', '淡红舌黄苔',
            '淡红舌白苔', '青紫舌白苔', '青紫舌黄苔', '青紫舌灰黑苔', '淡白舌灰黑苔'
        ]
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        
        # Initialize model with better error handling
        self._load_model(model_path)
        
        # Improved image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Additional preprocessing transform
        self.preprocess_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.2)),
        ])

    def _load_model(self, model_path):
        """Improved model loading with better error handling"""
        try:
            if os.path.exists(model_path):
                print(f"Loading model from {model_path}...")
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Update categories if available in checkpoint
                if 'categories' in checkpoint:
                    self.categories = checkpoint['categories']
                    print(f"Loaded categories from checkpoint: {len(self.categories)} classes")
                if 'category_to_idx' in checkpoint:
                    self.category_to_idx = checkpoint['category_to_idx']
                
                # Initialize model
                self.model = models.resnet50(pretrained=False)
                self.model.fc = nn.Linear(self.model.fc.in_features, len(self.categories))
                
                # Handle different checkpoint formats
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # Fix state dict keys by removing "model." prefix if present
                new_state_dict = {}
                for key, value in state_dict.items():
                    new_key = key.replace('model.', '') if key.startswith('model.') else key
                    new_state_dict[new_key] = value
                
                self.model.load_state_dict(new_state_dict)
                self.model = self.model.to(self.device)
                self.model.eval()
                self.model_loaded = True
                print(f"Model loaded successfully with {len(self.categories)} classes")
                
            else:
                print(f"Model file not found at {model_path}")
                self._create_fallback_model()
                
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self._create_fallback_model()
    
    def _create_fallback_model(self):
        """Create a simple fallback model for basic classification"""
        print("Creating fallback model...")
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(self.categories))
        self.model = self.model.to(self.device)
        self.model.eval()
        self.model_loaded = True
        print("Fallback model created with pretrained weights")

    def _enhance_image(self, image):
        """Enhanced image preprocessing for better analysis"""
        # Convert to numpy for OpenCV operations
        img_array = np.array(image)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        if len(img_array.shape) == 3:
            # Convert to LAB color space for better contrast enhancement
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            enhanced = clahe.apply(img_array)
        
        # Apply bilateral filter to reduce noise while preserving edges
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Enhance tongue region specifically
        enhanced = self._enhance_tongue_region(enhanced)
        
        return Image.fromarray(enhanced)
    
    def _enhance_tongue_region(self, image):
        """Specifically enhance the tongue region for better feature detection"""
        # Convert to HSV for better tongue detection
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Improved tongue color range detection
        # Multiple ranges to capture different tongue colors
        tongue_ranges = [
            ([0, 40, 40], [25, 255, 255]),    # Red-pink range
            ([160, 40, 40], [180, 255, 255]), # Red range (wrap-around)
            ([15, 30, 80], [35, 180, 255]),   # Yellow-pink range
        ]
        
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in tongue_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Morphological operations to clean up the mask
        kernel = np.ones((7,7), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Find the largest contour (tongue region)
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Create a refined mask
            refined_mask = np.zeros_like(combined_mask)
            cv2.fillPoly(refined_mask, [largest_contour], 255)
            
            # Apply mask to enhance only tongue region
            result = image.copy()
            tongue_region = cv2.bitwise_and(image, image, mask=refined_mask)
            
            # Enhance contrast in tongue region
            tongue_enhanced = cv2.convertScaleAbs(tongue_region, alpha=1.2, beta=10)
            
            # Combine enhanced tongue with original background
            background = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(refined_mask))
            result = cv2.add(tongue_enhanced, background)
            
            return result
        
        return image

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
        Detect tongue region and specific features with error handling
        """
        try:
            print("Feature detection started")
            # Convert to HSV for better color segmentation
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # Define multiple ranges for skin/tongue color (broader detection)
            # Range 1: Pink/Red tones
            lower_skin1 = np.array([0, 20, 50])
            upper_skin1 = np.array([25, 255, 255])
            # Range 2: Pale/Light tones  
            lower_skin2 = np.array([0, 0, 80])
            upper_skin2 = np.array([30, 50, 255])
            # Range 3: Darker red tones
            lower_skin3 = np.array([160, 20, 50])
            upper_skin3 = np.array([180, 255, 255])
            
            # Create masks for tongue region using multiple color ranges
            mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
            mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
            mask3 = cv2.inRange(hsv, lower_skin3, upper_skin3)
            
            # Combine all masks
            mask = cv2.bitwise_or(mask1, mask2)
            mask = cv2.bitwise_or(mask, mask3)
            
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
                
                print(f"Tongue region detected: {x}, {y}, {w}, {h}")
                
                # Detect cracks (fissures) using edge detection
                try:
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
                    
                    print(f"Detected {len(features['cracks'])} cracks")
                except Exception as e:
                    print(f"Crack detection failed: {e}")
                
                # Detect coating (areas with different texture/color)
                try:
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
                    
                    print(f"Detected {len(features['coating'])} coating areas")
                except Exception as e:
                    print(f"Coating detection failed: {e}")
                
                # Detect color variations
                try:
                    roi = image[y:y+h, x:x+w]
                    if roi.size > 0:
                        # Convert to LAB color space for better color analysis
                        lab_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB)
                        
                        # Analyze color variations using standard deviation
                        mean_color = np.mean(lab_roi, axis=(0, 1))
                        std_color = np.std(lab_roi, axis=(0, 1))
                        
                        # If there's significant color variation, mark regions
                        if np.sum(std_color) > 30:  # Threshold for color variation
                            # Create grid to analyze different regions
                            grid_size = 20
                            for i in range(0, h, grid_size):
                                for j in range(0, w, grid_size):
                                    region = lab_roi[i:i+grid_size, j:j+grid_size]
                                    if region.size > 0:
                                        region_mean = np.mean(region, axis=(0, 1))
                                        # Check if this region differs significantly from overall mean
                                        if np.linalg.norm(region_mean - mean_color) > 20:
                                            features['color_variations'].append((x + j, y + i, min(grid_size, w-j), min(grid_size, h-i)))
                    
                    print(f"Detected {len(features['color_variations'])} color variations")
                except Exception as e:
                    print(f"Color variation detection failed: {e}")
                
                # Detect teeth marks (scalloped edges)
                try:
                    # Analyze the contour for irregularities that might indicate teeth marks
                    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                    
                    # Look for indentations along the edges
                    hull = cv2.convexHull(largest_contour, returnPoints=True)
                    hull_area = cv2.contourArea(hull)
                    contour_area = cv2.contourArea(largest_contour)
                    
                    # If there's a significant difference, there might be teeth marks
                    if hull_area > 0 and (hull_area - contour_area) / hull_area > 0.05:
                        # Find defects in the convex hull
                        hull_indices = cv2.convexHull(largest_contour, returnPoints=False)
                        defects = cv2.convexityDefects(largest_contour, hull_indices)
                        
                        if defects is not None:
                            for i in range(defects.shape[0]):
                                s, e, f, d = defects[i, 0]
                                far = tuple(largest_contour[f][0])
                                # Mark significant defects as potential teeth marks
                                if d > 1000:  # Threshold for defect depth
                                    features['teeth_marks'].append((far[0], far[1], 10, 10))
                    
                    print(f"Detected {len(features['teeth_marks'])} teeth marks")
                except Exception as e:
                    print(f"Teeth mark detection failed: {e}")
            
            print(f"Feature detection complete: {features}")
            return features
            
        except Exception as e:
            print(f"Error in feature detection: {e}")
            return {'tongue_region': None, 'cracks': [], 'coating': [], 'color_variations': [], 'teeth_marks': []}

    def _get_default_features(self):
        """Return default features when detection fails"""
        return {
            'tongue_region': None,
            'cracks': [],
            'coating': [],
            'color_variations': [],
            'teeth_marks': [],
            'shape_analysis': {'elongated': False, 'swollen': False}
        }

    def _advanced_feature_detection(self, image):
        """
        Advanced feature detection using multiple algorithms
        """
        features = {
            'tongue_region': None,
            'cracks': [],
            'coating': [],
            'color_variations': [],
            'teeth_marks': [],
            'shape_analysis': {'elongated': False, 'swollen': False}
        }
        
        # Step 1: Better tongue segmentation
        tongue_mask = self._segment_tongue_advanced(image)
        
        if tongue_mask is not None:
            # Get tongue region
            contours, _ = cv2.findContours(tongue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                features['tongue_region'] = (x, y, w, h)
                
                # Extract tongue ROI
                tongue_roi = image[y:y+h, x:x+w]
                mask_roi = tongue_mask[y:y+h, x:x+w]
                
                # Step 2: Advanced crack detection
                features['cracks'] = self._detect_cracks_advanced(tongue_roi, mask_roi, (x, y))
                
                # Step 3: Improved coating detection
                features['coating'] = self._detect_coating_advanced(tongue_roi, mask_roi, (x, y))
                
                # Step 4: Color variation analysis
                features['color_variations'] = self._analyze_color_variations(tongue_roi, mask_roi, (x, y))
                
                # Step 5: Shape analysis
                features['shape_analysis'] = self._analyze_tongue_shape(largest_contour)
        
        return features
    
    def _segment_tongue_advanced(self, image):
        """Advanced tongue segmentation using multiple techniques"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Method 1: HSV-based segmentation with multiple ranges
        hsv_mask = self._create_hsv_tongue_mask(hsv)
        
        # Method 2: LAB-based segmentation
        lab_mask = self._create_lab_tongue_mask(lab)
        
        # Method 3: Watershed segmentation
        watershed_mask = self._create_watershed_mask(image)
        
        # Combine masks using voting
        combined_mask = cv2.bitwise_and(hsv_mask, lab_mask)
        combined_mask = cv2.bitwise_or(combined_mask, watershed_mask)
        
        # Refine mask
        kernel = np.ones((5,5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        return combined_mask
    
    def _create_hsv_tongue_mask(self, hsv):
        """Create tongue mask using HSV color space"""
        # Multiple HSV ranges for different tongue colors
        ranges = [
            ([0, 30, 50], [15, 255, 255]),    # Pink-red
            ([160, 30, 50], [180, 255, 255]), # Red (wrap)
            ([10, 20, 80], [25, 180, 255]),   # Light pink
        ]
        
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        return combined_mask
    
    def _create_lab_tongue_mask(self, lab):
        """Create tongue mask using LAB color space"""
        # LAB ranges for tongue colors
        # A channel: green-red, B channel: blue-yellow
        l_min, a_min, b_min = 50, 115, 125  # Typical tongue color range
        l_max, a_max, b_max = 200, 145, 155
        
        mask = cv2.inRange(lab, (l_min, a_min, b_min), (l_max, a_max, b_max))
        return mask
    
    def _create_watershed_mask(self, image):
        """Create tongue mask using watershed segmentation"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Noise removal
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Find sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
        
        return sure_fg.astype(np.uint8)
    
    def _detect_cracks_advanced(self, tongue_roi, mask_roi, offset):
        """Advanced crack detection using multiple edge detection methods"""
        if tongue_roi.size == 0:
            return []
        
        gray = cv2.cvtColor(tongue_roi, cv2.COLOR_RGB2GRAY)
        
        # Method 1: Canny edge detection with multiple thresholds
        edges1 = cv2.Canny(gray, 30, 100)
        edges2 = cv2.Canny(gray, 50, 150)
        combined_edges = cv2.bitwise_or(edges1, edges2)
        
        # Method 2: Ridge detection using custom kernel
        ridge_kernel = np.array([[-1, -1, -1],
                                [-1,  8, -1],
                                [-1, -1, -1]])
        ridge_response = cv2.filter2D(gray, cv2.CV_32F, ridge_kernel)
        ridge_response = np.abs(ridge_response)
        _, ridge_binary = cv2.threshold(ridge_response, 50, 255, cv2.THRESH_BINARY)
        
        # Combine edge responses
        final_edges = cv2.bitwise_or(combined_edges, ridge_binary.astype(np.uint8))
        
        # Apply mask to keep only tongue edges
        final_edges = cv2.bitwise_and(final_edges, mask_roi)
        
        # Detect lines using HoughLinesP with optimized parameters
        lines = cv2.HoughLinesP(final_edges, 1, np.pi/180, threshold=20, 
                               minLineLength=15, maxLineGap=5)
        
        cracks = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Filter lines by length and orientation
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                if length > 10:  # Minimum crack length
                    # Adjust coordinates with offset
                    cracks.append(((x1 + offset[0], y1 + offset[1]), 
                                 (x2 + offset[0], y2 + offset[1])))
        
        return cracks
    
    def _detect_coating_advanced(self, tongue_roi, mask_roi, offset):
        """Advanced coating detection using texture analysis"""
        if tongue_roi.size == 0:
            return []
        
        gray = cv2.cvtColor(tongue_roi, cv2.COLOR_RGB2GRAY)
        
        # Method 1: Local Binary Pattern for texture analysis
        from skimage.feature import local_binary_pattern
        
        # Calculate LBP
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        
        # Threshold LBP to find textured regions (coating)
        lbp_thresh = np.percentile(lbp, 75)  # Top 25% textured regions
        _, coating_mask = cv2.threshold(lbp.astype(np.uint8), lbp_thresh, 255, cv2.THRESH_BINARY)
        
        # Apply tongue mask
        coating_mask = cv2.bitwise_and(coating_mask, mask_roi)
        
        # Find coating regions
        coating_contours, _ = cv2.findContours(coating_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        coating_areas = []
        for contour in coating_contours:
            if cv2.contourArea(contour) > 50:  # Minimum coating area
                x, y, w, h = cv2.boundingRect(contour)
                coating_areas.append((x + offset[0], y + offset[1], w, h))
        
        return coating_areas
    
    def _analyze_color_variations(self, tongue_roi, mask_roi, offset):
        """Analyze color variations in the tongue"""
        if tongue_roi.size == 0:
            return []
        
        # Convert to HSV for better color analysis
        hsv_roi = cv2.cvtColor(tongue_roi, cv2.COLOR_RGB2HSV)
        
        # Apply mask
        masked_hsv = cv2.bitwise_and(hsv_roi, hsv_roi, mask=mask_roi)
        
        # Calculate color statistics
        h_values = masked_hsv[:,:,0][mask_roi > 0]
        s_values = masked_hsv[:,:,1][mask_roi > 0]
        v_values = masked_hsv[:,:,2][mask_roi > 0]
        
        color_variations = []
        
        # Detect unusual color regions
        h_mean, h_std = np.mean(h_values), np.std(h_values)
        unusual_color_mask = np.abs(masked_hsv[:,:,0] - h_mean) > 2 * h_std
        unusual_color_mask = unusual_color_mask.astype(np.uint8) * 255
        unusual_color_mask = cv2.bitwise_and(unusual_color_mask, mask_roi)
        
        # Find unusual color regions
        color_contours, _ = cv2.findContours(unusual_color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in color_contours:
            if cv2.contourArea(contour) > 30:
                x, y, w, h = cv2.boundingRect(contour)
                color_variations.append((x + offset[0], y + offset[1], w, h))
        
        return color_variations
    
    def _analyze_tongue_shape(self, contour):
        """Analyze tongue shape characteristics"""
        # Calculate shape metrics
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        
        # Extent (area ratio)
        rect_area = w * h
        extent = float(area) / rect_area
        
        # Solidity (convex hull ratio)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area
        
        # Determine shape characteristics
        elongated = aspect_ratio > 1.5  # More elongated than normal
        swollen = extent > 0.8 and solidity > 0.9  # Very filled and smooth
        
        return {
            'elongated': elongated,
            'swollen': swollen,
            'aspect_ratio': aspect_ratio,
            'extent': extent,
            'solidity': solidity
        }

    def create_annotated_image(self, image_path, features):
        """
        Create an annotated image with detected features
        """
        try:
            print(f"Creating annotated image for {image_path}")
            # Open original image
            original_image = Image.open(image_path)
            draw = ImageDraw.Draw(original_image)
            
            # Use default font to avoid hanging on font loading
            try:
                font = ImageFont.load_default()
                print("Using default font")
            except:
                font = None
                print("No font available")
            
            # Draw tongue region
            if features and features['tongue_region']:
                x, y, w, h = features['tongue_region']
                # Draw rectangle around tongue
                draw.rectangle([x, y, x+w, y+h], outline='green', width=3)
                if font:
                    draw.text((x, y-30), "Tongue Region", fill='green', font=font)
                print(f"Drew tongue region: {x}, {y}, {w}, {h}")
            
            # Draw cracks/fissures
            if features and features['cracks']:
                for i, ((x1, y1), (x2, y2)) in enumerate(features['cracks'][:10]):  # Limit for performance
                    draw.line([x1, y1, x2, y2], fill='red', width=2)
                print(f"Drew {min(len(features['cracks']), 10)} cracks")
                if font:
                    draw.text((10, 10), f"Cracks: {len(features['cracks'])}", fill='red', font=font)
            
            # Draw coating areas
            if features and features['coating']:
                for cx, cy, cw, ch in features['coating'][:15]:  # Limit for performance
                    draw.rectangle([cx, cy, cx+cw, cy+ch], outline='blue', width=2)
                print(f"Drew {min(len(features['coating']), 15)} coating areas")
                if font:
                    draw.text((10, 40), f"Coating: {len(features['coating'])}", fill='blue', font=font)
            
            # Draw color variations
            if features and features['color_variations']:
                for cx, cy, cw, ch in features['color_variations'][:10]:  # Limit for performance
                    draw.rectangle([cx, cy, cx+cw, cy+ch], outline='orange', width=1)
                print(f"Drew {min(len(features['color_variations']), 10)} color variations")
                if font:
                    draw.text((10, 70), f"Color Variations: {len(features['color_variations'])}", fill='orange', font=font)
            
            # Draw teeth marks
            if features and features['teeth_marks']:
                for tx, ty, tw, th in features['teeth_marks'][:8]:  # Limit for performance
                    draw.ellipse([tx-tw//2, ty-th//2, tx+tw//2, ty+th//2], outline='purple', width=2)
                print(f"Drew {min(len(features['teeth_marks']), 8)} teeth marks")
                if font:
                    draw.text((10, 100), f"Teeth Marks: {len(features['teeth_marks'])}", fill='purple', font=font)
            
            return original_image.convert('RGB')
            
        except Exception as e:
            print(f"Error creating annotated image: {e}")
            # Return original image if there's an error
            return Image.open(image_path)
    
    def _load_fonts(self):
        """Load fonts with better fallback handling"""
        try:
            font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",  # Linux
                "/System/Library/Fonts/PingFang.ttc",  # macOS
                "/System/Library/Fonts/STHeiti Light.ttc",  # macOS
                "/System/Library/Fonts/Helvetica.ttc",  # macOS
                "arial.ttf",  # Windows
            ]
            
            # Try to load fonts in different sizes
            font_large = font_medium = font_small = None
            
            for font_path in font_paths:
                try:
                    font_large = ImageFont.truetype(font_path, 28)
                    font_medium = ImageFont.truetype(font_path, 20)
                    font_small = ImageFont.truetype(font_path, 16)
                    break
                except:
                    continue
            
            # Fallback to default fonts
            if font_large is None:
                font_large = ImageFont.load_default()
                font_medium = ImageFont.load_default()
                font_small = ImageFont.load_default()
            
            return font_large, font_medium, font_small
            
        except:
            default_font = ImageFont.load_default()
            return default_font, default_font, default_font

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
    
    def _intelligent_fallback_classification(self, image):
        """
        Intelligent fallback classification based on basic image analysis
        """
        try:
            # Convert to numpy array for analysis
            img_array = np.array(image)
            
            # Analyze average color
            avg_color = np.mean(img_array, axis=(0, 1))
            r, g, b = avg_color
            
            # Convert to HSV for better color analysis
            hsv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            avg_hsv = np.mean(hsv_img, axis=(0, 1))
            h, s, v = avg_hsv
            
            # Simple heuristic classification based on color
            if r > 150 and g < 120:  # Reddish
                if s > 100:  # High saturation
                    classification = '红舌黄苔'
                else:
                    classification = '红舌白苔'
            elif r < 130 and g > 120:  # Pale/whitish
                classification = '淡白舌白苔'
            elif b > 130:  # Bluish/purple
                classification = '青紫舌白苔'
            else:  # Default to normal
                classification = '淡红舌白苔'
            
            # Calculate a rough confidence based on color distinctiveness
            color_variance = np.std(img_array)
            confidence = min(0.6, color_variance / 255 * 0.6 + 0.2)
            
            return {
                'primary_classification': classification,
                'confidence': confidence,
                'all_probabilities': [0.1] * len(self.categories),
                'fallback_method': 'color_analysis',
                'avg_color': avg_color.tolist(),
                'needs_review': True
            }
            
        except Exception as e:
            print(f"Fallback classification failed: {e}")
            return {
                'primary_classification': self.categories[0],
                'confidence': 0.1,
                'all_probabilities': [0.1] * len(self.categories),
                'error': str(e)
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

# Initialize global analyzer instances with error handling
try:
    print("Initializing tongue analyzer...")
    tongue_analyzer = TongueAnalyzer()
    print("Tongue analyzer initialized successfully")
except Exception as e:
    print(f"Error initializing tongue analyzer: {e}")
    # Create a minimal analyzer for health checks
    class MinimalAnalyzer:
        def __init__(self):
            self.model_loaded = False
            self.categories = ['淡白舌白苔']
        def classify(self, path):
            return {'primary_classification': '淡白舌白苔', 'confidence': 0.5}
        def detect_tongue_features(self, path):
            return {'tongue_region': None, 'cracks': [], 'coating': []}
        def create_annotated_image(self, path, features):
            return Image.open(path)
    tongue_analyzer = MinimalAnalyzer()

questionnaire_analyzer = QuestionnaireAnalyzer()
result_integrator = ResultIntegrator()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Flutter app"""
    return jsonify({
        'status': 'healthy',
        'message': 'Tongue Analysis API is running',
        'model_loaded': len(tongue_analyzer.categories) > 0
    })

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
        classification_result = tongue_analyzer.classify(temp_image_path)
        print(f"Classification complete: {classification_result['primary_classification']}")
        
        # Analyze questionnaire
        print("Starting questionnaire analysis...")
        questionnaire_results = questionnaire_analyzer.analyze(gender, water_cups, symptoms)
        print("Questionnaire analysis complete")
        
        # Integrate results
        print("Integrating results...")
        final_results = result_integrator.integrate(questionnaire_results, classification_result['primary_classification'])
        print("Integration complete")
        
        # Detect features for visualization (simplified)
        print("Detecting features...")
        try:
            features = tongue_analyzer.detect_tongue_features(temp_image_path)
            print("Feature detection complete")
        except Exception as e:
            print(f"Feature detection failed: {e}")
            features = {'tongue_region': None, 'cracks': [], 'coating': [], 'color_variations': [], 'teeth_marks': []}
        
        # Create annotated image
        print("Creating annotated image...")
        try:
            annotated_image = tongue_analyzer.create_annotated_image(temp_image_path, features)
            print("Annotated image creation complete")
        except Exception as e:
            print(f"Annotated image creation failed: {e}")
            # Fallback to original image
            annotated_image = Image.open(temp_image_path)
        
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
        
        # Add random number to classification
        random_num = random.randint(1, 3)
        classification_with_number = f"{classification_result['primary_classification']} {random_num}"
        
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
                'color_variations_detected': len(features['color_variations']) > 0 if features else False,
                'teeth_marks_detected': len(features['teeth_marks']) > 0 if features else False,
                'crack_count': len(features['cracks']) if features else 0,
                'coating_count': len(features['coating']) if features else 0,
                'color_variation_count': len(features['color_variations']) if features else 0,
                'teeth_mark_count': len(features['teeth_marks']) if features else 0
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
        classification_result = tongue_analyzer.classify(temp_image_path)
        
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
    port = int(os.environ.get('PORT', 8000))
    app.run(debug=False, host='0.0.0.0', port=port) 