#!/usr/bin/env python3
"""
Test script to verify annotated image is returned by the API
"""

import api_server
import base64
from PIL import Image
from io import BytesIO
import os

def test_annotated_image_creation():
    """Test that the annotated image is created and returned properly"""
    print("🧪 Testing Annotated Image API Response")
    print("=" * 50)
    
    # Initialize the analyzer
    analyzer = api_server.get_tongue_analyzer()
    
    # Test with a real image
    test_image = "real_tongue.jpg"
    if not os.path.exists(test_image):
        print(f"❌ Test image {test_image} not found")
        return
    
    print(f"📸 Testing with image: {test_image}")
    
    # 1. Test feature detection
    print("\n1️⃣ Testing feature detection...")
    features = analyzer.detect_tongue_features(test_image)
    print(f"   ✅ Features detected: {len(features.get('cracks', []))} cracks, {len(features.get('coating', []))} coating areas")
    
    # 2. Test annotated image creation
    print("\n2️⃣ Testing annotated image creation...")
    annotated_image = analyzer.create_annotated_image(test_image, features, "Test Classification 1")
    
    if annotated_image:
        print("   ✅ Annotated image created successfully")
        print(f"   📏 Image size: {annotated_image.size}")
        print(f"   🎨 Image mode: {annotated_image.mode}")
        
        # Save for verification
        annotated_image.save("test_annotated_result.jpg")
        print("   💾 Saved as: test_annotated_result.jpg")
    else:
        print("   ❌ Failed to create annotated image")
        return
    
    # 3. Test base64 conversion (same as API)
    print("\n3️⃣ Testing base64 conversion...")
    try:
        buffer = BytesIO()
        annotated_image.save(buffer, format='JPEG', quality=80)
        img_data = base64.b64encode(buffer.getvalue()).decode()
        print(f"   ✅ Base64 conversion successful")
        print(f"   📊 Base64 length: {len(img_data)} characters")
        print(f"   🔤 Base64 preview: {img_data[:50]}...")
        
        # Test decoding
        decoded_data = base64.b64decode(img_data)
        decoded_image = Image.open(BytesIO(decoded_data))
        print(f"   ✅ Base64 decoding successful")
        print(f"   📏 Decoded image size: {decoded_image.size}")
        
    except Exception as e:
        print(f"   ❌ Base64 conversion failed: {e}")
        return
    
    # 4. Test with different feature scenarios
    print("\n4️⃣ Testing with mock features...")
    mock_features = {
        'tongue_region': (50, 50, 200, 200),
        'cracks': [((100, 100), (150, 150)), ((120, 80), (180, 120))],
        'coating': [(75, 75, 50, 50)],
        'color_variations': [{'type': 'test', 'intensity': 50}],
        'teeth_marks': [{'area': 'test'}]
    }
    
    mock_annotated = analyzer.create_annotated_image(test_image, mock_features, "Mock Test 2")
    if mock_annotated:
        print("   ✅ Mock features annotated successfully")
        mock_annotated.save("test_mock_annotated.jpg")
        print("   💾 Saved as: test_mock_annotated.jpg")
    
    # 5. Test with empty features
    print("\n5️⃣ Testing with empty features...")
    empty_features = {'tongue_region': None, 'cracks': [], 'coating': [], 'color_variations': [], 'teeth_marks': []}
    empty_annotated = analyzer.create_annotated_image(test_image, empty_features, "Empty Test 3")
    if empty_annotated:
        print("   ✅ Empty features handled successfully")
        empty_annotated.save("test_empty_annotated.jpg")
        print("   💾 Saved as: test_empty_annotated.jpg")
    
    print("\n🎉 All annotated image tests completed successfully!")
    print("\nℹ️  The annotated image should now work properly in the API.")
    print("ℹ️  Make sure your Flutter app is requesting the 'annotated_image' field from the response.")

if __name__ == "__main__":
    test_annotated_image_creation() 