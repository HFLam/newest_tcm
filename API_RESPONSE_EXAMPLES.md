# API Response Examples - TCM Tongue Analysis Server

## 📊 What Data Will Be Returned When You Send an Image

When you send an image to the TCM server, you'll receive a comprehensive JSON response with multiple types of analysis data.

---

## 🎯 Main Analysis Endpoint: `/analyze`

**Request Method**: `POST`  
**Content-Type**: `multipart/form-data`

### Required Parameters:

- `image`: Image file (PNG, JPG, JPEG, GIF, BMP, TIFF)
- `gender`: String ("male" or "female")
- `water_cups`: Integer (number of cups of water per day)
- `symptoms`: Array of strings (optional)

### Example Request (cURL):

```bash
curl -X POST http://your-server.com/analyze \
  -F "image=@tongue_image.jpg" \
  -F "gender=female" \
  -F "water_cups=6" \
  -F "symptoms=疲劳" \
  -F "symptoms=口干"
```

---

## ✅ Successful Response Structure

```json
{
  "success": true,
  "annotated_image": "base64_encoded_image_string_here...",
  "classification": "淡红舌白苔 2",
  "final_results": "AI Classification: 淡红舌白苔 2\n\nSuggestions:\n1. Drink more water throughout the day to maintain hydration.\n2. Eat warming foods such as ginger or cinnamon to boost energy.\n\nGeneral Recommendations:\n1. Consider tracking your menstrual cycle as it may affect your health patterns.",
  "features_detected": {
    "tongue_region": true,
    "cracks_detected": true,
    "coating_detected": false,
    "crack_count": 8,
    "coating_count": 0
  },
  "questionnaire_results": {
    "recommendations": [
      "Consider tracking your menstrual cycle as it may affect your health patterns."
    ],
    "suggestions": [
      "Eat warming foods such as ginger or cinnamon to boost energy.",
      "Drink more water throughout the day to maintain hydration."
    ]
  },
  "filename": "tongue_image.jpg"
}
```

---

## 📋 Detailed Field Explanations

### 1. **`success`** (boolean)

- `true`: Analysis completed successfully
- `false`: Error occurred during processing

### 2. **`annotated_image`** (string)

- **Base64 encoded JPEG image** showing detected features
- **Visual annotations include:**
  - 🟢 **Lime rectangle**: Tongue region boundary
  - 🔴 **Red lines**: Detected cracks/fissures
  - 🔵 **Blue rectangles**: Coating areas
  - 📝 **Text labels**: Feature counts and AI classification
- **Usage**: Decode base64 and display as image in your app

### 3. **`classification`** (string)

- **AI tongue classification result** with random number (1-3)
- **Examples**:
  - `"淡红舌白苔 1"`
  - `"红舌黄苔 2"`
  - `"绛舌灰黑苔 3"`

### 4. **`final_results`** (string)

- **Formatted text summary** combining AI analysis + questionnaire
- **Includes**:
  - AI Classification
  - Symptoms (if detected)
  - Personalized suggestions
  - General recommendations
- **Usage**: Display directly to user as analysis summary

### 5. **`features_detected`** (object)

- **`tongue_region`** (boolean): Whether tongue area was detected
- **`cracks_detected`** (boolean): Whether cracks/fissures were found
- **`coating_detected`** (boolean): Whether tongue coating was detected
- **`crack_count`** (integer): Number of cracks detected (0-15+)
- **`coating_count`** (integer): Number of coating areas detected

### 6. **`questionnaire_results`** (object)

- **`recommendations`** (array): General health recommendations
- **`suggestions`** (array): Specific dietary/lifestyle suggestions based on symptoms

### 7. **`filename`** (string)

- Original uploaded filename for reference

---

## 🎯 Classification-Only Endpoint: `/classify-only`

**Request Method**: `POST`  
**Content-Type**: `multipart/form-data`

### Required Parameters:

- `image`: Image file only

### Response Structure:

```json
{
  "success": true,
  "classification": "淡红舌白苔 1",
  "all_probabilities": [0.23, 0.45, 0.12, 0.08, 0.05, ...]
}
```

### Field Explanations:

- **`classification`**: AI tongue classification with random number
- **`all_probabilities`**: Confidence scores for all 15 tongue categories

---

## ❌ Error Response Structure

```json
{
  "success": false,
  "error": "Error description here"
}
```

### Common Error Messages:

- `"No image file uploaded"`
- `"Invalid file type. Allowed types: png, jpg, jpeg, gif, bmp, tiff"`
- `"Invalid image file: [specific error]"`

---

## 🖼️ Annotated Image Details

The `annotated_image` field contains a **base64-encoded JPEG** with visual annotations:

### Visual Elements:

1. **🟢 Tongue Region**: Bright lime rectangle outlining detected tongue area
2. **🔴 Cracks**: Red lines showing detected fissures/cracks
3. **🔵 Coating**: Blue rectangles around coating areas
4. **📊 Feature Counts**: Text showing number of detected features
5. **🤖 AI Classification**: Classification result displayed on image

### How to Use in Your App:

```dart
// Flutter example
final imageBytes = base64Decode(result['annotated_image']);
Image.memory(imageBytes)
```

```javascript
// JavaScript example
const imageData = `data:image/jpeg;base64,${result.annotated_image}`;
document.getElementById("result-image").src = imageData;
```

---

## 📈 Feature Detection Capabilities

### **Crack Detection**:

- **Method**: Multi-scale edge detection with medical validation
- **Sensitivity**: Detects cracks as small as 12 pixels
- **Count Range**: 0-15+ cracks typically detected
- **Visualization**: Red lines on annotated image

### **Coating Detection**:

- **Method**: LAB color space analysis + texture analysis
- **Areas**: Rectangular regions of coating
- **Visualization**: Blue rectangles on annotated image

### **Color Variations**:

- **Method**: Cluster analysis in multiple color spaces
- **Types**: Lightness variations, color casts
- **Count**: Included in feature summary

### **Teeth Marks**:

- **Method**: Morphological analysis for indentations
- **Detection**: Edge-based pattern recognition
- **Count**: Included in feature summary

---

## 🎯 Integration Tips

### For Flutter Apps:

```dart
// Handle the response
if (result['success'] == true) {
  final annotatedImage = result['annotated_image'];
  final classification = result['classification'];
  final summary = result['final_results'];
  final featureCount = result['features_detected']['crack_count'];
}
```

### For Web Apps:

```javascript
// Handle the response
if (response.success) {
  displayAnnotatedImage(response.annotated_image);
  showClassification(response.classification);
  displaySummary(response.final_results);
}
```

---

## 📊 Response Size Information

- **Typical annotated image size**: 50-100KB (base64: ~70,000 characters)
- **Total JSON response size**: 80-120KB
- **Response time**: 2-8 seconds depending on image complexity

The server returns comprehensive medical-grade analysis with visual annotations ready for immediate display in your application! 🎉
