# TCM Tongue Analysis Python API Server

A Python Flask API server for Traditional Chinese Medicine tongue analysis using AI.

## Features

- AI-powered tongue classification using PyTorch
- Image processing and feature detection
- Questionnaire analysis integration
- RESTful API endpoints
- CORS enabled for Flutter app integration

## API Endpoints

- `GET /health` - Health check
- `POST /analyze` - Full tongue analysis with questionnaire
- `POST /classify-only` - Tongue classification only

## Local Development

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the server:**

   ```bash
   python3 api_server.py
   ```

3. **Test the API:**
   ```bash
   curl http://localhost:5000/health
   ```

## Deployment

### Railway (Recommended)

1. **Create a GitHub repository** and push this code
2. **Go to [railway.app](https://railway.app)**
3. **Connect your GitHub repo**
4. **Deploy automatically**

### Render

1. **Go to [render.com](https://render.com)**
2. **Create a new Web Service**
3. **Connect your GitHub repo**
4. **Set build command:** `pip install -r requirements.txt`
5. **Set start command:** `gunicorn --bind 0.0.0.0:$PORT api_server:app`

## Configuration

- Update the Flutter app's `cloudBaseUrl` with your deployed server URL
- The server automatically uses the `PORT` environment variable in production

## Model Files

Make sure `best_model.pth` and `best_tongue_classification_model.pth` are included in your deployment.
