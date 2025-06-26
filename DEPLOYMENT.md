# Deployment Guide for TCM Tongue Analysis API

This guide will help you deploy your API server online so you can use the Flutter app on your phone.

## Option 1: Railway (Recommended - Free & Easy)

### Prerequisites

- GitHub account
- Railway account (free at railway.app)

### Steps:

1. **Create a GitHub repository** and push your code:

   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote set-url origin https://github.com/HFLam/tcm-tongue-analysis.git
   git push -u origin main
   ```

2. **Deploy to Railway**:

   - Go to [railway.app](https://railway.app)
   - Sign up with GitHub
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository
   - Railway will automatically detect it's a Python app and deploy it

3. **Get your deployment URL**:

   - After deployment, Railway will give you a URL like: `https://your-app-name.railway.app`
   - Copy this URL

4. **Update your Flutter app**:

   - Open `lib/config/api_config.dart`
   - Replace `'https://your-railway-app.railway.app'` with your actual Railway URL
   - Make sure `useCloud = true`

5. **Build and install on your phone**:
   ```bash
   flutter build apk
   # Install the APK on your phone
   ```

## Option 2: Heroku (Alternative)

### Prerequisites

- Heroku account
- Heroku CLI

### Steps:

1. **Install Heroku CLI**:

   ```bash
   # macOS
   brew tap heroku/brew && brew install heroku
   ```

2. **Login to Heroku**:

   ```bash
   heroku login
   ```

3. **Create Heroku app**:

   ```bash
   heroku create your-tcm-app-name
   ```

4. **Deploy**:

   ```bash
   git add .
   git commit -m "Deploy to Heroku"
   git push heroku main
   ```

5. **Get your URL** and update the Flutter app as above.

## Option 3: Render (Another Alternative)

1. Go to [render.com](https://render.com)
2. Create account and new Web Service
3. Connect your GitHub repo
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `python3 api_server.py`
6. Deploy and get your URL

## Testing Your Deployment

After deployment, test your API:

```bash
# Health check
curl https://your-app-url.railway.app/health

# Should return:
# {"status": "healthy", "message": "Tongue Analysis API is running", "model_loaded": true}
```

## Updating the Flutter App

1. **For local development**: Set `useCloud = false` in `lib/config/api_config.dart`
2. **For phone usage**: Set `useCloud = true` and update the `cloudBaseUrl`

## Troubleshooting

### Common Issues:

1. **Model file not found**: Make sure `best_tongue_classification_model.pth` is in your repository
2. **Port issues**: The app automatically uses the `PORT` environment variable
3. **Memory issues**: Railway free tier has 512MB RAM limit - this should be enough for your model

### Checking Logs:

- **Railway**: Go to your project → Deployments → Click on deployment → View logs
- **Heroku**: `heroku logs --tail`
- **Render**: Go to your service → Logs

## Security Notes

- The API is currently open (no authentication)
- For production use, consider adding API keys or authentication
- The model files are quite large (~90MB) - make sure your hosting plan supports this

## Cost

- **Railway**: Free tier includes 500 hours/month
- **Heroku**: Free tier discontinued, paid plans start at $7/month
- **Render**: Free tier available with some limitations

## Next Steps

Once deployed, you can:

1. Build your Flutter app for Android/iOS
2. Install it on your phone
3. Use the app with the cloud API from anywhere!
