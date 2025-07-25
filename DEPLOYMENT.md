# Restaurant Graph Agent - Deployment Guide

This guide will help you deploy your restaurant graph agent project with the frontend on Vercel and backend on Railway.

## Prerequisites

1. **GitHub Account**: Your code should be in a GitHub repository
2. **Vercel Account**: Sign up at [vercel.com](https://vercel.com)
3. **Railway Account**: Sign up at [railway.app](https://railway.app)
4. **Environment Variables**: You'll need API keys for:
   - OpenAI API
   - Supabase
   - Neo4j (if using)

## Frontend Deployment (Vercel)

### Step 1: Prepare Frontend for Deployment

1. **Update API Configuration**:
   - The frontend is already configured to use environment variables
   - Update `frontend/src/constants/api.ts` with your Railway backend URL after deployment

2. **Build Test**:
   ```bash
   cd frontend
   npm install
   npm run build
   ```

### Step 2: Deploy to Vercel

1. **Connect Repository**:
   - Go to [vercel.com](https://vercel.com)
   - Click "New Project"
   - Import your GitHub repository

2. **Configure Project**:
   - **Framework Preset**: Next.js
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `.next`

3. **Environment Variables**:
   Add the following environment variable:
   ```
   NEXT_PUBLIC_API_BASE_URL=https://your-railway-backend-url.railway.app
   ```

4. **Deploy**:
   - Click "Deploy"
   - Vercel will automatically build and deploy your frontend

### Step 3: Update CORS in Backend

After getting your Vercel URL, update the backend CORS settings in Railway with your frontend URL.

## Backend Deployment (Railway)

### Step 1: Prepare Backend for Deployment

1. **Verify Dependencies**:
   - All required packages are in `backend/requirements.txt`
   - Python version is specified in `backend/runtime.txt`

2. **Test Locally**:
   ```bash
   cd backend
   pip install -r requirements.txt
   python -m uvicorn backend.services.backend_api:app --host 0.0.0.0 --port 8000
   ```

### Step 2: Deploy to Railway

1. **Connect Repository**:
   - Go to [railway.app](https://railway.app)
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository

2. **Configure Project**:
   - **Root Directory**: `backend`
   - Railway will automatically detect it's a Python project

3. **Environment Variables**:
   Add the following environment variables in Railway dashboard:
   ```
   OPENAI_API_KEY=your_openai_api_key
   SUPABASE_URL=your_supabase_url
   SUPABASE_ANON_KEY=your_supabase_anon_key
   SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key
   NEO4J_URI=your_neo4j_uri
   NEO4J_USERNAME=your_neo4j_username
   NEO4J_PASSWORD=your_neo4j_password
   ALLOWED_ORIGINS=https://your-vercel-frontend-url.vercel.app,http://localhost:3000
   ENVIRONMENT=production
   DEBUG=false
   LOG_LEVEL=info
   ```

4. **Deploy**:
   - Railway will automatically build and deploy your backend
   - The deployment will use the `Procfile` to start the application

### Step 3: Get Backend URL

1. **Find Your Backend URL**:
   - In Railway dashboard, go to your project
   - Click on the deployed service
   - Copy the generated URL (e.g., `https://your-app-name.railway.app`)

2. **Update Frontend**:
   - Go back to Vercel dashboard
   - Update the `NEXT_PUBLIC_API_BASE_URL` environment variable with your Railway URL

## Post-Deployment Setup

### Step 1: Test the Connection

1. **Test Backend Health**:
   ```bash
   curl https://your-railway-backend-url.railway.app/
   ```

2. **Test Frontend**:
   - Visit your Vercel URL
   - Try sending a message in the chat interface

### Step 2: Monitor Logs

1. **Railway Logs**:
   - In Railway dashboard, go to your service
   - Click "Logs" to monitor backend performance

2. **Vercel Logs**:
   - In Vercel dashboard, go to your project
   - Click "Functions" to see API route logs

### Step 3: Set Up Custom Domains (Optional)

1. **Vercel Custom Domain**:
   - In Vercel dashboard, go to your project
   - Click "Settings" → "Domains"
   - Add your custom domain

2. **Railway Custom Domain**:
   - In Railway dashboard, go to your service
   - Click "Settings" → "Custom Domains"
   - Add your custom domain

## Troubleshooting

### Common Issues

1. **Build Failures**:
   - Check that all dependencies are in `requirements.txt`
   - Verify Python version in `runtime.txt`
   - Check Railway logs for specific error messages

2. **CORS Errors**:
   - Ensure `ALLOWED_ORIGINS` includes your Vercel frontend URL
   - Check that the URL format is correct

3. **Environment Variables**:
   - Verify all required environment variables are set in Railway
   - Check that API keys are valid and have proper permissions

4. **Database Connection Issues**:
   - Ensure Supabase and Neo4j credentials are correct
   - Check that databases are accessible from Railway's servers

### Performance Optimization

1. **Railway Scaling**:
   - Railway automatically scales based on traffic
   - Monitor usage in Railway dashboard

2. **Vercel Optimization**:
   - Vercel automatically optimizes Next.js builds
   - Use Vercel Analytics to monitor performance

## Security Considerations

1. **Environment Variables**:
   - Never commit API keys to your repository
   - Use Railway's environment variable system
   - Rotate API keys regularly

2. **CORS Configuration**:
   - Only allow necessary origins
   - Use HTTPS in production

3. **API Rate Limiting**:
   - Consider implementing rate limiting for your API endpoints
   - Monitor usage to prevent abuse

## Cost Optimization

1. **Railway Pricing**:
   - Railway charges based on usage
   - Monitor your usage in the dashboard
   - Consider upgrading for higher limits

2. **Vercel Pricing**:
   - Vercel has a generous free tier
   - Monitor usage to stay within limits

## Support

If you encounter issues:

1. **Check Logs**: Both Vercel and Railway provide detailed logs
2. **Documentation**: Refer to [Vercel Docs](https://vercel.com/docs) and [Railway Docs](https://docs.railway.app)
3. **Community**: Use GitHub issues or community forums for help

## Next Steps

After successful deployment:

1. **Set up monitoring** for both frontend and backend
2. **Configure alerts** for downtime or errors
3. **Set up CI/CD** for automatic deployments
4. **Add analytics** to track user behavior
5. **Implement backup strategies** for your databases 