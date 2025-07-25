# Quick Deployment Guide - Restaurant Graph Agent

## 🚀 Ready to Deploy!

Your project is now configured for deployment to Vercel (frontend) and Railway (backend).

## 📋 Pre-Deployment Checklist

✅ **Project Structure**: Verified  
✅ **Frontend Build**: Tested and working  
✅ **Backend Dependencies**: Installed and ready  
✅ **Configuration Files**: All created  
✅ **Environment Setup**: Templates provided  

## 🎯 Quick Deployment Steps

### Step 1: Backend Deployment (Railway)

1. **Go to Railway**: [railway.app](https://railway.app)
2. **Create New Project**: "Deploy from GitHub repo"
3. **Select Repository**: Choose your restaurant-graph-agent repo
4. **Configure Settings**:
   - **Root Directory**: `backend`
   - **Build Command**: (auto-detected)
   - **Start Command**: (auto-detected from Procfile)

5. **Add Environment Variables**:
   ```
   OPENAI_API_KEY=your_openai_api_key
   SUPABASE_URL=your_supabase_url
   SUPABASE_ANON_KEY=your_supabase_anon_key
   SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key
   NEO4J_URI=your_neo4j_uri
   NEO4J_USERNAME=your_neo4j_username
   NEO4J_PASSWORD=your_neo4j_password
   ENVIRONMENT=production
   DEBUG=false
   LOG_LEVEL=info
   ```

6. **Deploy**: Click "Deploy" and wait for completion
7. **Get Backend URL**: Copy the generated URL (e.g., `https://your-app.railway.app`)

### Step 2: Frontend Deployment (Vercel)

1. **Go to Vercel**: [vercel.com](https://vercel.com)
2. **Create New Project**: "Import Git Repository"
3. **Select Repository**: Choose your restaurant-graph-agent repo
4. **Configure Settings**:
   - **Framework Preset**: Next.js
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `.next`

5. **Add Environment Variable**:
   ```
   NEXT_PUBLIC_API_BASE_URL=https://your-railway-backend-url.railway.app
   ```

6. **Deploy**: Click "Deploy" and wait for completion
7. **Get Frontend URL**: Copy the generated URL (e.g., `https://your-app.vercel.app`)

### Step 3: Update CORS Settings

1. **Go back to Railway dashboard**
2. **Add Environment Variable**:
   ```
   ALLOWED_ORIGINS=https://your-vercel-frontend-url.vercel.app,http://localhost:3000
   ```
3. **Redeploy**: Railway will automatically redeploy with the new CORS settings

## 🔧 Environment Variables Reference

### Backend (Railway) Required Variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | `sk-...` |
| `SUPABASE_URL` | Your Supabase project URL | `https://xxx.supabase.co` |
| `SUPABASE_ANON_KEY` | Supabase anonymous key | `eyJ...` |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service role key | `eyJ...` |
| `NEO4J_URI` | Neo4j database URI | `bolt://localhost:7687` |
| `NEO4J_USERNAME` | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | `password` |
| `ALLOWED_ORIGINS` | CORS allowed origins | `https://your-app.vercel.app` |

### Frontend (Vercel) Required Variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `NEXT_PUBLIC_API_BASE_URL` | Your Railway backend URL | `https://your-app.railway.app` |

## 🧪 Testing Your Deployment

### Test Backend Health:
```bash
curl https://your-railway-backend-url.railway.app/
```

### Test Frontend:
1. Visit your Vercel URL
2. Try sending a message in the chat interface
3. Check if the response comes from your Railway backend

## 🔍 Troubleshooting

### Common Issues:

1. **Build Failures**:
   - Check Railway logs for specific errors
   - Verify all environment variables are set
   - Ensure Python version matches `runtime.txt`

2. **CORS Errors**:
   - Verify `ALLOWED_ORIGINS` includes your Vercel URL
   - Check URL format (no trailing slashes)
   - Redeploy backend after updating CORS

3. **API Connection Issues**:
   - Verify `NEXT_PUBLIC_API_BASE_URL` is correct
   - Check that backend is running (Railway dashboard)
   - Test backend health endpoint

4. **Environment Variables**:
   - Double-check all required variables are set
   - Verify API keys are valid and have proper permissions
   - Check for typos in variable names

## 📊 Monitoring

### Railway Monitoring:
- **Logs**: View real-time logs in Railway dashboard
- **Metrics**: Monitor CPU, memory, and network usage
- **Deployments**: Track deployment status and history

### Vercel Monitoring:
- **Analytics**: View performance metrics
- **Functions**: Monitor API route performance
- **Deployments**: Track build and deployment status

## 💰 Cost Optimization

### Railway:
- Monitor usage in dashboard
- Consider upgrading for higher limits
- Set up usage alerts

### Vercel:
- Free tier is generous
- Monitor usage to stay within limits
- Consider Pro plan for custom domains

## 🆘 Support

- **Railway Docs**: [docs.railway.app](https://docs.railway.app)
- **Vercel Docs**: [vercel.com/docs](https://vercel.com/docs)
- **Project Issues**: Use GitHub issues for project-specific problems

## 🎉 Success!

Once deployed, your restaurant graph agent will be available at:
- **Frontend**: `https://your-app.vercel.app`
- **Backend**: `https://your-app.railway.app`

Your AI-powered restaurant assistant is now live and ready to serve customers! 🍽️🤖 