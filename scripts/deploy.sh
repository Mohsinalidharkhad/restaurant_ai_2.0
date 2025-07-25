#!/bin/bash

# Restaurant Graph Agent Deployment Script
# This script helps prepare and deploy the project to Vercel and Railway

set -e

echo "🚀 Restaurant Graph Agent Deployment Script"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "requirements.txt" ] || [ ! -d "frontend" ] || [ ! -d "backend" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

print_status "Checking project structure..."

# Check frontend
if [ -f "frontend/package.json" ]; then
    print_success "Frontend package.json found"
else
    print_error "Frontend package.json not found"
    exit 1
fi

# Check backend
if [ -f "backend/requirements.txt" ]; then
    print_success "Backend requirements.txt found"
else
    print_error "Backend requirements.txt not found"
    exit 1
fi

print_status "Testing frontend build..."
cd frontend
if npm run build > /dev/null 2>&1; then
    print_success "Frontend builds successfully"
else
    print_error "Frontend build failed. Please fix the issues first."
    exit 1
fi
cd ..

print_status "Testing backend dependencies..."
cd backend
if python3 -c "import fastapi, uvicorn" > /dev/null 2>&1; then
    print_success "Backend dependencies are available"
else
    print_warning "Backend dependencies not installed. Installing now..."
    python3 -m pip install -r requirements.txt
fi
cd ..

print_status "Checking environment files..."

# Check if .env files exist
if [ ! -f "backend/.env" ]; then
    print_warning "Backend .env file not found"
    print_status "Please create backend/.env with your environment variables"
    print_status "You can use backend/env.example as a template"
fi

print_status "Checking configuration files..."

# Check if all required config files exist
required_files=(
    "frontend/vercel.json"
    "backend/railway.json"
    "backend/Procfile"
    "backend/runtime.txt"
    "backend/requirements.txt"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        print_success "$file exists"
    else
        print_error "$file missing"
        exit 1
    fi
done

print_success "All configuration files are present"

echo ""
echo "📋 Deployment Checklist:"
echo "========================"
echo ""
echo "1. ✅ Project structure verified"
echo "2. ✅ Frontend builds successfully"
echo "3. ✅ Backend dependencies checked"
echo "4. ✅ Configuration files present"
echo ""
echo "🔧 Next Steps:"
echo "=============="
echo ""
echo "1. Push your code to GitHub if not already done"
echo "2. Deploy backend to Railway:"
echo "   - Go to https://railway.app"
echo "   - Create new project from GitHub repo"
echo "   - Set root directory to 'backend'"
echo "   - Add environment variables (see backend/env.example)"
echo ""
echo "3. Deploy frontend to Vercel:"
echo "   - Go to https://vercel.com"
echo "   - Create new project from GitHub repo"
echo "   - Set root directory to 'frontend'"
echo "   - Add NEXT_PUBLIC_API_BASE_URL environment variable"
echo ""
echo "4. Update CORS settings after getting URLs"
echo ""
echo "📖 For detailed instructions, see DEPLOYMENT.md"
echo ""
print_success "Deployment preparation completed!" 