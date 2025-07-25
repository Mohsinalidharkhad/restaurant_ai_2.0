/** @type {import('next').NextConfig} */
const nextConfig = {
  // Configure Next.js to use the frontend directory structure
  experimental: {
    appDir: true,
  },
  
  // Environment variables
  env: {
    BACKEND_URL: process.env.BACKEND_URL || 'http://localhost:8000',
  },
  
  // Configure webpack to handle the new directory structure
  webpack: (config, { isServer }) => {
    // Add custom webpack configurations if needed
    return config;
  },
  
  // File extensions to treat as pages
  pageExtensions: ['ts', 'tsx', 'js', 'jsx'],
  
  // Configure rewrites for API routing
  async rewrites() {
    return [
      // Proxy API requests to backend
      {
        source: '/api/:path*',
        destination: `${process.env.BACKEND_URL || 'http://localhost:8000'}/api/:path*`,
      },
    ];
  },
};

module.exports = nextConfig; 