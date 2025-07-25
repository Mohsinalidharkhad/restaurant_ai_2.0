// API Base URLs
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 
  (process.env.NODE_ENV === 'production' 
    ? 'https://your-railway-backend-url.railway.app' 
    : 'http://localhost:8000')

// API Endpoints
export const API_ENDPOINTS = {
  CHAT: '/api/chat',
  INITIALIZE: '/api/initialize',
  INITIALIZATION_STATUS: '/api/initialization-status',
} as const

// HTTP Methods
export const HTTP_METHODS = {
  GET: 'GET',
  POST: 'POST',
  PUT: 'PUT',
  DELETE: 'DELETE',
  PATCH: 'PATCH',
} as const

// Request Headers
export const DEFAULT_HEADERS = {
  'Content-Type': 'application/json',
  'Accept': 'application/json',
} as const

// API Response Status Codes
export const STATUS_CODES = {
  OK: 200,
  CREATED: 201,
  BAD_REQUEST: 400,
  UNAUTHORIZED: 401,
  FORBIDDEN: 403,
  NOT_FOUND: 404,
  INTERNAL_SERVER_ERROR: 500,
} as const

// Request Timeouts (in milliseconds)
export const TIMEOUTS = {
  DEFAULT: 10000,      // 10 seconds
  CHAT: 30000,         // 30 seconds for chat responses
  UPLOAD: 60000,       // 60 seconds for file uploads
} as const

// Retry Configuration
export const RETRY_CONFIG = {
  MAX_RETRIES: 3,
  RETRY_DELAY: 1000,   // 1 second
  BACKOFF_FACTOR: 2,   // Exponential backoff
} as const 