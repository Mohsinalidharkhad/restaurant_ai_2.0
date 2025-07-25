// Re-export all constants
export * from './api'
export * from './ui'

// Application Constants
export const APP_NAME = 'Restaurant Graph Agent'
export const APP_VERSION = '1.0.0'
export const APP_DESCRIPTION = 'AI-powered restaurant assistant with advanced menu knowledge and reservation management'

// Feature Flags
export const FEATURES = {
  CHAT_ENABLED: true,
  RESERVATIONS_ENABLED: true,
  MENU_SEARCH_ENABLED: true,
  FAQ_ENABLED: true,
  DEBUGGING_ENABLED: process.env.NODE_ENV === 'development',
} as const

// Local Storage Keys
export const STORAGE_KEYS = {
  CHAT_HISTORY: 'restaurant_chat_history',
  USER_PREFERENCES: 'restaurant_user_preferences',
  THEME: 'restaurant_theme',
  SIDEBAR_STATE: 'restaurant_sidebar_state',
} as const

// Date/Time Formats
export const DATE_FORMATS = {
  DISPLAY: 'MMM dd, yyyy',
  FULL: 'MMMM dd, yyyy',
  SHORT: 'MM/dd/yyyy',
  TIME: 'HH:mm',
  DATETIME: 'MMM dd, yyyy HH:mm',
} as const

// Restaurant Business Rules
export const RESTAURANT_RULES = {
  OPENING_HOUR: 11,           // 11 AM
  CLOSING_HOUR: 22,           // 10 PM
  MAX_PARTY_SIZE: 12,
  MIN_PARTY_SIZE: 1,
  ADVANCE_BOOKING_DAYS: 30,
  CANCELLATION_WINDOW_HOURS: 2,
} as const 