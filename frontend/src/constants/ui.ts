// Animation Durations (in milliseconds)
export const ANIMATION_DURATION = {
  FAST: 150,
  NORMAL: 300,
  SLOW: 500,
  TYPING: 100,
} as const

// Breakpoints (matching Tailwind CSS)
export const BREAKPOINTS = {
  SM: 640,
  MD: 768,
  LG: 1024,
  XL: 1280,
  '2XL': 1536,
} as const

// Z-Index Values
export const Z_INDEX = {
  DROPDOWN: 1000,
  STICKY: 1020,
  FIXED: 1030,
  MODAL_BACKDROP: 1040,
  MODAL: 1050,
  POPOVER: 1060,
  TOOLTIP: 1070,
  TOAST: 1080,
} as const

// Component Sizes
export const COMPONENT_SIZES = {
  BUTTON: {
    SM: 'h-8 px-3 text-sm',
    DEFAULT: 'h-10 px-4 py-2',
    LG: 'h-11 px-8',
    ICON: 'h-10 w-10',
  },
  INPUT: {
    SM: 'h-8 px-3 text-sm',
    DEFAULT: 'h-10 px-3 py-2',
    LG: 'h-11 px-4',
  },
  AVATAR: {
    SM: 'h-8 w-8',
    DEFAULT: 'h-10 w-10',
    LG: 'h-12 w-12',
    XL: 'h-16 w-16',
  },
} as const

// Color Palette (CSS Custom Properties)
export const COLORS = {
  PRIMARY: 'hsl(var(--primary))',
  SECONDARY: 'hsl(var(--secondary))',
  ACCENT: 'hsl(var(--accent))',
  DESTRUCTIVE: 'hsl(var(--destructive))',
  MUTED: 'hsl(var(--muted))',
  BORDER: 'hsl(var(--border))',
  BACKGROUND: 'hsl(var(--background))',
  FOREGROUND: 'hsl(var(--foreground))',
} as const

// Chat-specific UI Constants
export const CHAT_UI = {
  MAX_MESSAGE_LENGTH: 2000,
  TYPING_INDICATOR_DELAY: 300,
  MESSAGE_ANIMATION_DELAY: 100,
  SCROLL_THRESHOLD: 100,
  MAX_VISIBLE_MESSAGES: 50,
} as const

// Loading States
export const LOADING_MESSAGES = [
  'Thinking...',
  'Processing your request...',
  'Searching menu...',
  'Checking availability...',
  'Preparing response...',
] as const

// Error Messages
export const ERROR_MESSAGES = {
  NETWORK: 'Network error. Please check your connection.',
  TIMEOUT: 'Request timed out. Please try again.',
  SERVER: 'Server error. Please try again later.',
  VALIDATION: 'Please check your input and try again.',
  UNAUTHORIZED: 'You are not authorized to perform this action.',
  NOT_FOUND: 'The requested resource was not found.',
  GENERIC: 'Something went wrong. Please try again.',
} as const 