// Chat message types
export interface ChatMessage {
  id: string
  content: string
  role: 'user' | 'assistant'
  timestamp: Date
  metadata?: {
    toolCalls?: ToolCall[]
    thinking?: string
    citations?: string[]
  }
}

// Tool call types
export interface ToolCall {
  id: string
  name: string
  args: Record<string, any>
  result?: any
}

// Chat interface types
export interface ChatInterfaceProps {
  initialMessages?: ChatMessage[]
  onMessageSend?: (message: string) => void
  isLoading?: boolean
  className?: string
}

// Message display types
export interface MessageBubbleProps {
  message: ChatMessage
  isLatest?: boolean
  className?: string
}

// Welcome message types
export interface WelcomeMessageProps {
  restaurantName?: string
  features?: string[]
  className?: string
}

// Typing indicator types
export interface TypingIndicatorProps {
  isVisible: boolean
  message?: string
  className?: string
}

// Chat state types
export interface ChatState {
  messages: ChatMessage[]
  isLoading: boolean
  error: string | null
  initialized: boolean
}

// Chat actions
export type ChatAction =
  | { type: 'ADD_MESSAGE'; payload: ChatMessage }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'SET_INITIALIZED'; payload: boolean }
  | { type: 'CLEAR_MESSAGES' } 