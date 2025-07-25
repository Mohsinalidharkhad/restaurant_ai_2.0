// Chat Components
export { default as ChatInterface } from './chat/ChatInterface'
export { default as MessageBubble } from './chat/MessageBubble'
export { default as MessageList } from './chat/MessageList'
export { default as TypingIndicator } from './chat/TypingIndicator'
export { default as WelcomeMessage } from './chat/WelcomeMessage'

// Layout Components
export { default as Sidebar } from './layout/Sidebar'

// UI Components
export * from './ui/button'
export * from './ui/card'
export * from './ui/input'
export * from './ui/scroll-area'

// Re-export component types
export type { ChatMessage } from '../types/chat'
export type { ComponentProps } from '../types/components' 