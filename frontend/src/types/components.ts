import { ReactNode, ComponentProps as ReactComponentProps } from 'react'

// Base component props
export interface BaseComponentProps {
  className?: string
  children?: ReactNode
}

// Common component props extending HTML elements
export type ComponentProps<T = any> = BaseComponentProps & T

// Button variant types
export type ButtonVariant = 'default' | 'destructive' | 'outline' | 'secondary' | 'ghost' | 'link'
export type ButtonSize = 'default' | 'sm' | 'lg' | 'icon'

// Card types
export interface CardProps extends BaseComponentProps {
  variant?: 'default' | 'outline' | 'ghost'
}

// Input types
export interface InputProps extends ReactComponentProps<'input'> {
  error?: string
  label?: string
}

// Layout types
export interface SidebarProps extends BaseComponentProps {
  isOpen?: boolean
  onToggle?: () => void
  navigation?: NavigationItem[]
}

export interface NavigationItem {
  label: string
  href?: string
  icon?: ReactNode
  active?: boolean
  onClick?: () => void
  children?: NavigationItem[]
}

// Loading states
export interface LoadingState {
  isLoading: boolean
  message?: string
}

// Error states
export interface ErrorState {
  hasError: boolean
  message?: string
  details?: any
}

// API response types
export interface ApiResponse<T = any> {
  success: boolean
  data?: T
  error?: string
  message?: string
}

// Theme types
export type Theme = 'light' | 'dark' | 'system'

// Screen size types
export type ScreenSize = 'sm' | 'md' | 'lg' | 'xl' | '2xl'

// Generic event handler types
export type EventHandler<T = any> = (event: T) => void
export type AsyncEventHandler<T = any> = (event: T) => Promise<void> 