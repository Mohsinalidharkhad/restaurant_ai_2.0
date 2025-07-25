import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatTime(timestamp: number): string {
  return new Date(timestamp).toLocaleTimeString('en-US', {
    hour12: true,
    hour: 'numeric',
    minute: '2-digit'
  })
}

export function formatLatency(latency: number): string {
  if (latency < 1.0) {
    return `⚡ ${Math.round(latency * 1000)}ms`
  }
  return `⚡ ${latency.toFixed(2)}s`
}

export function extractPhoneNumber(text: string): string | null {
  const phoneRegex = /\b(\d{10})\b/
  const match = text.match(phoneRegex)
  return match ? match[1] : null
}

export function isValidPhoneNumber(phone: string): boolean {
  return /^\d{10}$/.test(phone)
}

export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout | null = null
  
  return (...args: Parameters<T>) => {
    if (timeout) clearTimeout(timeout)
    timeout = setTimeout(() => func(...args), wait)
  }
}

export async function apiCall(endpoint: string, options: RequestInit = {}): Promise<any> {
  const baseUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'
  
  const defaultOptions: RequestInit = {
    headers: {
      'Content-Type': 'application/json',
    },
  }
  
  const mergedOptions = {
    ...defaultOptions,
    ...options,
    headers: {
      ...defaultOptions.headers,
      ...options.headers,
    },
  }
  
  const response = await fetch(`${baseUrl}${endpoint}`, mergedOptions)
  
  if (!response.ok) {
    throw new Error(`API call failed: ${response.status} ${response.statusText}`)
  }
  
  return response.json()
} 