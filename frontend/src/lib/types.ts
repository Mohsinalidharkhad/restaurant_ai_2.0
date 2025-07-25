export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: number
  latency?: number
}

export interface UserInfo {
  name?: string
  phone?: string
  isRegistered: boolean
  hasPhone: boolean
}

export interface SessionState {
  threadId: string
  messages: Message[]
  userInfo: UserInfo
  processing: boolean
  lastRequestTime: number
}

export interface ChatResponse {
  success: boolean
  response?: string
  latency?: number
  error?: string
  userInfo?: UserInfo
}

export interface BackendConfig {
  configurable: {
    phone_number?: string
    thread_id: string
  }
}

export interface GraphState {
  messages: any[]
  is_registered: boolean
  phone_number?: string
  dialog_state: string[]
}

export interface ReservationDetails {
  reservation_id?: string
  customer_name?: string
  phone_number?: string
  pax?: number
  date?: string
  time?: string
  status?: string
}

export interface ApiError {
  error: string
  details?: string
} 