'use client'

import React, { useState, useRef, useEffect } from 'react'
import { nanoid } from 'nanoid'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { ScrollArea } from '@/components/ui/scroll-area'
import MessageList from '@/components/chat/MessageList'
import TypingIndicator from '@/components/chat/TypingIndicator'
import { SessionState, Message, ChatResponse, UserInfo } from '@/lib/types'
import { formatLatency, extractPhoneNumber } from '@/lib/utils'
import { Send, Loader2, Clock } from 'lucide-react'

interface ChatInterfaceProps {
  sessionState: SessionState
  onAddMessage: (message: Message) => void
  onUpdateUserInfo: (userInfo: Partial<UserInfo>) => void
  onSetProcessing: (processing: boolean) => void
  onUpdateLastRequestTime: (time: number) => void
}

interface InitializationStatus {
  initialized: boolean
  in_progress: boolean
  status_message: string
  elapsed_seconds?: number
  estimated_total_seconds?: number
  error?: string
}

export default function ChatInterface({
  sessionState,
  onAddMessage,
  onUpdateUserInfo,
  onSetProcessing,
  onUpdateLastRequestTime,
}: ChatInterfaceProps) {
  const [input, setInput] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const [isFirstRequest, setIsFirstRequest] = useState(true)
  const [initStatus, setInitStatus] = useState<InitializationStatus | null>(null)
  const [showInitProgress, setShowInitProgress] = useState(true)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [sessionState.messages, isTyping])

  // Check initialization status on component mount
  useEffect(() => {
    checkInitializationStatus()
    
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current)
      }
    }
  }, [])

  const checkInitializationStatus = async () => {
    try {
      const response = await fetch('/api/initialization-status')
      if (response.ok) {
        const status: InitializationStatus = await response.json()
        setInitStatus(status)
        
        if (status.in_progress) {
          // Start polling if initialization is in progress
          startPolling()
        } else if (status.initialized) {
          // Hide progress bar if initialization is complete
          setShowInitProgress(false)
          stopPolling()
        } else if (status.error) {
          // Stop polling on error
          stopPolling()
        }
      }
    } catch (error) {
      console.error('Failed to check initialization status:', error)
    }
  }

  const startPolling = () => {
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current)
    }
    
    pollingIntervalRef.current = setInterval(async () => {
      await checkInitializationStatus()
    }, 2000) // Poll every 2 seconds
  }

  const stopPolling = () => {
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current)
      pollingIntervalRef.current = null
    }
  }

  const getProgressPercentage = () => {
    if (!initStatus || !initStatus.elapsed_seconds || !initStatus.estimated_total_seconds) {
      return 0
    }
    return Math.min((initStatus.elapsed_seconds / initStatus.estimated_total_seconds) * 100, 95)
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!input.trim() || sessionState.processing) return

    // Don't allow sending messages while initializing
    if (initStatus?.in_progress) {
      return
    }

    const currentTime = Date.now()
    
    // Rate limiting: prevent requests within 1 second of each other
    if (currentTime - sessionState.lastRequestTime < 1000) {
      return
    }

    const userMessage: Message = {
      id: nanoid(),
      role: 'user',
      content: input.trim(),
      timestamp: currentTime,
    }

    // Extract phone number from user input if present
    const phoneNumber = extractPhoneNumber(input)
    if (phoneNumber && !sessionState.userInfo.phone) {
      onUpdateUserInfo({ phone: phoneNumber, hasPhone: true })
    }

    // Add user message immediately
    onAddMessage(userMessage)
    
    // Clear input and set processing state
    setInput('')
    onSetProcessing(true)
    onUpdateLastRequestTime(currentTime)
    setIsTyping(true)

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: input.trim(),
          thread_id: sessionState.threadId,
          phone_number: sessionState.userInfo.phone,
          user_info: sessionState.userInfo,
        }),
        // Add a longer timeout for auto-initialization (first request)
        signal: AbortSignal.timeout(180000), // 3 minutes for first request
      })

      if (!response.ok) {
        throw new Error(`Chat request failed: ${response.status}`)
      }

      const data: ChatResponse = await response.json()
      
      setIsTyping(false)
      setIsFirstRequest(false) // Mark that we've made our first request

      if (data.success && data.response) {
        const assistantMessage: Message = {
          id: nanoid(),
          role: 'assistant',
          content: data.response,
          timestamp: Date.now(),
          latency: data.latency,
        }

        onAddMessage(assistantMessage)

        // Update user info if provided
        if (data.userInfo) {
          onUpdateUserInfo(data.userInfo)
        }
      } else {
        // Error message
        const errorMessage: Message = {
          id: nanoid(),
          role: 'assistant',
          content: data.error || 'I apologize, but I encountered an error. Please try again.',
          timestamp: Date.now(),
        }
        onAddMessage(errorMessage)
      }
    } catch (error) {
      console.error('Chat error:', error)
      setIsTyping(false)
      
      let errorContent = 'I apologize, but I encountered an error. Please try again or clear the conversation to start fresh.'
      
      // Handle specific error types
      if (error instanceof Error) {
        if (error.name === 'TimeoutError' || error.message.includes('timeout')) {
          if (isFirstRequest) {
            errorContent = 'The system is taking longer than expected to initialize. This is normal for the first request. Please try again - subsequent requests will be much faster!'
          } else {
            errorContent = 'The request timed out. Please try again with a shorter message.'
          }
        } else if (error.message.includes('Failed to fetch') || error.message.includes('ECONNREFUSED')) {
          errorContent = 'Unable to connect to the restaurant system. Please ensure the backend is running and try again.'
        }
      }
      
      const errorMessage: Message = {
        id: nanoid(),
        role: 'assistant',
        content: errorContent,
        timestamp: Date.now(),
      }
      onAddMessage(errorMessage)
    } finally {
      onSetProcessing(false)
    }
  }

  const canSend = input.trim() && !sessionState.processing && !initStatus?.in_progress

  // Show initialization loading screen
  if (showInitProgress && initStatus?.in_progress) {
    return (
      <div className="flex flex-col h-full items-center justify-center p-8">
        <div className="bg-white rounded-lg shadow-lg p-8 max-w-md w-full text-center">
          <div className="mb-6">
            <Clock className="w-16 h-16 text-blue-500 mx-auto mb-4 animate-pulse" />
            <h2 className="text-xl font-semibold text-gray-800 mb-2">
              Initializing Restaurant Assistant
            </h2>
            <p className="text-gray-600">
              {initStatus.status_message}
            </p>
          </div>

          {/* Progress Bar */}
          <div className="mb-4">
            <div className="bg-gray-200 rounded-full h-3 overflow-hidden">
              <div 
                className="bg-blue-500 h-full transition-all duration-500 ease-out"
                style={{ width: `${getProgressPercentage()}%` }}
              />
            </div>
            {initStatus.elapsed_seconds && (
              <p className="text-sm text-gray-500 mt-2">
                {initStatus.elapsed_seconds}s elapsed
                {initStatus.estimated_total_seconds && 
                  ` • ~${initStatus.estimated_total_seconds}s total`
                }
              </p>
            )}
          </div>

          <div className="text-sm text-gray-500">
            <p className="mb-2">🤖 Loading AI models and knowledge base...</p>
            <p className="mb-2">🗄️ Connecting to restaurant database...</p>
            <p>⚡ This happens once per session</p>
          </div>

          <Button
            variant="outline"
            onClick={() => setShowInitProgress(false)}
            className="mt-4"
          >
            Continue to Chat
          </Button>
        </div>
      </div>
    )
  }

  // Show error state if initialization failed
  if (initStatus?.error && !initStatus.initialized) {
    return (
      <div className="flex flex-col h-full items-center justify-center p-8">
        <div className="bg-white rounded-lg shadow-lg p-8 max-w-md w-full text-center">
          <div className="text-red-500 mb-4">
            <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="text-2xl">⚠️</span>
            </div>
            <h2 className="text-xl font-semibold text-gray-800 mb-2">
              Initialization Failed
            </h2>
            <p className="text-gray-600 mb-4">
              {initStatus.status_message}
            </p>
            <Button onClick={() => window.location.reload()}>
              Reload Page
            </Button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full">
      {/* Initialization progress banner - show if still initializing but user chose to continue */}
      {initStatus?.in_progress && !showInitProgress && (
        <div className="bg-blue-50 border-b border-blue-200 p-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />
              <span className="text-sm text-blue-700">
                System initializing in background...
              </span>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowInitProgress(true)}
              className="text-blue-700 hover:text-blue-800"
            >
              Show Progress
            </Button>
          </div>
        </div>
      )}

      {/* Scrollable Messages Area */}
      <div className="flex-1 overflow-hidden">
        <ScrollArea className="h-full">
          <div className="p-6">
            <div className="space-y-4">
              <MessageList messages={sessionState.messages} />
              {isTyping && <TypingIndicator showInitializationMessage={isFirstRequest} />}
            </div>
            <div ref={messagesEndRef} />
          </div>
        </ScrollArea>
      </div>

      {/* Fixed Input Area - always visible at bottom */}
      <div className="flex-shrink-0 border-t border-gray-200 bg-white p-4">
        <form onSubmit={handleSubmit} className="flex gap-2">
          <Input
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={
              initStatus?.in_progress 
                ? "System initializing... please wait"
                : "Type your message here..."
            }
            disabled={sessionState.processing || initStatus?.in_progress}
            className="flex-1"
            autoFocus={!initStatus?.in_progress}
          />
          <Button
            type="submit"
            disabled={!canSend}
            className="px-6"
          >
            {sessionState.processing ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Send className="w-4 h-4" />
            )}
          </Button>
        </form>
        
        {sessionState.processing && (
          <p className="text-sm text-gray-500 mt-2 text-center">
            ⏳ Please wait for the current message to be processed...
          </p>
        )}
        
        {initStatus?.in_progress && (
          <p className="text-sm text-blue-600 mt-2 text-center">
            🔄 System is initializing, please wait before sending messages...
          </p>
        )}
        
        {!canSend && input.trim() && !sessionState.processing && !initStatus?.in_progress && (
          <p className="text-sm text-gray-500 mt-2 text-center">
            ⏱️ Please wait a moment before sending another message...
          </p>
        )}
      </div>
    </div>
  )
} 