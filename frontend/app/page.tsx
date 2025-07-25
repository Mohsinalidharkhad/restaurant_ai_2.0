'use client'

import React, { useState, useEffect } from 'react'
import { nanoid } from 'nanoid'
import ChatInterface from '@/components/chat/ChatInterface'
import Sidebar from '@/components/layout/Sidebar'
import WelcomeMessage from '@/components/chat/WelcomeMessage'
import { SessionState, UserInfo, Message } from '@/lib/types'

export default function Home() {
  const [sessionState, setSessionState] = useState<SessionState>({
    threadId: nanoid(),
    messages: [],
    userInfo: {
      isRegistered: false,
      hasPhone: false,
    },
    processing: false,
    lastRequestTime: 0,
  })

  const [initialized, setInitialized] = useState(false)

  useEffect(() => {
    // Initialize the session on first load
    const initializeSession = async () => {
      try {
        // Pre-warm the backend system similar to Streamlit
        const response = await fetch('/api/initialize', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            thread_id: sessionState.threadId,
          }),
        })

        if (response.ok) {
          console.log('✅ System pre-warmed and ready!')
        } else {
          console.warn('⚠️ System pre-warming failed, but continuing...')
        }
      } catch (error) {
        console.warn('⚠️ System pre-warming failed:', error)
      }

      // Add initial welcome message
      const initialMessage: Message = {
        id: nanoid(),
        role: 'assistant',
        content: "Hello! Welcome to Silk Route Eatery. I'm here to help you explore our menu and answer any questions about our dishes. What would you like to know about our food today?",
        timestamp: Date.now(),
      }

      setSessionState(prev => ({
        ...prev,
        messages: [initialMessage]
      }))

      setInitialized(true)
    }

    initializeSession()
  }, [sessionState.threadId])

  const updateUserInfo = (userInfo: Partial<UserInfo>) => {
    setSessionState(prev => ({
      ...prev,
      userInfo: { ...prev.userInfo, ...userInfo }
    }))
  }

  const addMessage = (message: Message) => {
    setSessionState(prev => ({
      ...prev,
      messages: [...prev.messages, message]
    }))
  }

  const setProcessing = (processing: boolean) => {
    setSessionState(prev => ({
      ...prev,
      processing
    }))
  }

  const updateLastRequestTime = (time: number) => {
    setSessionState(prev => ({
      ...prev,
      lastRequestTime: time
    }))
  }

  if (!initialized) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
          <h2 className="text-xl font-semibold mb-2">🔄 Starting Restaurant Assistant</h2>
          <p className="text-gray-600">Pre-warming system for faster responses...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-screen bg-white overflow-hidden">
      {/* Fixed Sidebar - doesn't scroll */}
      <div className="flex-shrink-0">
        <Sidebar userInfo={sessionState.userInfo} />
      </div>
      
      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-h-0">
        {/* Fixed Header - always visible */}
        <header className="flex-shrink-0 bg-white border-b border-gray-200 px-6 py-4 z-10">
          <div className="text-center">
            <h1 className="text-3xl font-bold text-green-700 mb-2">🍽️ Silk Route Eatery</h1>
            <p className="text-gray-600">Your AI-powered dining companion for personalized menu recommendations and reservations management</p>
          </div>
        </header>

        {/* Scrollable Content Area */}
        <div className="flex-1 overflow-hidden">
          {/* Welcome Message - only show when needed */}
          {sessionState.messages.length === 1 && (
            <WelcomeMessage />
          )}

          {/* Chat Interface - will handle its own scrolling */}
          <ChatInterface
            sessionState={sessionState}
            onAddMessage={addMessage}
            onUpdateUserInfo={updateUserInfo}
            onSetProcessing={setProcessing}
            onUpdateLastRequestTime={updateLastRequestTime}
          />
        </div>
      </div>
    </div>
  )
} 