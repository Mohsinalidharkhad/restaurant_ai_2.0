'use client'

import React, { useState, useEffect } from 'react'

interface TypingIndicatorProps {
  showInitializationMessage?: boolean
}

export default function TypingIndicator({ showInitializationMessage = false }: TypingIndicatorProps) {
  const [dots, setDots] = useState('')
  const [showInitMessage, setShowInitMessage] = useState(false)

  useEffect(() => {
    const interval = setInterval(() => {
      setDots(prev => {
        if (prev === '...') return ''
        return prev + '.'
      })
    }, 500)

    // Show initialization message after 5 seconds
    const initTimeout = setTimeout(() => {
      if (showInitializationMessage) {
        setShowInitMessage(true)
      }
    }, 5000)

    return () => {
      clearInterval(interval)
      clearTimeout(initTimeout)
    }
  }, [showInitializationMessage])

  return (
    <div className="flex items-center space-x-2 p-4">
      <div className="flex space-x-1">
        <div className="w-2 h-2 bg-green-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
        <div className="w-2 h-2 bg-green-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
        <div className="w-2 h-2 bg-green-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
      </div>
      <span className="text-gray-500 text-sm">
        {showInitMessage ? (
          <>
            Initializing restaurant system{dots}
            <br />
            <span className="text-xs text-gray-400">This may take up to 2 minutes on first use</span>
          </>
        ) : (
          <>AI Waiter is thinking{dots}</>
        )}
      </span>
    </div>
  )
} 