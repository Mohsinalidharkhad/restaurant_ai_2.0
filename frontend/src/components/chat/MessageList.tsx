'use client'

import React from 'react'
import { Message } from '@/lib/types'
import MessageBubble from './MessageBubble'

interface MessageListProps {
  messages: Message[]
}

export default function MessageList({ messages }: MessageListProps) {
  return (
    <>
      {messages.map((message) => (
        <MessageBubble key={message.id} message={message} />
      ))}
    </>
  )
} 