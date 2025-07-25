'use client'

import React from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Message } from '@/lib/types'
import { formatTime, formatLatency } from '@/lib/utils'
import { User, Bot } from 'lucide-react'

interface MessageBubbleProps {
  message: Message
}

export default function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user'

  return (
    <div className={`chat-message message-fade-in ${isUser ? 'user-message' : 'assistant-message'}`}>
      <div className="flex items-start gap-3">
        <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
          isUser ? 'bg-blue-500 text-white' : 'bg-green-500 text-white'
        }`}>
          {isUser ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
        </div>
        
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-sm font-medium text-gray-700">
              {isUser ? 'You' : 'AI Waiter'}
            </span>
            <span className="text-xs text-gray-500">
              {formatTime(message.timestamp)}
            </span>
          </div>
          
          <div className="prose prose-sm max-w-none">
            {isUser ? (
              // For user messages, keep simple text formatting
              <p className="text-gray-900 leading-relaxed whitespace-pre-wrap">
                {message.content}
              </p>
            ) : (
              // For assistant messages, render markdown with table support
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                className="text-gray-900 leading-relaxed"
                components={{
                  // Custom table styling
                  table: ({ children }) => (
                    <div className="overflow-x-auto my-4">
                      <table className="min-w-full border-collapse border border-gray-300 text-sm">
                        {children}
                      </table>
                    </div>
                  ),
                  thead: ({ children }) => (
                    <thead className="bg-gray-50">{children}</thead>
                  ),
                  th: ({ children }) => (
                    <th className="border border-gray-300 px-3 py-2 text-left font-semibold text-gray-700">
                      {children}
                    </th>
                  ),
                  td: ({ children }) => (
                    <td className="border border-gray-300 px-3 py-2 text-gray-900">
                      {children}
                    </td>
                  ),
                  // Custom paragraph styling
                  p: ({ children }) => (
                    <p className="mb-2 last:mb-0">{children}</p>
                  ),
                  // Custom list styling
                  ul: ({ children }) => (
                    <ul className="list-disc list-inside mb-2 space-y-1">{children}</ul>
                  ),
                  ol: ({ children }) => (
                    <ol className="list-decimal list-inside mb-2 space-y-1">{children}</ol>
                  ),
                  // Custom strong/bold styling
                  strong: ({ children }) => (
                    <strong className="font-semibold text-gray-900">{children}</strong>
                  ),
                  // Custom code styling
                  code: ({ children }) => (
                    <code className="bg-gray-100 px-1 py-0.5 rounded text-sm font-mono">
                      {children}
                    </code>
                  ),
                }}
              >
                {message.content}
              </ReactMarkdown>
            )}
          </div>
          
          {message.latency && !isUser && (
            <div className="mt-2">
              <span className="latency-badge">
                {formatLatency(message.latency)}
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
} 