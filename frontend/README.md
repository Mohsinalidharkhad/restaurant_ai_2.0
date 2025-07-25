# Frontend - Restaurant Graph Agent

This directory contains all frontend-related code for the Restaurant Graph Agent application, built with Next.js 13+ and React.

## 📁 Directory Structure

```
frontend/
├── app/                     # Next.js 13+ App Router
│   ├── api/                 # API route handlers
│   ├── page.tsx             # Main application page
│   ├── layout.tsx           # Root layout component
│   └── globals.css          # Global styles
├── src/                     # Source code
│   ├── components/          # React components
│   │   ├── chat/            # Chat-related components
│   │   ├── layout/          # Layout components (Sidebar, etc.)
│   │   ├── ui/              # Reusable UI components
│   │   └── common/          # Common/shared components
│   ├── lib/                 # Utilities and helper functions
│   ├── hooks/               # Custom React hooks
│   ├── types/               # TypeScript type definitions
│   ├── constants/           # Application constants
│   └── styles/              # Additional stylesheets
└── README.md               # This file
```

## 🛠️ Technology Stack

- **Next.js 13+** - React framework with App Router
- **React 18** - UI library
- **TypeScript** - Type safety
- **Tailwind CSS** - Utility-first CSS framework
- **Radix UI** - Unstyled, accessible UI components
- **Lucide React** - Icon library

## 🧩 Component Organization

### Chat Components (`src/components/chat/`)
- `ChatInterface.tsx` - Main chat interface container
- `MessageBubble.tsx` - Individual message display
- `MessageList.tsx` - Message list container
- `TypingIndicator.tsx` - Typing animation component
- `WelcomeMessage.tsx` - Initial welcome display

### Layout Components (`src/components/layout/`)
- `Sidebar.tsx` - Application sidebar navigation

### UI Components (`src/components/ui/`)
- `button.tsx` - Reusable button component
- `card.tsx` - Card container component
- `input.tsx` - Form input component
- `scroll-area.tsx` - Scrollable area component

## 📋 Development Guidelines

### Component Structure
```typescript
// Component template
import { ComponentProps } from '@/src/types/components'

interface ComponentNameProps {
  // Props definition
}

export default function ComponentName({ ...props }: ComponentNameProps) {
  return (
    // JSX
  )
}
```

### Import Organization
```typescript
// 1. React/Next.js imports
import React from 'react'
import { NextPage } from 'next'

// 2. Third-party imports
import { cn } from '@/src/lib/utils'

// 3. Internal component imports
import { Button } from '@/src/components/ui/button'

// 4. Type imports
import type { ChatMessage } from '@/src/types/chat'
```

### File Naming Conventions
- **Components**: PascalCase (e.g., `ChatInterface.tsx`)
- **Utilities**: camelCase (e.g., `utils.ts`)
- **Types**: camelCase (e.g., `chatTypes.ts`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `API_ENDPOINTS.ts`)

## 🔧 Key Features

### Responsive Design
All components are built with mobile-first responsive design using Tailwind CSS.

### Accessibility
Components follow WCAG guidelines and use semantic HTML with proper ARIA attributes.

### Type Safety
Full TypeScript integration with strict type checking for all components and utilities.

### Performance
- Code splitting with Next.js dynamic imports
- Optimized re-renders with React.memo where appropriate
- Efficient state management

## 🚀 Getting Started

The frontend is integrated with the main application. To run the development server:

```bash
npm run dev
```

This will start the Next.js development server on `http://localhost:3000`.

## 📝 Adding New Components

1. Create the component in the appropriate directory
2. Add proper TypeScript types
3. Include in the appropriate index file for easier imports
4. Add to this README if it's a major component

## 🔍 Import Paths

Use the configured path aliases:
- `@/frontend/src/*` - Source directory
- `@/frontend/app/*` - App directory

Example:
```typescript
import { ChatInterface } from '@/frontend/src/components/chat/ChatInterface'
``` 