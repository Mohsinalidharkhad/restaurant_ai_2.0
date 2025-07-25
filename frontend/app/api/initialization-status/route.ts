import { NextRequest, NextResponse } from 'next/server'

export async function GET(request: NextRequest) {
  try {
    // Proxy the initialization status from the backend
    const response = await fetch(`http://127.0.0.1:8000/api/initialization-status`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    })

    if (response.ok) {
      const data = await response.json()
      return NextResponse.json(data)
    } else {
      console.warn('Backend initialization status check failed')
      return NextResponse.json({ 
        initialized: false, 
        in_progress: false, 
        status_message: 'Unable to check initialization status',
        error: 'Backend unavailable'
      })
    }
  } catch (error) {
    console.error('Initialization status error:', error)
    return NextResponse.json({ 
      initialized: false, 
      in_progress: false, 
      status_message: 'Unable to connect to backend',
      error: 'Connection failed'
    })
  }
} 