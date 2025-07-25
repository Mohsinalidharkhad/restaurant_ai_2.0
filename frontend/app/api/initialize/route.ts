import { NextRequest, NextResponse } from 'next/server'

export async function POST(request: NextRequest) {
  try {
    const { thread_id } = await request.json()

    // Initialize the Python backend system
    const response = await fetch(`http://127.0.0.1:8000/api/initialize`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        thread_id,
      }),
    })

    if (response.ok) {
      const data = await response.json()
      return NextResponse.json({ success: true, data })
    } else {
      console.warn('Backend initialization failed, but continuing...')
      return NextResponse.json({ success: false, message: 'Backend initialization failed' })
    }
  } catch (error) {
    console.error('Initialization error:', error)
    return NextResponse.json({ success: false, error: 'Initialization failed' })
  }
} 