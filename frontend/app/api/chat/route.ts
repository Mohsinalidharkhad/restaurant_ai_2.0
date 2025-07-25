import { NextRequest, NextResponse } from 'next/server'

interface ChatRequest {
  message: string
  thread_id: string
  phone_number?: string
  user_info?: {
    name?: string
    phone?: string
    isRegistered: boolean
    hasPhone: boolean
  }
}

interface UserInfoFromMessage {
  name?: string
  phone?: string
  isRegistered?: boolean
  hasPhone?: boolean
}

export async function POST(request: NextRequest) {
  const startTime = Date.now()

  try {
    const { message, thread_id, phone_number, user_info }: ChatRequest = await request.json()

    console.log('[Frontend API] Received request:', { message, thread_id, phone_number, user_info })

    if (!message || !thread_id) {
      return NextResponse.json({
        success: false,
        error: 'Message and thread_id are required',
      })
    }

    // Call the Python backend
    const backendPayload = {
      message,
      config: {
        configurable: {
          thread_id,
          phone_number: phone_number || null,
        },
      },
      user_info,
    }

    console.log('[Frontend API] Sending to backend:', JSON.stringify(backendPayload, null, 2))
    console.log('[Frontend API] Making fetch request at:', new Date().toISOString())

    const backendResponse = await fetch(`http://127.0.0.1:8000/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(backendPayload),
      // Remove timeout to allow auto-initialization to complete
      // Auto-initialization can take up to 2 minutes on first request
    })

    console.log('[Frontend API] Received response at:', new Date().toISOString())
    console.log('[Frontend API] Backend response status:', backendResponse.status)
    console.log('[Frontend API] Backend response headers:', Object.fromEntries(backendResponse.headers.entries()))
    console.log('[Frontend API] Response content-length:', backendResponse.headers.get('content-length'))
    console.log('[Frontend API] Response content-type:', backendResponse.headers.get('content-type'))

    if (!backendResponse.ok) {
      const errorText = await backendResponse.text()
      console.error('[Frontend API] Backend error response:', errorText)
      throw new Error(`Backend request failed: ${backendResponse.status} - ${errorText}`)
    }

    let data
    try {
      data = await backendResponse.json()
      console.log('[Frontend API] Backend response data:', JSON.stringify(data, null, 2))
    } catch (parseError) {
      console.error('[Frontend API] Failed to parse backend response as JSON:', parseError)
      const responseText = await backendResponse.text()
      console.error('[Frontend API] Raw backend response:', responseText)
      throw new Error('Backend returned invalid JSON response')
    }
    
    const endTime = Date.now()
    const latency = (endTime - startTime) / 1000

    // Validate backend response format
    if (typeof data !== 'object' || data === null) {
      console.error('[Frontend API] Backend response is not an object:', data)
      throw new Error('Backend returned invalid response format')
    }

    if (!data.hasOwnProperty('success')) {
      console.error('[Frontend API] Backend response missing success field:', data)
      throw new Error('Backend response missing required fields')
    }

    // Check if backend indicates failure
    if (data.success === false) {
      console.error('[Frontend API] Backend reported failure:', data.error)
      return NextResponse.json({
        success: false,
        error: data.error || 'Backend processing failed',
        latency,
      })
    }

    // Extract user information from the response
    let updatedUserInfo: UserInfoFromMessage = {}
    
    if (data.user_info) {
      updatedUserInfo = {
        name: data.user_info.name,
        phone: data.user_info.phone || phone_number,
        isRegistered: data.user_info.is_registered || false,
        hasPhone: !!(data.user_info.phone || phone_number),
      }
    } else if (data.is_registered !== undefined) {
      updatedUserInfo = {
        isRegistered: data.is_registered,
        hasPhone: !!(phone_number || data.phone_number),
        phone: phone_number || data.phone_number,
      }
    }

    const responsePayload = {
      success: true,
      response: data.response || data.message || 'No response from backend',
      latency,
      userInfo: Object.keys(updatedUserInfo).length > 0 ? updatedUserInfo : undefined,
    }

    console.log('[Frontend API] Sending response:', JSON.stringify(responsePayload, null, 2))
    
    return NextResponse.json(responsePayload)

  } catch (error) {
    console.error('[Frontend API] Error:', error)
    console.error('[Frontend API] Error stack:', error instanceof Error ? error.stack : 'No stack trace')
    
    const endTime = Date.now()
    const latency = (endTime - startTime) / 1000

    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error occurred',
      latency,
    })
  }
} 