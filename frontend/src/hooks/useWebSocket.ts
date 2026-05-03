import { useEffect, useRef, useState, useCallback } from 'react'

export type ConnectionState = 'connecting' | 'connected' | 'reconnecting' | 'disconnected'

interface UseWebSocketOptions {
  onMessage: (data: string) => void
}

interface UseWebSocketReturn {
  sendMessage: (msg: string) => void
  connectionState: ConnectionState
  reconnect: () => void
}

const RECONNECT_DELAYS = [1000, 2000, 4000, 8000, 16000, 30000] // exponential back-off

export const useWebSocket = (url: string, options: UseWebSocketOptions): UseWebSocketReturn => {
  const socketRef = useRef<WebSocket | null>(null)
  const [connectionState, setConnectionState] = useState<ConnectionState>('connecting')
  const reconnectAttemptRef = useRef(0)
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const intentionalCloseRef = useRef(false)
  const onMessageRef = useRef(options.onMessage)
  onMessageRef.current = options.onMessage

  const clearReconnectTimer = () => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current)
      reconnectTimerRef.current = null
    }
  }

  const connect = useCallback(() => {
    clearReconnectTimer()
    if (socketRef.current && socketRef.current.readyState < WebSocket.CLOSING) {
      socketRef.current.close()
    }

    setConnectionState(reconnectAttemptRef.current > 0 ? 'reconnecting' : 'connecting')
    const ws = new WebSocket(url)
    socketRef.current = ws

    ws.onopen = () => {
      reconnectAttemptRef.current = 0
      setConnectionState('connected')
    }

    ws.onmessage = (event: MessageEvent) => {
      const data = typeof event.data === 'string' ? event.data : String(event.data)
      onMessageRef.current(data)
    }

    ws.onclose = () => {
      if (intentionalCloseRef.current) return
      setConnectionState('reconnecting')
      const delay = RECONNECT_DELAYS[Math.min(reconnectAttemptRef.current, RECONNECT_DELAYS.length - 1)]
      reconnectAttemptRef.current += 1
      reconnectTimerRef.current = setTimeout(connect, delay)
    }

    ws.onerror = () => {
      // onclose fires after onerror; handled there
    }
  }, [url])

  useEffect(() => {
    intentionalCloseRef.current = false
    connect()
    return () => {
      intentionalCloseRef.current = true
      clearReconnectTimer()
      socketRef.current?.close()
    }
  }, [connect])

  const sendMessage = useCallback((msg: string) => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(msg)
    } else {
      console.warn('WebSocket not open; message dropped:', msg)
    }
  }, [])

  const reconnect = useCallback(() => {
    reconnectAttemptRef.current = 0
    connect()
  }, [connect])

  return { sendMessage, connectionState, reconnect }
}

