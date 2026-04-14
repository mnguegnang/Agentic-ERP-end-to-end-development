import { useEffect, useRef, useState, useCallback } from 'react'

interface UseWebSocketReturn {
  sendMessage: (msg: string) => void
  lastMessage: string | null
  readyState: number
}

// TODO Stage 3: add token-streaming decoder and reconnection back-off

export const useWebSocket = (url: string): UseWebSocketReturn => {
  const socketRef = useRef<WebSocket | null>(null)
  const [lastMessage, setLastMessage] = useState<string | null>(null)
  const [readyState, setReadyState] = useState<number>(WebSocket.CONNECTING)

  useEffect(() => {
    const ws = new WebSocket(url)
    socketRef.current = ws

    ws.onopen = () => setReadyState(WebSocket.OPEN)
    ws.onclose = () => setReadyState(WebSocket.CLOSED)
    ws.onerror = () => setReadyState(WebSocket.CLOSED)
    ws.onmessage = (event: MessageEvent) => {
      setLastMessage(typeof event.data === 'string' ? event.data : String(event.data))
    }

    return () => ws.close()
  }, [url])

  const sendMessage = useCallback((msg: string) => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(msg)
    }
  }, [])

  return { sendMessage, lastMessage, readyState }
}
