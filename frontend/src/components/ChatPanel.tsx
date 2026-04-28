import React, { useState, useEffect, useRef } from 'react'
import { useWebSocket } from '../hooks/useWebSocket'

// TODO Stage 3: replace stub messages with AgentState-driven streaming tokens

interface Message {
  role: 'user' | 'assistant'
  content: string
}

const ChatPanel: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)

  const { sendMessage, lastMessage, readyState } = useWebSocket(
    `ws://${window.location.hostname}:8000/ws/chat`,
  )

  useEffect(() => {
    if (lastMessage) {
      setMessages((prev) => [...prev, { role: 'assistant', content: lastMessage }])
    }
  }, [lastMessage])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSend = () => {
    const text = input.trim()
    if (!text || readyState !== WebSocket.OPEN) return
    setMessages((prev) => [...prev, { role: 'user', content: text }])
    sendMessage(text)
    setInput('')
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="flex flex-col h-full">
      <header className="px-4 py-2 border-b border-gray-700 font-semibold text-sm">
        Agentic ERP Copilot
        <span
          className={`ml-2 text-xs ${readyState === WebSocket.OPEN ? 'text-green-400' : 'text-red-400'}`}
        >
          {readyState === WebSocket.OPEN ? '● connected' : '● disconnected'}
        </span>
      </header>

      <div className="flex-1 overflow-y-auto px-4 py-2 space-y-2">
        {messages.map((m, i) => (
          <div
            key={i}
            className={`rounded-lg px-3 py-2 text-sm max-w-prose ${
              m.role === 'user'
                ? 'bg-blue-700 ml-auto'
                : 'bg-gray-700'
            }`}
          >
            {m.content}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="px-4 py-2 border-t border-gray-700 flex gap-2">
        <textarea
          className="flex-1 rounded bg-gray-800 text-sm text-white px-3 py-2 resize-none focus:outline-none"
          rows={2}
          placeholder="Ask about supply chain disruptions, reorder points, VRP routes…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
        />
        <button
          className="bg-blue-600 hover:bg-blue-500 rounded px-4 text-sm font-semibold"
          onClick={handleSend}
        >
          Send
        </button>
      </div>
    </div>
  )
}

export default ChatPanel
