import React, { useState, useEffect, useRef } from 'react'
import { useWebSocket } from '../hooks/useWebSocket'

interface Message {
  role: 'user' | 'assistant'
  content: string
  humanApprovalRequired?: boolean
  decisionId?: string
  approvalStatus?: 'pending' | 'approved' | 'rejected'
}

const API_BASE = `http://${window.location.hostname}:8000`

const ChatPanel: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)

  const { sendMessage, lastMessage, readyState } = useWebSocket(
    `ws://${window.location.hostname}:8000/ws/chat`,
  )

  useEffect(() => {
    if (lastMessage) {
      try {
        const parsed = JSON.parse(lastMessage)
        const text = parsed.content ?? lastMessage
        const needsApproval = parsed.human_approval_required === true
        const decisionId = parsed.decision_id ?? undefined
        setMessages((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: text,
            humanApprovalRequired: needsApproval,
            decisionId,
            approvalStatus: needsApproval ? 'pending' : undefined,
          },
        ])
      } catch {
        setMessages((prev) => [...prev, { role: 'assistant', content: lastMessage }])
      }
    }
  }, [lastMessage])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSend = () => {
    const text = input.trim()
    if (!text || readyState !== WebSocket.OPEN) return
    setMessages((prev) => [...prev, { role: 'user', content: text }])
    sendMessage(JSON.stringify({ role: 'user', content: text }))
    setInput('')
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleApproval = async (msgIndex: number, approved: boolean) => {
    const msg = messages[msgIndex]
    if (!msg.decisionId) return

    try {
      const res = await fetch(`${API_BASE}/api/approve/${msg.decisionId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          approved,
          approved_by: 'supply-chain-manager',
          reason: approved ? 'Reviewed and approved' : 'Rejected by manager',
        }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const record = await res.json()
      const newStatus: 'approved' | 'rejected' = record.status

      setMessages((prev) =>
        prev.map((m, i) =>
          i === msgIndex ? { ...m, approvalStatus: newStatus } : m,
        ),
      )
    } catch (err) {
      console.error('Approval request failed:', err)
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
              m.role === 'user' ? 'bg-blue-700 ml-auto' : 'bg-gray-700'
            }`}
          >
            {m.humanApprovalRequired && (
              <div className="mb-2 rounded bg-yellow-500 text-black font-bold px-2 py-1 text-xs">
                ⚠️ HUMAN APPROVAL REQUIRED — cost exceeds $10,000 threshold
              </div>
            )}

            {m.content}

            {/* Approval action panel — shown only when pending */}
            {m.humanApprovalRequired && m.decisionId && m.approvalStatus === 'pending' && (
              <div className="mt-3 flex gap-2">
                <button
                  onClick={() => handleApproval(i, true)}
                  className="rounded bg-green-600 hover:bg-green-500 px-3 py-1 text-white text-xs font-semibold"
                >
                  ✅ Approve
                </button>
                <button
                  onClick={() => handleApproval(i, false)}
                  className="rounded bg-red-600 hover:bg-red-500 px-3 py-1 text-white text-xs font-semibold"
                >
                  ❌ Reject
                </button>
                <span className="text-xs text-gray-400 self-center">
                  ID: {m.decisionId.slice(0, 8)}…
                </span>
              </div>
            )}

            {/* Outcome badge — shown after manager acts */}
            {m.approvalStatus === 'approved' && (
              <div className="mt-2 rounded bg-green-700 text-white px-2 py-1 text-xs font-semibold">
                ✅ Approved by supply-chain manager — execution authorised.
              </div>
            )}
            {m.approvalStatus === 'rejected' && (
              <div className="mt-2 rounded bg-red-700 text-white px-2 py-1 text-xs font-semibold">
                ❌ Rejected by supply-chain manager — execution blocked.
              </div>
            )}
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
