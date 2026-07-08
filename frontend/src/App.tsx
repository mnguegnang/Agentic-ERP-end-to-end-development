import React, { useState, useCallback, useEffect, useRef } from 'react'
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels'
import ChatPanel from './components/ChatPanel'
import GraphViewer from './components/GraphViewer'
import SolverResults from './components/SolverResults'
import { useWebSocket } from './hooks/useWebSocket'
import type { Message, SolverOutput, KGSubgraph, RagDocument } from './types'

// ── Toast types ────────────────────────────────────────────────────────────
interface Toast { id: number; message: string; type: 'info' | 'success' | 'warning' }

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? `http://${window.location.hostname}:8000`
const WS_URL = import.meta.env.VITE_WS_BASE_URL ?? `ws://${window.location.hostname}:8000`

const App: React.FC = () => {
  // ── Theme (improvement #6) ──────────────────────────────────────────
  const [isDark, setIsDark] = useState(() => {
    return localStorage.getItem('erp-theme') !== 'light'
  })
  useEffect(() => {
    localStorage.setItem('erp-theme', isDark ? 'dark' : 'light')
    document.documentElement.classList.toggle('light-theme', !isDark)
  }, [isDark])

  // ── Session persistence (improvement #9) ───────────────────────────
  const [messages, setMessages] = useState<Message[]>(() => {
    try {
      const saved = localStorage.getItem('erp-chat-history')
      if (!saved) return []
      const parsed = JSON.parse(saved) as Array<Message & { timestamp: string }>
      return parsed.map((m) => ({ ...m, timestamp: new Date(m.timestamp) }))
    } catch { return [] }
  })
  useEffect(() => {
    // Keep last 100 messages in localStorage
    localStorage.setItem('erp-chat-history', JSON.stringify(messages.slice(-100)))
  }, [messages])

  const [isLoading, setIsLoading] = useState(false)
  const [solverResult, setSolverResult] = useState<SolverOutput | null>(null)
  const [solverHistory, setSolverHistory] = useState<SolverOutput[]>([])  // improvement #8
  const [kgSubgraph, setKgSubgraph] = useState<KGSubgraph | null>(null)
  const [toasts, setToasts] = useState<Toast[]>([])
  const toastIdRef = useRef(0)

  const showToast = useCallback((message: string, type: Toast['type'] = 'info') => {
    const id = ++toastIdRef.current
    setToasts((prev) => [...prev, { id, message, type }])
    setTimeout(() => setToasts((prev) => prev.filter((t) => t.id !== id)), 4000)
  }, [])

  const handleIncoming = useCallback((raw: string) => {
    setIsLoading(false)
    try {
      const parsed = JSON.parse(raw) as {
        role?: string
        content?: string
        intent?: string
        intent_confidence?: number
        tool_used?: string | null
        solver_result?: Record<string, unknown> | null
        rag_documents?: RagDocument[] | null
        human_approval_required?: boolean
        decision_id?: string | null
        error?: string | null
        kg_subgraph?: KGSubgraph | null
      }

      const content = parsed.content ?? raw
      const msg: Message = {
        id: `${Date.now()}-${Math.random()}`,
        role: 'assistant',
        content,
        intent: parsed.intent,
        intentConfidence: parsed.intent_confidence,
        toolUsed: parsed.tool_used ?? null,
        solverResult: parsed.solver_result ?? null,
        ragDocuments: parsed.rag_documents ?? null,
        humanApprovalRequired: parsed.human_approval_required === true,
        decisionId: parsed.decision_id ?? undefined,
        approvalStatus: parsed.human_approval_required ? 'pending' : undefined,
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, msg])

      // Update side panels
      if (parsed.solver_result && parsed.intent) {
        const sr = parsed.solver_result as Record<string, unknown>
        const newResult: SolverOutput = {
          solver: parsed.intent,
          status: (sr.status as string) ?? 'UNKNOWN',
          objective: (sr.total_cost as number) ?? (sr.objective as number) ?? undefined,
          raw: sr,
        }
        setSolverResult(newResult)
        setSolverHistory((prev) => [newResult, ...prev].slice(0, 20))  // keep last 20
      }
      if (parsed.kg_subgraph) {
        setKgSubgraph(parsed.kg_subgraph)
      }
    } catch {
      // fallback: show raw text as an assistant message
      setMessages((prev) => [
        ...prev,
        {
          id: `${Date.now()}-${Math.random()}`,
          role: 'assistant',
          content: raw,
          timestamp: new Date(),
        },
      ])
    }
  }, [])

  const { sendMessage, connectionState, reconnect } = useWebSocket(`${WS_URL}/ws/chat`, {
    onMessage: handleIncoming,
  })

  const handleSend = useCallback(
    (text: string) => {
      setMessages((prev) => [
        ...prev,
        {
          id: `${Date.now()}-${Math.random()}`,
          role: 'user',
          content: text,
          timestamp: new Date(),
        },
      ])
      setIsLoading(true)
      sendMessage(JSON.stringify({ role: 'user', content: text }))
    },
    [sendMessage],
  )

  const handleApprove = useCallback(
    async (msgId: string, approved: boolean, password: string): Promise<boolean> => {
      const msg = messages.find((m) => m.id === msgId)
      if (!msg || msg.role !== 'assistant' || !msg.decisionId) return false

      try {
        const res = await fetch(`${API_BASE}/api/approve/${msg.decisionId}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            approved,
            approved_by: 'supply-chain-manager',
            reason: approved ? 'Reviewed and approved' : 'Rejected by manager',
            password,
          }),
        })
        if (res.status === 403) {
          // Wrong manager password — decision stays pending; the card shows
          // the inline error and lets the manager retry.
          showToast('Invalid manager password — approval denied', 'warning')
          return false
        }
        if (res.status === 503) {
          showToast('Approvals are locked: no manager password configured on the server', 'warning')
          return false
        }
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const record = (await res.json()) as { status: string; final_response?: string | null }
        const newStatus = record.status as 'approved' | 'rejected'
        setMessages((prev) =>
          prev.map((m) => (m.id === msgId ? { ...m, approvalStatus: newStatus } : m)),
        )
        // The resumed graph synthesizes the final answer with the decision
        // applied — surface it as a new assistant message.
        if (record.final_response) {
          setMessages((prev) => [
            ...prev,
            {
              id: `${Date.now()}-${Math.random()}`,
              role: 'assistant',
              content: record.final_response as string,
              timestamp: new Date(),
            },
          ])
        }
        showToast(
          `Decision ${newStatus === 'approved' ? 'approved ✓' : 'rejected ✗'}`,
          newStatus === 'approved' ? 'success' : 'warning',
        )
        return true
      } catch (err) {
        console.error('Approval request failed:', err)
        showToast('Approval request failed — see console', 'warning')
        return false
      }
    },
    [messages, showToast],
  )

  const borderCls = isDark ? 'border-slate-700/50' : 'border-slate-300'
  const handleCls = isDark
    ? 'w-1 bg-slate-800 hover:bg-indigo-600/40 transition-colors cursor-col-resize'
    : 'w-1 bg-slate-200 hover:bg-indigo-400/40 transition-colors cursor-col-resize'

  return (
    <div className={`flex flex-col h-full overflow-hidden transition-colors duration-200
      ${isDark ? 'bg-slate-950 text-slate-100' : 'bg-slate-50 text-slate-900'}`}>

      {/* ── Resizable 3-column layout ── */}
      <PanelGroup direction="horizontal" className="flex-1 min-h-0">

        {/* Left — Knowledge Graph */}
        <Panel defaultSize={20} minSize={12} maxSize={40} id="kg-panel" order={1}>
          <div className={`flex flex-col h-full border-r ${borderCls} ${isDark ? 'bg-slate-950' : 'bg-white'}`}>
            <GraphViewer
              subgraph={kgSubgraph}
              isDark={isDark}
              onNodeClick={(nodeId) => handleSend(`Show me the supply chain relationships for ${nodeId}`)}
            />
          </div>
        </Panel>

        <PanelResizeHandle className={handleCls} />

        {/* Centre — Chat */}
        <Panel defaultSize={55} minSize={30} id="chat-panel" order={2}>
          <div className={`flex flex-col h-full ${isDark ? 'bg-slate-950' : 'bg-slate-50'}`}>
            <ChatPanel
              messages={messages}
              isLoading={isLoading}
              connectionState={connectionState}
              onSend={handleSend}
              onApprove={handleApprove}
              onReconnect={reconnect}
              isDark={isDark}
              onToggleTheme={() => setIsDark((v) => !v)}
            />
          </div>
        </Panel>

        <PanelResizeHandle className={handleCls} />

        {/* Right — Solver Results */}
        <Panel defaultSize={25} minSize={15} maxSize={45} id="solver-panel" order={3}>
          <div className={`flex flex-col h-full border-l ${borderCls} ${isDark ? 'bg-slate-950' : 'bg-white'}`}>
            <SolverResults result={solverResult} history={solverHistory} isDark={isDark} />
          </div>
        </Panel>

      </PanelGroup>

      {/* Toast stack */}
      <div className="fixed bottom-4 left-1/2 -translate-x-1/2 z-50 flex flex-col gap-2 items-center pointer-events-none">
        {toasts.map((t) => (
          <div
            key={t.id}
            className={`px-4 py-2 rounded-lg text-sm font-medium shadow-lg border animate-fade-in pointer-events-auto
              ${t.type === 'success'
                ? isDark ? 'bg-emerald-900/90 border-emerald-700 text-emerald-200' : 'bg-emerald-100 border-emerald-400 text-emerald-800'
                : t.type === 'warning'
                ? isDark ? 'bg-amber-900/90 border-amber-700 text-amber-200' : 'bg-amber-100 border-amber-400 text-amber-800'
                : isDark ? 'bg-slate-800/90 border-slate-600 text-slate-200' : 'bg-white border-slate-300 text-slate-700'}`}
          >
            {t.message}
          </div>
        ))}
      </div>
    </div>
  )
}

export default App

