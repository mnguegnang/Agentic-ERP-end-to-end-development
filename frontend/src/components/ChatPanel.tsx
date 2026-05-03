import React, { useState, useEffect, useRef, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  Send,
  Wifi,
  WifiOff,
  RefreshCw,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  ChevronDown,
  ChevronUp,
  FileText,
  BrainCircuit,
  Bot,
  User,
  Search,
  X,
  Download,
  Moon,
  Sun,
} from 'lucide-react'
import type { Message, AssistantMessage } from '../types'
import type { ConnectionState } from '../hooks/useWebSocket'

// ─── Intent metadata ─────────────────────────────────────────────────────────
const INTENT_META: Record<string, { label: string; color: string }> = {
  mcnf_solve: { label: 'MCNF', color: 'bg-blue-500/20 text-blue-300 border-blue-500/30' },
  vrp_route: { label: 'VRP', color: 'bg-cyan-500/20 text-cyan-300 border-cyan-500/30' },
  jsp_schedule: { label: 'JSP', color: 'bg-indigo-500/20 text-indigo-300 border-indigo-500/30' },
  robust_allocate: { label: 'ROBUST', color: 'bg-violet-500/20 text-violet-300 border-violet-500/30' },
  meio_optimize: { label: 'MEIO', color: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30' },
  bullwhip_analyze: { label: 'BULLWHIP', color: 'bg-teal-500/20 text-teal-300 border-teal-500/30' },
  disruption_resource: { label: 'DISRUPTION', color: 'bg-orange-500/20 text-orange-300 border-orange-500/30' },
  kg_query: { label: 'KG QUERY', color: 'bg-purple-500/20 text-purple-300 border-purple-500/30' },
  contract_query: { label: 'CONTRACT', color: 'bg-amber-500/20 text-amber-300 border-amber-500/30' },
  multi_step: { label: 'MULTI-STEP', color: 'bg-pink-500/20 text-pink-300 border-pink-500/30' },
  unclear: { label: 'UNCLEAR', color: 'bg-slate-500/20 text-slate-400 border-slate-500/30' },
}

// ─── Connection status pill ───────────────────────────────────────────────────
interface ConnectionPillProps {
  state: ConnectionState
  onReconnect: () => void
}

const ConnectionPill: React.FC<ConnectionPillProps> = ({ state, onReconnect }) => {
  const configs = {
    connected: {
      icon: <Wifi size={11} />,
      label: 'Connected',
      cls: 'text-emerald-400 bg-emerald-400/10 border-emerald-400/30',
    },
    connecting: {
      icon: <RefreshCw size={11} className="animate-spin" />,
      label: 'Connecting…',
      cls: 'text-amber-400 bg-amber-400/10 border-amber-400/30',
    },
    reconnecting: {
      icon: <RefreshCw size={11} className="animate-spin" />,
      label: 'Reconnecting…',
      cls: 'text-amber-400 bg-amber-400/10 border-amber-400/30',
    },
    disconnected: {
      icon: <WifiOff size={11} />,
      label: 'Disconnected',
      cls: 'text-red-400 bg-red-400/10 border-red-400/30',
    },
  }
  const c = configs[state]

  return (
    <div className="flex items-center gap-2">
      <span
        className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[10px] font-medium ${c.cls}`}
      >
        {c.icon}
        {c.label}
      </span>
      {state === 'disconnected' && (
        <button
          onClick={onReconnect}
          className="text-[10px] text-slate-400 hover:text-slate-200 underline"
        >
          Retry
        </button>
      )}
    </div>
  )
}

// ─── Typing indicator ─────────────────────────────────────────────────────────
const TypingIndicator: React.FC<{ isDark?: boolean }> = ({ isDark = true }) => (
  <div className="flex items-start gap-3 px-4 py-1">
    <div className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full border ${isDark ? 'bg-slate-700 border-slate-600' : 'bg-slate-100 border-slate-300'}`}>
      <Bot size={14} className={isDark ? 'text-slate-400' : 'text-slate-500'} />
    </div>
    <div className="msg-bubble-assistant flex items-center gap-1 py-3 px-4">
      <span className={`typing-dot w-2 h-2 rounded-full inline-block ${isDark ? 'bg-slate-400' : 'bg-slate-400'}`} />
      <span className={`typing-dot w-2 h-2 rounded-full inline-block ${isDark ? 'bg-slate-400' : 'bg-slate-400'}`} />
      <span className={`typing-dot w-2 h-2 rounded-full inline-block ${isDark ? 'bg-slate-400' : 'bg-slate-400'}`} />
    </div>
  </div>
)

// ─── Sources accordion ───────────────────────────────────────────────────────
interface SourcesProps {
  docs: NonNullable<AssistantMessage['ragDocuments']>
  isDark?: boolean
}
const Sources: React.FC<SourcesProps> = ({ docs, isDark = true }) => {
  const [open, setOpen] = useState(false)
  return (
    <div className={`mt-2.5 rounded-lg border overflow-hidden ${isDark ? 'border-slate-600/40 bg-slate-900/50' : 'border-slate-200 bg-slate-50'}`}>
      <button
        onClick={() => setOpen((v) => !v)}
        className={`flex w-full items-center justify-between px-3 py-2 text-[11px] font-semibold transition-colors ${isDark ? 'text-slate-400 hover:text-slate-200' : 'text-slate-500 hover:text-slate-700'}`}
      >
        <span className="flex items-center gap-1.5">
          <FileText size={11} />
          {docs.length} source{docs.length !== 1 ? 's' : ''}
        </span>
        {open ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
      </button>
      {open && (
        <div className={`divide-y ${isDark ? 'divide-slate-700/40' : 'divide-slate-200'}`}>
          {docs.slice(0, 5).map((d, i) => (
            <div key={i} className="px-3 py-2">
              {d.score !== undefined && (
                <span className={`float-right text-[10px] font-mono ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>
                  {(d.score * 100).toFixed(0)}%
                </span>
              )}
              <p className={`text-[11px] line-clamp-3 pr-12 ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>{d.chunk_text}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ─── HiTL approval card ──────────────────────────────────────────────────────
interface ApprovalCardProps {
  msg: AssistantMessage
  onApprove: (id: string, approved: boolean) => void
}
const ApprovalCard: React.FC<ApprovalCardProps> = ({ msg, onApprove }) => {
  if (!msg.humanApprovalRequired) return null
  return (
    <div className="approval-card">
      <div className="flex items-start gap-2 mb-2.5">
        <AlertTriangle size={14} className="text-amber-400 mt-0.5 shrink-0" />
        <div>
          <p className="text-xs font-semibold text-amber-300">Human Approval Required</p>
          <p className="text-[11px] text-slate-400 mt-0.5">
            This decision exceeds the $10,000 cost threshold. A supply-chain manager must
            approve or reject before execution.
          </p>
        </div>
      </div>

      {msg.approvalStatus === 'pending' && (
        <div className="flex items-center gap-2">
          <button className="btn-approve" onClick={() => onApprove(msg.id, true)}>
            <CheckCircle2 size={12} />
            Approve
          </button>
          <button className="btn-reject" onClick={() => onApprove(msg.id, false)}>
            <XCircle size={12} />
            Reject
          </button>
          <span className="text-[10px] font-mono text-slate-500 ml-auto">
            ID: {msg.decisionId?.slice(0, 8)}…
          </span>
        </div>
      )}

      {msg.approvalStatus === 'approved' && (
        <div className="flex items-center gap-2 rounded-lg bg-emerald-900/40 border border-emerald-500/30 px-3 py-2">
          <CheckCircle2 size={13} className="text-emerald-400" />
          <span className="text-xs font-semibold text-emerald-300">
            Approved — execution authorised by supply-chain manager
          </span>
        </div>
      )}

      {msg.approvalStatus === 'rejected' && (
        <div className="flex items-center gap-2 rounded-lg bg-red-900/40 border border-red-500/30 px-3 py-2">
          <XCircle size={13} className="text-red-400" />
          <span className="text-xs font-semibold text-red-300">
            Rejected — execution blocked by supply-chain manager
          </span>
        </div>
      )}
    </div>
  )
}

// ─── Single message bubble ───────────────────────────────────────────────────
interface MessageBubbleProps {
  msg: Message
  onApprove: (id: string, approved: boolean) => void
  isDark?: boolean
}

const MessageBubble: React.FC<MessageBubbleProps> = ({ msg, onApprove, isDark = true }) => {
  const timeStr = msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })

  if (msg.role === 'user') {
    return (
      <div className="flex items-end justify-end gap-2 px-4 py-1">
        <div className="flex flex-col items-end gap-1 min-w-0">
          <div className="msg-bubble-user whitespace-pre-wrap break-words">{msg.content}</div>
          <span className={`text-[10px] pr-1 ${isDark ? 'text-slate-600' : 'text-slate-400'}`}>{timeStr}</span>
        </div>
        <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-blue-600/20 border border-blue-500/30">
          <User size={13} className="text-blue-400" />
        </div>
      </div>
    )
  }

  // Assistant
  const a = msg as AssistantMessage
  const intentMeta = a.intent ? INTENT_META[a.intent] : null

  return (
    <div className="flex items-start gap-2 px-4 py-1">
      <div className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full border mt-0.5 ${isDark ? 'bg-slate-700 border-slate-600' : 'bg-slate-100 border-slate-300'}`}>
        <Bot size={13} className={isDark ? 'text-slate-300' : 'text-slate-500'} />
      </div>

      <div className="flex flex-col gap-1 min-w-0 flex-1">
        {/* Intent + confidence badges */}
        {intentMeta && (
          <div className="flex items-center gap-2 mb-0.5">
            <span className={`intent-badge border ${intentMeta.color}`}>
              <BrainCircuit size={9} />
              {intentMeta.label}
            </span>
            {a.intentConfidence !== undefined && (
              <span className="text-[10px] font-mono text-slate-500">
                {(a.intentConfidence * 100).toFixed(0)}% conf.
              </span>
            )}
            {a.toolUsed && (
              <span className="text-[10px] text-slate-500">
                tool:{' '}
                <span className="font-mono text-slate-400">{a.toolUsed}</span>
              </span>
            )}
          </div>
        )}

        <div className="msg-bubble-assistant">
          <div className={`prose prose-sm max-w-none
            prose-headings:font-semibold prose-headings:mt-3 prose-headings:mb-1
            prose-h3:text-sm prose-h2:text-base
            prose-p:leading-relaxed prose-p:my-1
            prose-strong:font-semibold
            prose-ul:my-1 prose-ul:pl-4 prose-li:my-0.5
            prose-ol:my-1 prose-ol:pl-4
            prose-code:px-1 prose-code:rounded prose-code:text-xs
            prose-pre:border prose-pre:rounded-lg
            prose-blockquote:border-l-blue-500
            ${isDark
              ? 'prose-invert prose-headings:text-slate-200 prose-p:text-slate-300 prose-strong:text-slate-100 prose-em:text-slate-300 prose-li:text-slate-300 prose-code:text-blue-300 prose-code:bg-slate-800 prose-pre:bg-slate-900 prose-pre:border-slate-700 prose-blockquote:text-slate-400 prose-hr:border-slate-700'
              : 'prose-headings:text-slate-800 prose-p:text-slate-700 prose-strong:text-slate-900 prose-em:text-slate-600 prose-li:text-slate-700 prose-code:text-blue-700 prose-code:bg-slate-100 prose-pre:bg-slate-50 prose-pre:border-slate-200 prose-blockquote:text-slate-500 prose-hr:border-slate-200'}`}>
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{a.content}</ReactMarkdown>
          </div>

          {/* HiTL card */}
          <ApprovalCard msg={a} onApprove={onApprove} />

          {/* RAG sources */}
          {a.ragDocuments && a.ragDocuments.length > 0 && (
            <Sources docs={a.ragDocuments} isDark={isDark} />
          )}
        </div>

        <span className={`text-[10px] pl-1 ${isDark ? 'text-slate-600' : 'text-slate-400'}`}>{timeStr}</span>
      </div>
    </div>
  )
}

// ─── Empty state ─────────────────────────────────────────────────────────────
const EmptyState: React.FC<{ isDark?: boolean }> = ({ isDark = true }) => (
  <div className="flex-1 flex flex-col items-center justify-center gap-6 px-8 text-center">
    <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-blue-600/10 border border-blue-500/20">
      <BrainCircuit size={30} className="text-blue-400" />
    </div>
    <div>
      <h2 className={`text-base font-semibold mb-1 ${isDark ? 'text-slate-200' : 'text-slate-800'}`}>
        Agentic ERP Supply Chain Copilot
      </h2>
      <p className={`text-sm leading-relaxed max-w-xs ${isDark ? 'text-slate-500' : 'text-slate-500'}`}>
        Ask about supplier networks, contract terms, route optimisation, inventory
        levels, or disruption scenarios.
      </p>
    </div>
    <div className="grid grid-cols-1 gap-2 w-full max-w-sm">
      {[
        'Which suppliers provide bearings for the assembly line?',
        'What are the payment terms in our contracts?',
        'Route 500 units from factory (node A) to customer (node B). Arc capacity 1000, cost_per_unit=2.',
        'Schedule vehicle routes from depot to 3 delivery locations with capacity 200 units each.',
      ].map((hint) => (
        <div
          key={hint}
          className={`rounded-lg border px-3 py-2 text-left text-xs cursor-default transition-colors ${
            isDark
              ? 'border-slate-700/50 bg-slate-800/40 text-slate-400 hover:border-slate-600 hover:text-slate-300'
              : 'border-slate-200 bg-white text-slate-600 hover:border-slate-400 hover:text-slate-800 shadow-sm'
          }`}
        >
          {hint}
        </div>
      ))}
    </div>
  </div>
)

// ─── Main ChatPanel ───────────────────────────────────────────────────────────
interface ChatPanelProps {
  messages: Message[]
  isLoading: boolean
  connectionState: ConnectionState
  onSend: (text: string) => void
  onApprove: (msgId: string, approved: boolean) => void
  onReconnect: () => void
  isDark: boolean
  onToggleTheme: () => void
}

const ChatPanel: React.FC<ChatPanelProps> = ({
  messages,
  isLoading,
  connectionState,
  onSend,
  onApprove,
  onReconnect,
  isDark,
  onToggleTheme,
}) => {
  const [input, setInput] = useState('')
  const [searchQuery, setSearchQuery] = useState('')
  const [showSearch, setShowSearch] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const searchRef = useRef<HTMLInputElement>(null)

  // Auto-scroll to latest message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  // Focus search when opened
  useEffect(() => {
    if (showSearch) searchRef.current?.focus()
  }, [showSearch])

  const canSend = input.trim().length > 0 && connectionState === 'connected' && !isLoading

  const handleSend = useCallback(() => {
    const text = input.trim()
    if (!text || !canSend) return
    onSend(text)
    setInput('')
    textareaRef.current?.focus()
  }, [input, canSend, onSend])

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  // Auto-resize textarea
  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value)
    const el = e.target
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 140)}px`
  }

  // Export chat as JSON
  const handleExport = useCallback(() => {
    const data = messages.map((m) => ({
      role: m.role,
      content: m.content,
      timestamp: m.timestamp.toISOString(),
      ...(m.role === 'assistant' ? { intent: (m as AssistantMessage).intent } : {}),
    }))
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `erp-chat-${new Date().toISOString().slice(0, 10)}.json`
    a.click()
    URL.revokeObjectURL(url)
  }, [messages])

  // Filter messages by search
  const filteredMessages = searchQuery
    ? messages.filter((m) =>
        m.content.toLowerCase().includes(searchQuery.toLowerCase()),
      )
    : messages

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* ── Header ── */}
      <header className="panel-header shrink-0 justify-between">
        <div className="flex items-center gap-2">
          <BrainCircuit size={13} className="text-blue-400" />
          <span>Agentic ERP Copilot</span>
        </div>
        <div className="flex items-center gap-2">
          <ConnectionPill state={connectionState} onReconnect={onReconnect} />
          {messages.length > 0 && (
            <>
              <button
                onClick={() => setShowSearch((v) => !v)}
                title="Search messages"
                className="p-1 rounded text-slate-500 hover:text-slate-300 transition-colors"
              >
                <Search size={13} />
              </button>
              <button
                onClick={handleExport}
                title="Export chat as JSON"
                className="p-1 rounded text-slate-500 hover:text-slate-300 transition-colors"
              >
                <Download size={13} />
              </button>
            </>
          )}
          <a
            href={`http://${window.location.hostname}:8000/docs`}
            target="_blank"
            rel="noreferrer"
            title="API docs"
            className="p-1 rounded text-slate-500 hover:text-slate-300 transition-colors text-[10px] font-mono"
          >
            API
          </a>
          <button
            onClick={onToggleTheme}
            title="Toggle theme"
            className="p-1 rounded text-slate-500 hover:text-slate-300 transition-colors"
          >
            {isDark ? <Sun size={13} /> : <Moon size={13} />}
          </button>
        </div>
      </header>

      {/* ── Search bar ── */}
      {showSearch && (
        <div className={`shrink-0 flex items-center gap-2 px-3 py-2 border-b ${isDark ? 'border-slate-700/60 bg-slate-900/80' : 'border-slate-200 bg-white'}`}>
          <Search size={12} className="text-slate-500 shrink-0" />
          <input
            ref={searchRef}
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search messages…"
            className={`flex-1 bg-transparent text-sm placeholder-slate-500 focus:outline-none ${isDark ? 'text-slate-200' : 'text-slate-800'}`}
          />
          {searchQuery && (
            <span className={`text-[10px] shrink-0 ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>
              {filteredMessages.length} result{filteredMessages.length !== 1 ? 's' : ''}
            </span>
          )}
          <button
            onClick={() => { setShowSearch(false); setSearchQuery('') }}
            className={isDark ? 'text-slate-500 hover:text-slate-300' : 'text-slate-400 hover:text-slate-600'}
          >
            <X size={12} />
          </button>
        </div>
      )}

      {/* ── Messages ── */}
      <div className="flex-1 min-h-0 overflow-y-auto py-3 space-y-1">
        {filteredMessages.length === 0 && !searchQuery ? (
          <EmptyState isDark={isDark} />
        ) : filteredMessages.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <p className="text-sm text-slate-500">No messages match "{searchQuery}"</p>
          </div>
        ) : (
          filteredMessages.map((m) => (
            <MessageBubble key={m.id} msg={m} onApprove={onApprove} isDark={isDark} />
          ))
        )}
        {isLoading && !searchQuery && <TypingIndicator isDark={isDark} />}
        <div ref={bottomRef} />
      </div>

      {/* ── Input ── */}
      <div className={`shrink-0 border-t p-3 ${isDark ? 'border-slate-700/60 bg-slate-900/60 backdrop-blur-sm' : 'border-slate-200 bg-white'}`}>
        <div className={`flex items-end gap-2 rounded-xl border px-3 py-2 focus-within:border-blue-500/50 transition-colors ${isDark ? 'border-slate-600/50 bg-slate-800/60' : 'border-slate-300 bg-slate-50'}`}>
          <textarea
            ref={textareaRef}
            rows={1}
            className={`flex-1 min-h-0 resize-none bg-transparent text-sm placeholder-slate-400 focus:outline-none leading-relaxed ${isDark ? 'text-slate-200' : 'text-slate-800'}`}
            placeholder="Ask about supply chain disruptions, contracts, VRP routes, inventory…"
            value={input}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            disabled={connectionState !== 'connected'}
          />
          <button
            className="send-btn shrink-0"
            onClick={handleSend}
            disabled={!canSend}
            aria-label="Send message"
          >
            <Send size={15} className="text-white" />
          </button>
        </div>
        <p className={`text-[10px] mt-1.5 pl-1 ${isDark ? 'text-slate-600' : 'text-slate-400'}`}>
          Press <kbd className={`rounded px-1 py-0.5 ${isDark ? 'bg-slate-700 text-slate-400' : 'bg-slate-200 text-slate-600'}`}>Enter</kbd> to
          send · <kbd className={`rounded px-1 py-0.5 ${isDark ? 'bg-slate-700 text-slate-400' : 'bg-slate-200 text-slate-600'}`}>Shift+Enter</kbd>{' '}
          for new line
        </p>
      </div>
    </div>
  )
}

export default ChatPanel
