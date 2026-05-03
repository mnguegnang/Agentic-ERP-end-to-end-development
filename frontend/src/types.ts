/** Shared TypeScript types for the Agentic ERP frontend. */

export interface RagDocument {
  id: number
  supplier_id?: number
  chunk_text: string
  score?: number
}

interface BaseMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
}

export interface UserMessage extends BaseMessage {
  role: 'user'
}

export interface AssistantMessage extends BaseMessage {
  role: 'assistant'
  intent?: string
  intentConfidence?: number
  toolUsed?: string | null
  solverResult?: Record<string, unknown> | null
  ragDocuments?: RagDocument[] | null
  humanApprovalRequired?: boolean
  decisionId?: string
  approvalStatus?: 'pending' | 'approved' | 'rejected'
}

export type Message = UserMessage | AssistantMessage

// ─── Solver output ────────────────────────────────────────────────────────────
export interface SolverOutput {
  solver: string
  status: string
  objective?: number
  raw: Record<string, unknown>
}

// ─── KG subgraph ─────────────────────────────────────────────────────────────
export interface KGNode {
  id: string
  label: string
  group?: string
}

export interface KGEdge {
  from: string
  to: string
  label?: string
}

export interface KGSubgraph {
  nodes: KGNode[]
  edges: KGEdge[]
}
