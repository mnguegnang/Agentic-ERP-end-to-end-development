import React, { useEffect, useRef } from 'react'
import { Network, Options } from 'vis-network'
import { GitFork, Share2 } from 'lucide-react'
import type { KGSubgraph } from '../types'

const DARK_OPTIONS: Options = {
  nodes: {
    shape: 'dot',
    size: 10,
    font: { size: 11, color: '#94a3b8' },
    borderWidth: 1.5,
    color: {
      background: '#1e3a5f',
      border: '#3b82f6',
      highlight: { background: '#2563eb', border: '#60a5fa' },
    },
  },
  edges: {
    arrows: { to: { enabled: true, scaleFactor: 0.6 } },
    font: { size: 9, color: '#475569', align: 'middle' },
    color: { color: '#334155', highlight: '#3b82f6' },
    smooth: { type: 'dynamic', enabled: true, roundness: 0.5 },
  },
  physics: {
    stabilization: { iterations: 150 },
    barnesHut: { gravitationalConstant: -2000, centralGravity: 0.2, springLength: 60 },
  },
  interaction: { hover: true, tooltipDelay: 200 },
}

const LIGHT_OPTIONS: Options = {
  nodes: {
    shape: 'dot',
    size: 10,
    font: { size: 11, color: '#334155' },
    borderWidth: 1.5,
    color: {
      background: '#dbeafe',
      border: '#3b82f6',
      highlight: { background: '#93c5fd', border: '#2563eb' },
    },
  },
  edges: {
    arrows: { to: { enabled: true, scaleFactor: 0.6 } },
    font: { size: 9, color: '#64748b', align: 'middle' },
    color: { color: '#94a3b8', highlight: '#3b82f6' },
    smooth: { type: 'dynamic', enabled: true, roundness: 0.5 },
  },
  physics: {
    stabilization: { iterations: 150 },
    barnesHut: { gravitationalConstant: -2000, centralGravity: 0.2, springLength: 60 },
  },
  interaction: { hover: true, tooltipDelay: 200 },
}

interface GraphViewerProps {
  subgraph: KGSubgraph | null
  isDark?: boolean
  onNodeClick?: (nodeId: string) => void
}

const GraphViewer: React.FC<GraphViewerProps> = ({ subgraph, isDark = true, onNodeClick }) => {
  const containerRef = useRef<HTMLDivElement>(null)
  const networkRef = useRef<Network | null>(null)

  useEffect(() => {
    if (!containerRef.current) return
    networkRef.current?.destroy()
    networkRef.current = new Network(
      containerRef.current,
      { nodes: subgraph?.nodes ?? [], edges: subgraph?.edges ?? [] },
      isDark ? DARK_OPTIONS : LIGHT_OPTIONS,
    )
    if (onNodeClick) {
      networkRef.current.on('click', (params) => {
        if (params.nodes.length > 0) {
          onNodeClick(String(params.nodes[0]))
        }
      })
    }
    return () => networkRef.current?.destroy()
  }, [subgraph, onNodeClick, isDark])

  return (
    <div className="flex flex-col h-full min-h-0">
      <header className="panel-header shrink-0 justify-between">
        <div className="flex items-center gap-2">
          <GitFork size={13} className="text-purple-400" />
          <span>Knowledge Graph</span>
        </div>
        {subgraph && (
          <span className={`text-[10px] font-mono ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>
            {subgraph.nodes.length}N / {subgraph.edges.length}E
          </span>
        )}
      </header>

      {subgraph ? (
        <div ref={containerRef} className="flex-1 min-h-0" />
      ) : (
        <div className="flex-1 flex flex-col items-center justify-center gap-3 px-4 text-center">
          <div className={`flex h-12 w-12 items-center justify-center rounded-xl border ${isDark ? 'bg-slate-800 border-slate-700' : 'bg-slate-100 border-slate-200'}`}>
            <Share2 size={22} className={isDark ? 'text-slate-600' : 'text-slate-400'} />
          </div>
          <p className={`text-sm ${isDark ? 'text-slate-500' : 'text-slate-500'}`}>No subgraph selected</p>
          <p className={`text-xs max-w-[160px] leading-relaxed ${isDark ? 'text-slate-600' : 'text-slate-400'}`}>
            Ask the copilot to traverse the supply network to visualise relationships here.
          </p>
        </div>
      )}
    </div>
  )
}

export default GraphViewer
