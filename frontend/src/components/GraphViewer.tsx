import React, { useEffect, useRef } from 'react'
import { Network, Options } from 'vis-network'

// TODO Stage 3: wire real kg_subgraph from AgentState via WebSocket message

interface KGSubgraph {
  nodes: Array<{ id: string; label: string; group?: string }>
  edges: Array<{ from: string; to: string; label?: string }>
}

interface GraphViewerProps {
  subgraph: KGSubgraph | null
}

const NETWORK_OPTIONS: Options = {
  nodes: {
    shape: 'dot',
    size: 12,
    font: { size: 11, color: '#e5e7eb' },
    borderWidth: 2,
  },
  edges: {
    arrows: 'to',
    font: { size: 9, color: '#9ca3af' },
    color: { color: '#4b5563' },
  },
  physics: { stabilization: { iterations: 100 } },
}

const GraphViewer: React.FC<GraphViewerProps> = ({ subgraph }) => {
  const containerRef = useRef<HTMLDivElement>(null)
  const networkRef = useRef<Network | null>(null)

  useEffect(() => {
    if (!containerRef.current) return
    networkRef.current = new Network(
      containerRef.current,
      { nodes: subgraph?.nodes ?? [], edges: subgraph?.edges ?? [] },
      NETWORK_OPTIONS,
    )
    return () => networkRef.current?.destroy()
  }, [subgraph])

  return (
    <div className="flex flex-col h-full">
      <header className="px-4 py-2 border-b border-gray-700 font-semibold text-sm">
        Knowledge Graph
      </header>
      {subgraph ? (
        <div ref={containerRef} className="flex-1" />
      ) : (
        <div className="flex-1 flex items-center justify-center text-gray-500 text-sm">
          No subgraph selected — ask the copilot to traverse the supply network
        </div>
      )}
    </div>
  )
}

export default GraphViewer
