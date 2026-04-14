import React from 'react'
import ChatPanel from './components/ChatPanel'
import GraphViewer from './components/GraphViewer'
import SolverResults from './components/SolverResults'

// TODO Stage 3: wire agent state (kg_subgraph, solver_output) into props
const App: React.FC = () => {
  return (
    <div className="flex h-screen bg-gray-900 text-white">
      {/* Left panel: Knowledge Graph */}
      <aside className="w-1/3 border-r border-gray-700 overflow-hidden">
        <GraphViewer subgraph={null} />
      </aside>

      {/* Center panel: Chat */}
      <main className="flex-1 flex flex-col">
        <ChatPanel />
      </main>

      {/* Right panel: Solver Results */}
      <aside className="w-1/4 border-l border-gray-700 overflow-y-auto">
        <SolverResults result={null} />
      </aside>
    </div>
  )
}

export default App
