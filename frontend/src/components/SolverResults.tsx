import React from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'

// TODO Stage 3: map each solver type (MCNF, MEIO, VRP, JSP…) to its own chart variant

interface SolverOutput {
  solver: string
  status: string
  objective?: number
  chart_data?: Array<Record<string, unknown>>
}

interface SolverResultsProps {
  result: SolverOutput | null
}

const SolverResults: React.FC<SolverResultsProps> = ({ result }) => {
  return (
    <div className="flex flex-col h-full">
      <header className="px-4 py-2 border-b border-gray-700 font-semibold text-sm">
        Solver Results
      </header>

      {result ? (
        <div className="p-4 space-y-3 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-400">Solver</span>
            <span className="font-mono">{result.solver}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Status</span>
            <span
              className={
                result.status === 'OPTIMAL' ? 'text-green-400' : 'text-yellow-400'
              }
            >
              {result.status}
            </span>
          </div>
          {result.objective !== undefined && (
            <div className="flex justify-between">
              <span className="text-gray-400">Objective</span>
              <span className="font-mono">{result.objective.toLocaleString()}</span>
            </div>
          )}

          {result.chart_data && result.chart_data.length > 0 && (
            <ResponsiveContainer width="100%" height={160}>
              <BarChart data={result.chart_data as Record<string, string | number>[]}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="name" tick={{ fill: '#9ca3af', fontSize: 10 }} />
                <YAxis tick={{ fill: '#9ca3af', fontSize: 10 }} />
                <Tooltip
                  contentStyle={{ background: '#1f2937', border: 'none', fontSize: 11 }}
                />
                <Bar dataKey="value" fill="#3b82f6" />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      ) : (
        <div className="flex-1 flex items-center justify-center text-gray-500 text-sm px-4 text-center">
          No solver has run yet — ask the copilot to optimize routes, inventory, or schedules
        </div>
      )}
    </div>
  )
}

export default SolverResults
