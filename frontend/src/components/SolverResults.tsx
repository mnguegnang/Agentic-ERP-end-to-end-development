import React, { useState } from 'react'
import { Activity, CheckCircle2, AlertTriangle, Clock, BarChart2 } from 'lucide-react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'
import type { SolverOutput } from '../types'

// ─── Status badge ─────────────────────────────────────────────────────────────
const StatusBadge: React.FC<{ status: string }> = ({ status }) => {
  const ok = status === 'OPTIMAL' || status === 'FEASIBLE' || status === 'correct'
  const pending = status === 'no_solver_needed' || status === 'pending'
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-semibold ${
        ok
          ? 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/30'
          : pending
            ? 'bg-slate-500/20 text-slate-400 border border-slate-500/30'
            : 'bg-amber-500/15 text-amber-300 border border-amber-500/30'
      }`}
    >
      {ok ? (
        <CheckCircle2 size={11} />
      ) : pending ? (
        <Clock size={11} />
      ) : (
        <AlertTriangle size={11} />
      )}
      {status}
    </span>
  )
}

// ─── MCNF flow table ──────────────────────────────────────────────────────────
const McnfDetails: React.FC<{ raw: Record<string, unknown> }> = ({ raw }) => {
  const flows = raw.flows as Array<{from?: string; from_node?: string; to: string; flow: number}> | undefined
  const totalCost = raw.total_cost as number | undefined
  return (
    <>
      {totalCost !== undefined && (
        <div className="kv-row">
          <span className="text-slate-400">Total Cost</span>
          <span className="font-mono font-semibold text-emerald-300">
            ${totalCost.toLocaleString()}
          </span>
        </div>
      )}
      {flows && flows.length > 0 && (
        <div className="mt-3">
          <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-500 mb-2">
            Flows
          </p>
          <table className="solver-table">
            <thead>
              <tr>
                <th>From</th>
                <th>To</th>
                <th className="text-right">Units</th>
              </tr>
            </thead>
            <tbody>
              {flows.map((f, i) => (
                <tr key={i}>
                  <td>{f.from ?? f.from_node ?? '—'}</td>
                  <td>{f.to}</td>
                  <td className="text-right">{f.flow.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  )
}

// ─── VRP route table ──────────────────────────────────────────────────────────
const VrpDetails: React.FC<{ raw: Record<string, unknown>; isDark?: boolean }> = ({ raw, isDark = true }) => {
  const routes = raw.routes as Array<{vehicle: number; stops: number[]; distance: number}> | undefined
  const totalDist = raw.total_distance as number | undefined
  return (
    <>
      {totalDist !== undefined && (
        <div className="kv-row">
          <span className="text-slate-400">Total Distance</span>
          <span className="font-mono font-semibold text-emerald-300">
            {totalDist.toLocaleString()} km
          </span>
        </div>
      )}
      {routes && routes.length > 0 && (
        <div className="mt-3">
          <p className={`text-[10px] font-semibold uppercase tracking-wider mb-2 ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>
            Vehicle Routes
          </p>
          {routes.map((r) => (
            <div
              key={r.vehicle}
              className={`mb-2 rounded-lg border px-3 py-2 ${isDark ? 'border-slate-700/40 bg-slate-800/40' : 'border-slate-200 bg-slate-50'}`}
            >
              <p className={`text-[11px] font-semibold mb-1 ${isDark ? 'text-slate-300' : 'text-slate-700'}`}>
                Vehicle {r.vehicle}
              </p>
              <p className={`font-mono text-[10px] ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
                {r.stops.join(' → ')}
              </p>
              <p className={`text-[10px] mt-1 ${isDark ? 'text-slate-500' : 'text-slate-500'}`}>
                {r.distance.toLocaleString()} km
              </p>
            </div>
          ))}
        </div>
      )}
    </>
  )
}

// ─── Generic key-value display ────────────────────────────────────────────────
const GenericDetails: React.FC<{ raw: Record<string, unknown> }> = ({ raw }) => {
  const skip = new Set(['status'])
  const entries = Object.entries(raw).filter(([k, v]) => !skip.has(k) && v !== null && v !== undefined)
  return (
    <>
      {entries.slice(0, 10).map(([k, v]) => (
        <div className="kv-row" key={k}>
          <span className="text-slate-400 truncate">{k.replace(/_/g, ' ')}</span>
          <span className="font-mono text-slate-300 truncate ml-4 text-right">
            {typeof v === 'number'
              ? v.toLocaleString(undefined, { maximumFractionDigits: 4 })
              : typeof v === 'object'
                ? JSON.stringify(v).slice(0, 40)
                : String(v)}
          </span>
        </div>
      ))}
    </>
  )
}

// ─── Chart for numeric arrays ─────────────────────────────────────────────────
const SolverChart: React.FC<{ raw: Record<string, unknown> }> = ({ raw }) => {
  // Find first numeric array in raw for charting (e.g. demand_series, amplification)
  const chartEntry = Object.entries(raw).find(
    ([, v]) => Array.isArray(v) && (v as unknown[]).length > 1 && typeof (v as unknown[])[0] === 'number',
  )
  if (!chartEntry) return null
  const [label, arr] = chartEntry
  const data = (arr as number[]).map((v, i) => ({ name: `T${i + 1}`, value: v }))

  return (
    <div className="mt-4">
      <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-500 mb-2">
        {label.replace(/_/g, ' ')}
      </p>
      <ResponsiveContainer width="100%" height={100}>
        <BarChart data={data} margin={{ top: 0, right: 0, bottom: 0, left: -20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" vertical={false} />
          <XAxis dataKey="name" tick={{ fill: '#64748b', fontSize: 9 }} />
          <YAxis tick={{ fill: '#64748b', fontSize: 9 }} />
          <Tooltip
            contentStyle={{
              background: '#0f172a',
              border: '1px solid #334155',
              borderRadius: 6,
              fontSize: 11,
            }}
          />
          <Bar dataKey="value" fill="#3b82f6" radius={[2, 2, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

// ─── Main SolverResults panel ─────────────────────────────────────────────────
interface SolverResultsProps {
  result: SolverOutput | null
  history?: SolverOutput[]
  isDark?: boolean
}

const SOLVER_LABELS: Record<string, string> = {
  mcnf_solve: 'Min-Cost Network Flow',
  vrp_route: 'Vehicle Routing (VRP)',
  jsp_schedule: 'Job-Shop Schedule',
  robust_allocate: 'Robust Allocation',
  meio_optimize: 'Multi-Echelon Inventory',
  bullwhip_analyze: 'Bullwhip Analysis',
  disruption_resource: 'Disruption Response',
}

const SolverResults: React.FC<SolverResultsProps> = ({ result, history = [], isDark = true }) => {
  const [showHistory, setShowHistory] = useState(false)
  const resultsToShow = showHistory ? history : (result ? [result] : [])
  const displayResult = resultsToShow[0] ?? null
  return (
    <div className="flex flex-col h-full min-h-0">
      <header className="panel-header shrink-0 justify-between">
        <div className="flex items-center gap-2">
          <Activity size={13} className="text-blue-400" />
          <span>Solver Results</span>
        </div>
        <div className="flex items-center gap-2">
          {history.length > 1 && (
            <button
              onClick={() => setShowHistory((v) => !v)}
              className={`text-[10px] underline ${isDark ? 'text-slate-500 hover:text-slate-300' : 'text-slate-400 hover:text-slate-700'}`}
            >
              {showHistory ? 'Latest' : `History (${history.length})`}
            </button>
          )}
          {displayResult && <BarChart2 size={13} className="text-slate-500" />}
        </div>
      </header>

      {displayResult ? (
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {showHistory && (
            <p className={`text-[10px] font-mono ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>Showing {resultsToShow.length} runs</p>
          )}
          {/* Solver name */}
          <div>
            <p className={`text-[10px] font-semibold uppercase tracking-wider mb-1 ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>
              Solver
            </p>
            <p className={`text-sm font-semibold ${isDark ? 'text-slate-200' : 'text-slate-800'}`}>
              {SOLVER_LABELS[displayResult.solver] ?? displayResult.solver}
            </p>
          </div>

          {/* Status */}
          <div>
            <p className={`text-[10px] font-semibold uppercase tracking-wider mb-1.5 ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>
              Status
            </p>
            <StatusBadge status={displayResult.status} />
          </div>

          {/* Objective */}
          {displayResult.objective !== undefined && (
            <div className={`rounded-xl border p-3 text-center ${isDark ? 'border-blue-500/20 bg-blue-500/5' : 'border-blue-300/50 bg-blue-50'}`}>
              <p className={`text-[10px] font-semibold uppercase tracking-wider mb-0.5 ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>
                Objective
              </p>
              <p className={`text-2xl font-bold font-mono ${isDark ? 'text-blue-300' : 'text-blue-600'}`}>
                {displayResult.objective.toLocaleString()}
              </p>
            </div>
          )}

          {/* Type-specific details */}
          <div className="space-y-0">
            {displayResult.solver === 'mcnf_solve' && <McnfDetails raw={displayResult.raw} />}
            {displayResult.solver === 'vrp_route' && <VrpDetails raw={displayResult.raw} isDark={isDark} />}
            {!['mcnf_solve', 'vrp_route'].includes(displayResult.solver) && (
              <GenericDetails raw={displayResult.raw} />
            )}
          </div>

          {/* Chart */}
          <SolverChart raw={displayResult.raw} />
        </div>
      ) : (
        <div className="flex-1 flex flex-col items-center justify-center gap-3 px-4 text-center">
          <div className={`flex h-12 w-12 items-center justify-center rounded-xl border ${isDark ? 'bg-slate-800 border-slate-700' : 'bg-slate-100 border-slate-200'}`}>
            <Activity size={22} className={isDark ? 'text-slate-600' : 'text-slate-400'} />
          </div>
          <p className={`text-sm leading-relaxed ${isDark ? 'text-slate-500' : 'text-slate-500'}`}>
            No solver has run yet.
          </p>
          <p className={`text-xs max-w-[180px] ${isDark ? 'text-slate-600' : 'text-slate-400'}`}>
            Ask about route optimisation, inventory levels, or job scheduling to
            see results here.
          </p>
        </div>
      )}
    </div>
  )
}

export default SolverResults
