export type Effort = 'low' | 'medium' | 'high' | 'xhigh' | 'max'

export type Probs = Record<string, number>

export type Route = {
  /** The model the session runs on: picked on its first prompt, null to keep the session's own. */
  model: string | null
  /** This prompt's effort. */
  effort: Effort
  score: number
  task: Probs
  signals: Probs
  /** Classifier latency. */
  ms: number
}

export type Cache = { read: number; written: number; uncached: number }

declare module 'claude-code' {
  interface PluginState {
    'magic-router': {
      /** Whether this session's model has been picked (on its first prompt, even if that failed). */
      isPicked: boolean
      route: Route | null
      /** What the engine asked for before the first rewrite; a change means /model or /effort took over. */
      baseline: { model: string; effort: string | number | null } | null
      /** Why the router is standing aside, when it is. */
      note: string | null
      cache: Cache | null
      isHidden: boolean
      /** Each subagent's route by agent id, picked on its first request; null leaves it alone. */
      agents: Record<string, Route | null>
    }
  }
}
