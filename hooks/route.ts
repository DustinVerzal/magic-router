// The routing policy: what the classifier is asked, and how its answer becomes a model and an effort.
// The model is picked once, on a session's first prompt (a switch later forfeits the prompt cache);
// effort is picked again on every prompt.

import type { Effort, Probs } from '../types'

export const MODELS = { sonnet: 'claude-sonnet-5-5', opus: 'claude-opus-5-5' } as const

// The type of work, one label per eval family of the Artificial Analysis Intelligence Index v4.1.
// The classifier answers one distribution over these.
export const TASKS: Record<string, string> = {
  agentic_coding: 'multi-step software engineering in an existing codebase: implement a feature, fix a bug, refactor, migrate, run and fix tests', // Terminal-Bench
  scientific_coding: 'write a self-contained algorithm, function or numerical/scientific computation from a specification', // SciCode
  tool_use: 'operate tools, CLIs or APIs to carry out a procedure: git, deploys, config changes, environment setup, data lookups', // τ³-Bench
  knowledge_work: 'produce a professional deliverable such as documentation, a report, a plan, a spec, an email or slides', // GDPval-AA
  long_context: 'read and synthesize across many files or long documents: explain how a codebase works, audit, review a large diff', // AA-LCR
  knowledge_qa: 'answer a factual or conceptual question without changing anything', // AA-Omniscience, GPQA
  reasoning: 'hard math, proofs, physics, algorithm design or puzzles that need deep step-by-step reasoning', // HLE, CritPt
}

// Complexity, asked as independent yes/no signals: on a 17-prompt hand-labelled set their weighted sum
// tracked complexity at r=0.85, where five ordinal labels ("trivial" .. "extreme") came out near flat.
export const SIGNALS: Record<string, string> = {
  multi_file: 'requires changes across multiple files, modules or services',
  planning: 'requires design decisions, architecture or a plan before acting',
  ambiguous: 'open-ended or underspecified requirements',
  deep_reasoning: 'requires difficult multi-step reasoning, proofs or math',
  large_scope: 'builds a substantial system or feature from scratch',
  verification: 'needs tests, review or careful verification of correctness',
  quick: 'a quick question or a tiny, mechanical single-step edit',
}

export const SIGNAL_WEIGHTS: Probs = {
  multi_file: 1,
  planning: 1,
  ambiguous: 0.5,
  deep_reasoning: 1.5,
  large_scope: 1.5,
  verification: 0.5,
  quick: -1.5,
}

// Lean up where Opus' lead on the matching AA evals is widest (agentic, long context, hard reasoning).
export const TASK_BIAS: Probs = {
  agentic_coding: 0.3,
  reasoning: 0.3,
  long_context: 0.3,
  scientific_coding: 0.1,
  knowledge_work: 0,
  tool_use: -0.5,
  knowledge_qa: -0.3,
}

export const OPUS_AT = 1.1

// Each effort's lowest score, highest first. Tuned by eye on 15 prompts: retune on your own traffic.
export const EFFORTS: readonly (readonly [number, Effort])[] = [
  [2.0, 'max'],
  [1.5, 'xhigh'],
  [1.0, 'high'],
  [0.4, 'medium'],
  [-Infinity, 'low'],
]

// ponytail: replies this short ("yes", "go ahead", "continue") keep the last effort instead of being
// read as trivial; classify them against the previous prompt if that misroutes.
export const FOLLOW_UP_WORDS = 4

const times = (p: Probs, w: Probs) => Object.fromEntries(Object.entries(w).map(([k, weight]) => [k, (p[k] ?? 0) * weight]))

// The score's terms by label (task and signal labels never collide): what pushed a prompt up or down.
export const terms = (task: Probs, signals: Probs): Probs => ({ ...times(signals, SIGNAL_WEIGHTS), ...times(task, TASK_BIAS) })

export const score = (task: Probs, signals: Probs) => Object.values(terms(task, signals)).reduce((a, b) => a + b, 0)

export const effortFor = (s: number): Effort => EFFORTS.find(([min]) => s >= min)![1]

export const modelFor = (s: number): string => (s >= OPUS_AT ? MODELS.opus : MODELS.sonnet)

export const isFollowUp = (text: string) => text.trim().split(/\s+/).length < FOLLOW_UP_WORDS
