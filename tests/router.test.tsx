import { expect, test } from 'claude-code/testing'
import type { On } from 'claude-code'

import { FABLE_AT, effortFor, modelFor, score } from '../hooks/route'

// What the daemon answered for these prompts (GLiNER2.5-Decide, rounded).
const FIX = {
  typo: { text: 'fix the typo in README.md', choose: { agentic_coding: 0.367, scientific_coding: 0.023, tool_use: 0.129, knowledge_work: 0.214, long_context: 0.155, knowledge_qa: 0.073, reasoning: 0.039 }, flags: { multi_file: 0.126, planning: 0.135, ambiguous: 0.196, deep_reasoning: 0.1, large_scope: 0.101, verification: 0.33, quick: 0.44 }, ms: 757 },
  push: { text: 'push this branch and open a PR against main', choose: { agentic_coding: 0.35, scientific_coding: 0.041, tool_use: 0.358, knowledge_work: 0.058, long_context: 0.141, knowledge_qa: 0.02, reasoning: 0.032 }, flags: { multi_file: 0.249, planning: 0.3, ambiguous: 0.239, deep_reasoning: 0.138, large_scope: 0.22, verification: 0.349, quick: 0.337 }, ms: 321 },
  jwt: { text: 'Refactor the auth module to use JWT refresh tokens across the API gateway and update all tests', choose: { agentic_coding: 0.567, scientific_coding: 0.031, tool_use: 0.176, knowledge_work: 0.038, long_context: 0.129, knowledge_qa: 0.021, reasoning: 0.039 }, flags: { multi_file: 0.695, planning: 0.162, ambiguous: 0.092, deep_reasoning: 0.122, large_scope: 0.205, verification: 0.405, quick: 0.327 }, ms: 327 },
  scheduler: { text: 'build a distributed job scheduler with leader election, retries and a web dashboard from scratch', choose: { agentic_coding: 0.421, scientific_coding: 0.155, tool_use: 0.136, knowledge_work: 0.055, long_context: 0.14, knowledge_qa: 0.029, reasoning: 0.065 }, flags: { multi_file: 0.284, planning: 0.262, ambiguous: 0.088, deep_reasoning: 0.125, large_scope: 0.812, verification: 0.306, quick: 0.129 }, ms: 326 },
}

const SESSION = { model: 'claude-opus-5-5', effort: 'medium' } as const

// The engine beneath the plugin: a daemon answering from FIX plus `extra` (or down), and a model that records each request.
function engine(on: On, extra: object = {}) {
  const daemon = { isUp: true }
  const sent: { model: string; effort: unknown }[] = []
  on('http.fetch', ($, e) => {
    if (!daemon.isUp) return { deny: 'ECONNREFUSED' }
    if (e.url.endsWith('/health')) return { value: { status: 200, ok: true, headers: {}, text: '{"ready": true}' } }
    const { text } = JSON.parse(e.init?.body ?? '{}')
    const found = Object.values(FIX).find(f => f.text === text)
    return { value: { status: 200, ok: true, headers: {}, text: JSON.stringify({ ...found, ...extra }) } }
  })
  on('prompt.submit', ($, e) => ({ text: e.text }))
  on('ui.render', { component: 'AbovePrompt' }, ($, e) => {
    const { Box } = $.ui.resolve(e)
    return <Box /> // the engine's own band: empty
  })
  on('turn.step', async function* ($, e) {
    sent.push({ model: e.model, effort: e.effort })
    return { turnId: e.turnId, index: e.index, answer: '', toolUses: [], stopReason: 'end_turn' as const, usage: null }
  })
  return { daemon, sent }
}

const submit = ($: any, text: string) => $.prompt.submit({ text, wait: false, origin: { kind: 'composer' } })

async function step($: any, asked: { model: string; effort: string; agentId?: string } = SESSION) {
  const stream = $.turn.step({ turnId: 't', index: 0, messageCount: 1, ...asked })
  for await (const _ of stream);
  return stream.result
}

test('the policy routes the recorded classifications', () => {
  const route = (f: (typeof FIX)[keyof typeof FIX]) => {
    const s = score(f.choose, f.flags)
    return `${modelFor(s)} ${effortFor(s)}`
  }
  expect(route(FIX.typo)).toBe('claude-sonnet-5-5 low')
  expect(route(FIX.push)).toBe('claude-sonnet-5-5 medium')
  expect(route(FIX.jwt)).toBe('claude-opus-5-5 high')
  expect(route(FIX.scheduler)).toBe('claude-opus-5-5 max')
  expect(FABLE_AT === Infinity ? modelFor(99) : 'claude-fable-5-1').toBe(FABLE_AT === Infinity ? 'claude-opus-5-5' : modelFor(FABLE_AT)) // Fable stays off until FABLE_AT is lowered
})

test('the first prompt picks the model for the session; every prompt picks its effort', async ($: any, on) => {
  const { sent } = engine(on)

  await submit($, FIX.typo.text)
  await step($)
  expect(sent.at(-1)).toEqual({ model: 'claude-sonnet-5-5', effort: 'low' })

  await submit($, FIX.scheduler.text)
  await step($)
  expect(sent.at(-1)).toEqual({ model: 'claude-sonnet-5-5', effort: 'max' })

  await submit($, 'yes, go ahead')
  await step($)
  expect(sent.at(-1)).toEqual({ model: 'claude-sonnet-5-5', effort: 'max' })

  await step($, { ...SESSION, agentId: 'a1' })
  expect(sent.at(-1)).toEqual(SESSION)
})

test("a tuned daemon's effort answer wins over the score; the score still picks the model", async ($: any, on) => {
  const { sent } = engine(on, { effort: { low: 0.1, medium: 0.15, high: 0.6, xhigh: 0.1, max: 0.05 } })

  await submit($, FIX.typo.text)
  await step($)
  expect(sent.at(-1)).toEqual({ model: 'claude-sonnet-5-5', effort: 'high' })
})

test('a classifier down on the first prompt keeps the session model for good', async ($: any, on) => {
  const { daemon, sent } = engine(on)

  daemon.isUp = false
  await submit($, FIX.scheduler.text)
  await step($)
  expect(sent.at(-1)).toEqual(SESSION)

  daemon.isUp = true
  await submit($, FIX.jwt.text)
  await step($)
  expect(sent.at(-1)).toEqual({ model: SESSION.model, effort: 'high' })
})

test('a model chosen by hand stands the router aside', async ($: any, on) => {
  const { sent } = engine(on)

  await submit($, FIX.typo.text)
  await step($)
  expect(sent.at(-1)).toEqual({ model: 'claude-sonnet-5-5', effort: 'low' })

  await step($, { model: 'claude-fable-5-1', effort: 'high' })
  expect(sent.at(-1)).toEqual({ model: 'claude-fable-5-1', effort: 'high' })
})

test('the band shows the route and nothing about the classification', async ($: any, on) => {
  engine(on)
  await submit($, FIX.scheduler.text)
  await step($)

  for (const surface of ['terminal', 'desktop'] as const) {
    const ui = await $.ui.mount({
      plugin: 'model-router',
      surface,
      component: 'AbovePrompt',
      props: { hasSurvey: false, isWorking: false, maxRows: 6 },
    })
    expect(await ui.find({ text: /opus 5\.5/ })).toBeDefined()
    expect(await ui.find({ text: /agentic|large scope|score/ })).toBeUndefined()
    expect(await ui.find({ text: /^max$/ })).toBeDefined()
    await ui.unmount()
  }

  const ui = await $.ui.mount({ plugin: 'model-router', surface: 'terminal', component: 'AbovePrompt', props: { hasSurvey: false, isWorking: false, maxRows: 6 } })
  await ui.press({ key: 'hide' })
  expect(await ui.find({ text: /opus 5\.5/ })).toBeUndefined()
})
