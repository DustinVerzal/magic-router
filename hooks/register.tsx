import { atom, read, update } from 'claude-code'
import type { EngineInterface as $, Register } from 'claude-code'

import type { Effort, Probs, Route } from '../types'
import { EFFORTS, MODELS, PICKS, SIGNALS, TASKS, effortFor, isFollowUp, modelFor, score } from './route'

// The daemon's URL: ROUTER_PORT, as server/classifier.py and scripts/gliner.sh read it.
const daemon = async ($: $) =>
  `http://127.0.0.1:${(await $.env.get('ROUTER_PORT')) || '8765'}`

const isPicked = atom({ plugin: 'magic-router', key: 'isPicked' } as const, false)
const route = atom({ plugin: 'magic-router', key: 'route' } as const, null)
const baseline = atom({ plugin: 'magic-router', key: 'baseline' } as const, null)
const note = atom({ plugin: 'magic-router', key: 'note' } as const, null)
const cache = atom({ plugin: 'magic-router', key: 'cache' } as const, null)
const isHidden = atom({ plugin: 'magic-router', key: 'isHidden' } as const, false)
const agents = atom({ plugin: 'magic-router', key: 'agents' } as const, {})

// What a person sent; task notifications, peers and plugins keep the current route.
const ROUTED_ORIGINS = new Set(['composer', 'bridge', 'sdk'])

// effort and pick come back only from a daemon with a tuned adapter (scripts/tune.py).
type Classified = { choose: Probs; flags: Probs; effort?: Probs; pick?: Probs; ms: number }

const LADDER = EFFORTS.map(([, e]) => e).reverse()
const top = <T extends string = Effort>(p: Probs) => Object.keys(p).reduce((a, b) => (p[b] > p[a] ? b : a)) as T
// ponytail: theme keys, so the colours follow light and dark themes; low and medium share green
const HEAT: Record<Effort, string> = { low: 'success', medium: 'success', high: 'warning', xhigh: 'error', max: 'error' }
const pct = (p: number) => `${Math.round(p * 100)}%`
// The routed model, unless the asked one is a variant of it ('claude-opus-5-5[1m]', the 1M-context window):
// the bare id would quietly drop the variant.
const onto = (asked: string, routed: string | null) => (routed === null || asked.replace(/\[.*\]$/, '') === routed ? asked : routed)
const short = (model: string) => model.replace(/^claude-/, '').replace(/-(\d+)-(\d+)$/, ' $1.$2')

// undefined when the daemon is down or refuses.
const classify = async ($: $, text: string): Promise<Classified | undefined> =>
  $.http
    .fetch(`${await daemon($)}/classify`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ text: text.slice(0, 2000), choose: TASKS, flags: SIGNALS, effort: LADDER, pick: PICKS }),
    })
    .then(
      r => (r.ok ? JSON.parse(r.text) : undefined),
      () => undefined,
    )

// `model` given keeps that model (a session's already picked); left out, the classification picks one.
const routeFor = (found: Classified, model?: string | null): Route => {
  const s = score(found.choose, found.flags)
  // Fable's score threshold outranks the adapter, which only chooses between Sonnet and Opus.
  const byScore = modelFor(s)
  return {
    model: model !== undefined ? model : found.pick && byScore !== MODELS.fable ? MODELS[top<(typeof PICKS)[number]>(found.pick)] : byScore,
    // An adapter tuned on your own prompts beat the score on held-out ones before it was installed.
    effort: found.effort ? top(found.effort) : effortFor(s),
    score: s,
    task: found.choose,
    signals: found.flags,
    ms: found.ms,
  }
}

// A subagent's conversation is its own, so it can take a model of its own at no cost to the main loop's cache;
// it is routed once, on its first request, from the task it was given. One whose first request carries more
// than that task (a fork, which shares its parent's context and cache) is left alone, as is a missed classification.
const routeAgent = async ($: $, agentId: string, messageCount: number, model?: null) => {
  if (messageCount !== 1) return null
  const messages = await $.session.messages({ agentId })
  const task = Array.isArray(messages) ? messages.find(m => m.role === 'user')?.text : undefined
  const found = task ? await classify($, task) : undefined
  return found === undefined ? null : routeFor(found, model)
}

export const register: Register = (on, options) => {
  // userConfig `model: keep` never picks a model, for the main loop or its subagents: the session stays on
  // the one you chose and only effort is routed.
  const kept = options.model === 'keep' ? null : undefined

  // Start the shared daemon if no session has: it outlives this one, so only the first session after a
  // reboot pays the model load (~10s; the first run ever also installs torch and downloads the weights).
  on('session.start', async ($, e, next) => {
    const started = await next(e)
    const url = await daemon($)
    const isUp = await $.http.fetch(`${url}/health`).then(
      r => r.ok,
      () => false,
    )
    if (!isUp) {
      const log = `${(await $.env.get('HOME')) ?? '/tmp'}/.cache/magic-router/classifier.log`
      // The subshell's own redirect matters: a backgrounded list without one keeps run's stdout pipe open
      // until its 30s timeout. uv lands in ~/.local/bin without touching shell profiles, where gliner.sh looks too.
      await $.process.run([
        'sh',
        '-c',
        'mkdir -p "$(dirname "$2")"; ( ROUTER_PORT="$3"; export ROUTER_PORT; PATH="$HOME/.local/bin:$PATH"; command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | UV_NO_MODIFY_PATH=1 sh; exec nohup uv run --script "$1" ) </dev/null >>"$2" 2>&1 &',
        'sh',
        `${$.plugin.root}/server/classifier.py`,
        log,
        url.split(':').pop()!,
      ])
    }
    return started
  })

  on('prompt.submit', async ($, e, next) => {
    if (!ROUTED_ORIGINS.has(e.origin.kind) || e.text.startsWith('/')) return next(e)

    const picked = await read($, isPicked)
    if (picked && isFollowUp(e.text)) return next(e)

    const found = await classify($, e.text)

    if (found === undefined) {
      // A first prompt the classifier missed still settles the model (the session's own): picking one
      // later would switch models mid-conversation.
      await update($, isPicked, () => true)
      const url = await daemon($)
      await update($, note, () => `classifier unreachable at ${url}; see ~/.cache/magic-router/classifier.log`)
      return next(e)
    }

    const last = await read($, route)
    const chosen = routeFor(found, picked ? (last?.model ?? null) : kept)
    await update($, isPicked, () => true)
    await update($, route, () => chosen)
    await update($, isHidden, () => false)
    if ((await read($, note))?.startsWith('classifier')) await update($, note, () => null)
    return next(e)
  })

  on('turn.step', async function* ($, e, next) {
    const r = await read($, route)
    if (r === null) return yield* next(e)

    const was = await read($, baseline)
    if (e.agentId !== undefined) {
      if (was === null || (await read($, note))?.startsWith('stood aside')) return yield* next(e)
      const id = e.agentId
      let mine = (await read($, agents))[id]
      if (mine === undefined) {
        mine = await routeAgent($, id, e.messageCount, kept)
        await update($, agents, a => ({ ...a, [id]: mine! }))
      }
      if (mine === null) return yield* next(e)
      // What the subagent inherited from the session follows its route; what its caller or definition pinned stays.
      return yield* next({
        ...e,
        model: e.model === was.model ? onto(e.model, mine.model) : e.model,
        effort: (e.effort ?? null) === was.effort ? mine.effort : e.effort,
      })
    }
    const asked = { model: e.model, effort: e.effort ?? null }
    if (was === null) {
      await update($, baseline, () => asked)
    } else if (was.model !== asked.model || was.effort !== asked.effort) {
      // /model, /effort or a fallback changed what the engine asks for: stand aside for the session.
      if ((await read($, note)) === null) await update($, note, () => `stood aside: ${asked.model} chosen by hand`)
      return yield* next(e)
    }
    return yield* next({ ...e, model: onto(e.model, r.model), effort: r.effort })
  })

  on('turn.complete', async ($, e, next) => {
    if (e.agentId === undefined && e.usage !== undefined) {
      const u = e.usage
      await update($, cache, () => ({
        read: u.cache_read_input_tokens,
        written: u.cache_creation_input_tokens,
        uncached: u.input_tokens,
      }))
    }
    return next(e)
  })

  // A /clear starts a new conversation: the next prompt picks a model again.
  on('session.end', async ($, e, next) => {
    if (e.reason === 'clear') {
      await update($, isPicked, () => false)
      await update($, route, () => null)
      await update($, baseline, () => null)
      await update($, note, () => null)
      await update($, cache, () => null)
      await update($, agents, () => ({}))
    }
    return next(e)
  })

  on('ui.render', { component: 'AbovePrompt' }, async ($, e, next) => {
    const [r, n, c] = [await read($, route), await read($, note), await read($, cache)]
    if (e.props.hasSurvey || (r === null && n === null) || (await read($, isHidden))) return next(e)

    const { Box, Button, Text } = $.ui.resolve(e)
    const total = c === null ? 0 : c.read + c.written + c.uncached

    return (
      <Box flexDirection="column">
        <Box flexDirection="row" gap={2}>
          <Box width={6}>
            <Text bold>router</Text>
          </Box>
          {r !== null && <Text>{r.model === null ? 'session model' : short(r.model)}</Text>}
          {r !== null && (
            <Box flexDirection="row" gap={1}>
              <Text dimColor>effort</Text>
              <Text>
                {LADDER.map((e, i) => (
                  <Text key={e} color={HEAT[e]} dimColor={i > LADDER.indexOf(r.effort)}>
                    {i <= LADDER.indexOf(r.effort) ? '▰' : '▱'}
                  </Text>
                ))}
              </Text>
              <Text bold color={HEAT[r.effort]}>
                {r.effort}
              </Text>
            </Box>
          )}
          <Text dimColor wrap="truncate">
            {[r && `${r.ms}ms`, total > 0 && `cache ${pct(c!.read / total)} read`]
              .filter(Boolean)
              .join(' · ')}
          </Text>
          <Button key="hide" label="hide" onPress={() => update($, isHidden, () => true)} />
        </Box>
        {n !== null && (
          <Text color="warning" wrap="truncate">
            {n}
          </Text>
        )}
      </Box>
    )
  })
}
