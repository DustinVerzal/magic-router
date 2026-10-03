import { atom, read, update } from 'claude-code'
import type { Register } from 'claude-code'

import type { Effort, Probs, Route } from '../types'
import { EFFORTS, SIGNALS, TASKS, effortFor, isFollowUp, modelFor, score } from './route'

const DAEMON = 'http://127.0.0.1:8765' // ponytail: fixed port, matches ROUTER_PORT's default in server/classifier.py

const isPicked = atom({ plugin: 'model-router', key: 'isPicked' } as const, false)
const route = atom({ plugin: 'model-router', key: 'route' } as const, null)
const baseline = atom({ plugin: 'model-router', key: 'baseline' } as const, null)
const note = atom({ plugin: 'model-router', key: 'note' } as const, null)
const cache = atom({ plugin: 'model-router', key: 'cache' } as const, null)
const isHidden = atom({ plugin: 'model-router', key: 'isHidden' } as const, false)

// What a person sent; task notifications, peers and plugins keep the current route.
const ROUTED_ORIGINS = new Set(['composer', 'bridge', 'sdk'])

type Classified = { choose: Probs; flags: Probs; ms: number }

const LADDER = EFFORTS.map(([, e]) => e).reverse()
// ponytail: theme keys, so the colours follow light and dark themes; low and medium share green
const HEAT: Record<Effort, string> = { low: 'success', medium: 'success', high: 'warning', xhigh: 'error', max: 'error' }
const pct = (p: number) => `${Math.round(p * 100)}%`
const short = (model: string) => model.replace(/^claude-/, '').replace(/-(\d+)-(\d+)$/, ' $1.$2')

export const register: Register = on => {
  // Start the shared daemon if no session has: it outlives this one, so only the first session after a
  // reboot pays the model load (~10s; the first run ever also installs torch and downloads the weights).
  on('session.start', async ($, e, next) => {
    const started = await next(e)
    const isUp = await $.http.fetch(`${DAEMON}/health`).then(
      r => r.ok,
      () => false,
    )
    if (!isUp) {
      const server = `${$.plugin.root}/server/classifier.py`
      const profile = await $.env.get('USERPROFILE') // set on Windows only
      if (profile) {
        // ponytail: untested on real Windows; same flow as the sh branch below, via PowerShell
        const log = `${profile}\\.cache\\model-router\\classifier.log`
        const ps = `$env:PATH = "$HOME\\.local\\bin;$env:PATH"; New-Item -ItemType Directory -Force '${profile}\\.cache\\model-router' | Out-Null; if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { $env:UV_NO_MODIFY_PATH = '1'; irm https://astral.sh/uv/install.ps1 | iex }; Start-Process cmd -WindowStyle Hidden -ArgumentList '/c', 'uv run --script "${server}" >> "${log}" 2>&1'`
        await $.process.run(['powershell', '-NoProfile', '-Command', ps])
        return started
      }
      const log = `${(await $.env.get('HOME')) ?? '/tmp'}/.cache/model-router/classifier.log`
      // The subshell's own redirect matters: a backgrounded list without one keeps run's stdout pipe open
      // until its 30s timeout. uv lands in ~/.local/bin without touching shell profiles, where gliner.sh looks too.
      await $.process.run([
        'sh',
        '-c',
        'mkdir -p "$(dirname "$2")"; ( PATH="$HOME/.local/bin:$PATH"; command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | UV_NO_MODIFY_PATH=1 sh; exec nohup uv run --script "$1" ) </dev/null >>"$2" 2>&1 &',
        'sh',
        server,
        log,
      ])
    }
    return started
  })

  on('prompt.submit', async ($, e, next) => {
    if (!ROUTED_ORIGINS.has(e.origin.kind) || e.text.startsWith('/')) return next(e)

    const picked = await read($, isPicked)
    if (picked && isFollowUp(e.text)) return next(e)

    const found: Classified | undefined = await $.http
      .fetch(`${DAEMON}/classify`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ text: e.text.slice(0, 2000), choose: TASKS, flags: SIGNALS }),
      })
      .then(
        r => (r.ok ? JSON.parse(r.text) : undefined),
        () => undefined,
      )

    if (found === undefined) {
      // A first prompt the classifier missed still settles the model (the session's own): picking one
      // later would switch models mid-conversation.
      await update($, isPicked, () => true)
      await update($, note, () => `classifier unreachable at ${DAEMON}; see ~/.cache/model-router/classifier.log`)
      return next(e)
    }

    const s = score(found.choose, found.flags)
    const last = await read($, route)
    const chosen: Route = {
      model: picked ? (last?.model ?? null) : modelFor(s),
      effort: effortFor(s),
      score: s,
      task: found.choose,
      signals: found.flags,
      ms: found.ms,
    }
    await update($, isPicked, () => true)
    await update($, route, () => chosen)
    await update($, isHidden, () => false)
    if ((await read($, note))?.startsWith('classifier')) await update($, note, () => null)
    return next(e)
  })

  // The main loop's requests only: subagents keep the model and effort they were given.
  on('turn.step', async function* ($, e, next) {
    const r = await read($, route)
    if (e.agentId !== undefined || r === null) return yield* next(e)

    const was = await read($, baseline)
    const asked = { model: e.model, effort: e.effort ?? null }
    if (was === null) {
      await update($, baseline, () => asked)
    } else if (was.model !== asked.model || was.effort !== asked.effort) {
      // /model, /effort or a fallback changed what the engine asks for: stand aside for the session.
      if ((await read($, note)) === null) await update($, note, () => `stood aside: ${asked.model} chosen by hand`)
      return yield* next(e)
    }
    return yield* next({ ...e, model: r.model ?? e.model, effort: r.effort })
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
