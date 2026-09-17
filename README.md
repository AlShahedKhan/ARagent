# Minimal Coding Agent

A from-scratch coding agent in ~250 lines of Python, built directly on the
Anthropic API's tool-use feature. No frameworks — this is the actual core
loop every coding agent (Claude Code, Aider, SWE-agent, etc.) is built from,
so it's meant to be read, not just run.

## Install it as a real CLI command

This ships as a proper installable package (`pyproject.toml` included), so
anyone can get a `coding-agent` command on their PATH — no need to run
`python agent.py` or remember where the file lives.

```bash
# From this folder (agent.py + pyproject.toml + README.md together):
pip install .

# Now available anywhere, from any directory:
coding-agent --help
```

**For development** (edits to agent.py take effect immediately, no reinstall):
```bash
pip install -e .
```

**To share it with others**, push this folder to a GitHub repo, then anyone
can install it directly from the repo with [pipx](https://pipx.pypa.io)
(recommended — installs into an isolated environment, keeps the `pip
install` out of their other Python projects) or plain pip:

```bash
pipx install git+https://github.com/<you>/coding-agent-cli.git
# or
pip install git+https://github.com/<you>/coding-agent-cli.git
```

**To make it a `pip install coding-agent-cli`-from-anywhere package** (the
"anyone" version — no repo access needed), publish it to PyPI:

```bash
pip install build twine
python -m build              # produces dist/*.whl and dist/*.tar.gz
twine upload dist/*          # prompts for your PyPI credentials
```

After that, `pip install coding-agent-cli` (or `pipx install
coding-agent-cli`) works for anyone, worldwide, with no GitHub access
needed. Bump the `version` in `pyproject.toml` before each re-upload —
PyPI rejects re-uploading the same version number.

## Set your API key(s)

Set the key for whichever provider(s) you'll use — the person running the
CLI needs their own key, same as any API-based tool:

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # console.anthropic.com — provider=anthropic
export OPENAI_API_KEY=sk-...          # platform.openai.com  — provider=openai
export ZHIPU_API_KEY=...              # z.ai / open.bigmodel.cn — provider=zhipu (GLM)
```

## Run it

```bash
# Interactive mode — defaults to Claude
coding-agent

# Choose the model backend
coding-agent --provider openai         # GPT-5.6 Sol
coding-agent --provider zhipu          # GLM-5.3

# Override the exact model string for the chosen provider
coding-agent --provider anthropic --model claude-opus-5

# One-shot task, and/or point it at a specific project directory
coding-agent --workdir /path/to/your/project "write a Python function that checks if a string is a palindrome, save it to utils.py"
```

Every flag has a matching environment variable if you prefer that instead
(`--provider` → `AGENT_PROVIDER`, `--model` → `AGENT_MODEL`, `--workdir` →
`AGENT_WORKDIR`) — flags take precedence when both are set. Running the
file directly (`python agent.py ...`) still works identically to
`coding-agent ...` if you haven't installed it.

### A note on model names

- **Claude**: defaults to `claude-fable-5-1`. Other current strings:
  `claude-sonnet-5`, `claude-opus-5`, `claude-haiku-4-5-20251001`.
- **GPT**: defaults to `gpt-5.6-sol`, OpenAI's current flagship as of this
  writing. GPT-6 has not shipped a public API yet — it's still rumored/
  leaked, not officially released. Once OpenAI publishes its real API model
  string, just set `AGENT_MODEL` to it (or edit the default in
  `build_provider()` in agent.py) — nothing else needs to change.
- **GLM**: defaults to `glm-5.3` via Z.ai's OpenAI-compatible endpoint
  (`https://api.z.ai/api/paas/v4/`). Verify the current model string and
  endpoint in Z.ai's docs before relying on this, since providers do rename
  and re-version models over time.

### Adding another provider

Any OpenAI-compatible API (most third-party model providers are) can reuse
`OpenAICompatibleProvider` — just give it a label, model string, API-key
env var name, and base URL, then add a branch for it in `build_provider()`
in agent.py. For a genuinely different API shape, implement the `Provider`
interface (`send`, `append_tool_results`) the way `AnthropicProvider` does.

## What it can do

Works the same way regardless of which model is behind it — the tool loop,
sandboxing, and `AGENT_RULES.md` conventions file apply no matter which
provider you pick. Five tools, which is enough to do real work:

| Tool | Purpose |
|---|---|
| `read_file` | Read a file with line numbers |
| `write_file` | Create a file or fully overwrite one |
| `edit_file` | Find-and-replace a unique snippet (safer than full rewrites) |
| `list_files` | List a directory |
| `run_bash` | Run any shell command (tests, git, installs, etc.) |

## The core loop (`agent_loop` in agent.py)

This is the part worth understanding line by line:

1. Send the conversation history + tool definitions to Claude.
2. Claude replies with either plain text (done) or one or more `tool_use`
   blocks (it wants to act).
3. If it's tool calls, execute each one locally and package the results as
   `tool_result` blocks.
4. Append both Claude's tool-call message and your tool-result message to
   the conversation, and loop back to step 1.
5. Stop when Claude replies with no tool calls — that's its way of saying
   "I'm done."

Every other feature (better prompting, more tools, safety checks) is
layered on top of this loop.

## Safety notes — read before pointing this at anything important

- **Sandboxing is minimal.** `_safe_path()` only stops the agent from
  escaping `WORKDIR` via file tools — it does **not** stop `run_bash` from
  doing anything a shell can do (installing packages, network calls,
  deleting files, etc.). Don't point `AGENT_WORKDIR` at anything you're not
  willing to have modified, and consider running this inside a container or
  VM for anything beyond toy projects.
- **No human-in-the-loop by default.** The agent doesn't currently pause to
  ask permission before writing files or running commands. For real use,
  add a confirmation prompt before `run_bash` and `write_file` calls —
  that's a natural next feature to build.
- **`max_turns=40`** caps runaway loops, but the agent can still burn a lot
  of API calls/tokens on a hard task. Watch usage while testing.

## Where to take it next

Roughly in order of value:

1. **Confirmation prompts** before risky tool calls (writes, bash) — biggest
   safety win for the least effort.
2. **Diff display** before applying `edit_file`/`write_file` changes, so you
   can see what's about to happen.
3. **A real sandbox** — run `run_bash` (and ideally file writes too) inside
   a Docker container instead of directly on your machine.
4. **Context management** — for real codebases you can't just read every
   file; add a `grep`/search tool and teach the agent to explore before
   editing.
5. **Persistent conversation across runs** — currently each task starts a
   fresh `messages` list; you could save/load history to build up
   longer-running sessions.
6. **Streaming output** — use the streaming variant of `messages.create` so
   you see Claude's text as it's generated instead of after each turn.

## Testing it

Try something small first:

```bash
mkdir /tmp/agent-test && cd /tmp/agent-test
AGENT_WORKDIR=/tmp/agent-test python /path/to/agent.py "create a file called hello.py that prints 'Hello, world!', then run it"
```

You should see it call `write_file`, then `run_bash`, then report success.
"# ARagent" 
