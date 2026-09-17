#!/usr/bin/env python3
"""
A minimal coding agent with a pluggable model backend. It can read files,
write files, make precise edits, list directories, and run shell commands —
and it can do all of that talking to Claude, GPT, or GLM, chosen at startup.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...   # for provider=anthropic (default)
    export OPENAI_API_KEY=sk-...          # for provider=openai
    export ZHIPU_API_KEY=...              # for provider=zhipu (GLM)

    python agent.py                                  # uses AGENT_PROVIDER (default: anthropic)
    AGENT_PROVIDER=openai python agent.py             # GPT
    AGENT_PROVIDER=zhipu python agent.py              # GLM
    python agent.py "add a docstring to every function in utils.py"

How it works (the core agent loop — provider-agnostic):
    1. Send the conversation + tool definitions to the model.
    2. If it responds with a tool call, execute that tool locally.
    3. Send the tool's result back as a tool result.
    4. Repeat until the model responds with plain text (no more tool calls) —
       that means it thinks the task is done.

Adding a new provider: implement the Provider interface below (send,
append_assistant_message, append_tool_results) and add it to PROVIDERS.
Any OpenAI-compatible endpoint (most third-party model APIs are) can reuse
OpenAICompatibleProvider with a different base_url/model/api key env var.
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from abc import ABC, abstractmethod

MAX_TOKENS = 4096

# Name of the file the agent looks for at startup to learn YOUR conventions —
# coding style, architecture notes, things to always/never do. This is how
# you "teach" it your codebase's rules without retraining anything: it's
# just plain text injected into the system prompt on every run. Put this
# file at the root of whatever project the agent is pointed at.
RULES_FILENAME = "AGENT_RULES.md"

# These are set by configure() once the CLI args/env vars are known — kept
# as module globals since the tool functions below (tool_read_file etc.)
# need to see WORKDIR without threading it through every call.
WORKDIR: Path
_PROJECT_RULES: str
SYSTEM_PROMPT: str


def configure(workdir: str | None = None) -> None:
    """Resolve the working directory, load project rules, and build the
    system prompt. Called once at startup with the --workdir flag (or the
    AGENT_WORKDIR env var, or '.' as a last resort)."""
    global WORKDIR, _PROJECT_RULES, SYSTEM_PROMPT
    WORKDIR = Path(workdir or os.environ.get("AGENT_WORKDIR", ".")).resolve()

    rules_path = WORKDIR / RULES_FILENAME
    _PROJECT_RULES = rules_path.read_text().strip() if rules_path.exists() else ""

    SYSTEM_PROMPT = f"""You are a careful, autonomous coding agent. Your working \
directory is {WORKDIR}. You can read files, write files, make targeted edits, \
list directories, and run shell commands to accomplish coding tasks.

Guidelines:
- Explore before you edit: read relevant files first so you understand the \
existing code and conventions before changing anything.
- Prefer edit_file (find-and-replace) over write_file for existing files — \
it's less likely to accidentally destroy unrelated code. Use write_file only \
for brand-new files or full rewrites.
- Make sure any find-and-replace text you use is unique in the file, or the \
edit will fail — include enough surrounding context to make it unambiguous.
- After making changes, verify your work: run tests, run the code, or at \
least re-read the file to confirm the edit applied as intended.
- When you believe the task is complete, stop calling tools and reply with a \
plain-text summary of what you did. That's the signal the loop ends on.
- If something is ambiguous or risky (e.g. deleting files, running \
destructive commands), explain your plan in text first rather than just \
doing it.
"""
    if _PROJECT_RULES:
        SYSTEM_PROMPT += f"""

Project-specific rules (from {RULES_FILENAME} — these are hard requirements, \
not suggestions; follow them in every file you write or edit, and apply them \
retroactively if you touch code that violates them):

{_PROJECT_RULES}
"""


# Configure with defaults immediately so the module works if imported or run
# without going through main() (e.g. in a REPL or test). main() reconfigures
# with the parsed --workdir before anything else happens.
configure()

# Tools are defined once, in a provider-neutral shape (name, description,
# JSON-schema input). Each provider adapts this shape to whatever its API
# expects (Anthropic's input_schema vs. OpenAI-style function calling).
TOOL_DEFS = [
    {
        "name": "read_file",
        "description": "Read the full contents of a file, with line numbers.",
        "schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to the working directory."}
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Create a new file or completely overwrite an existing one with the given content.",
        "schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to the working directory."},
                "content": {"type": "string", "description": "Full file content to write."},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": (
            "Replace an exact, unique snippet of text in an existing file with new text. "
            "Safer than write_file for existing files since it only touches what you specify. "
            "old_str must match the file's current content exactly and appear exactly once."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to the working directory."},
                "old_str": {"type": "string", "description": "Exact text to find (must be unique in the file)."},
                "new_str": {"type": "string", "description": "Text to replace it with."},
            },
            "required": ["path", "old_str", "new_str"],
        },
    },
    {
        "name": "list_files",
        "description": "List files and directories at a given path (non-recursive).",
        "schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to the working directory. Use '.' for the root."}
            },
            "required": ["path"],
        },
    },
    {
        "name": "run_bash",
        "description": (
            "Run a shell command in the working directory and return stdout/stderr/exit code. "
            "Use for running tests, installing packages, git commands, etc."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to run."}
            },
            "required": ["command"],
        },
    },
]


def _safe_path(rel_path: str) -> Path:
    """Resolve a path and make sure it stays inside WORKDIR (basic sandboxing)."""
    p = (WORKDIR / rel_path).resolve()
    if WORKDIR not in p.parents and p != WORKDIR:
        raise ValueError(f"Refusing to access path outside working directory: {rel_path}")
    return p


def tool_read_file(path: str) -> str:
    p = _safe_path(path)
    if not p.exists():
        return f"Error: file not found: {path}"
    text = p.read_text(errors="replace")
    numbered = "\n".join(f"{i+1:>5}\t{line}" for i, line in enumerate(text.splitlines()))
    return numbered or "(empty file)"


def tool_write_file(path: str, content: str) -> str:
    p = _safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"Wrote {len(content)} bytes to {path}"


def tool_edit_file(path: str, old_str: str, new_str: str) -> str:
    p = _safe_path(path)
    if not p.exists():
        return f"Error: file not found: {path}"
    text = p.read_text()
    count = text.count(old_str)
    if count == 0:
        return "Error: old_str not found in file. Check exact whitespace/text."
    if count > 1:
        return f"Error: old_str appears {count} times — must be unique. Add more surrounding context."
    p.write_text(text.replace(old_str, new_str, 1))
    return f"Edit applied to {path}"


def tool_list_files(path: str) -> str:
    p = _safe_path(path)
    if not p.exists():
        return f"Error: path not found: {path}"
    if not p.is_dir():
        return f"Error: not a directory: {path}"
    entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name))
    lines = [f"{'[dir] ' if e.is_dir() else '      '}{e.name}" for e in entries]
    return "\n".join(lines) or "(empty directory)"


def tool_run_bash(command: str) -> str:
    try:
        result = subprocess.run(
            command, shell=True, cwd=WORKDIR,
            capture_output=True, text=True, timeout=60,
        )
        out = f"exit code: {result.returncode}\n"
        if result.stdout:
            out += f"stdout:\n{result.stdout}\n"
        if result.stderr:
            out += f"stderr:\n{result.stderr}\n"
        return out
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 60s"


DISPATCH = {
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "edit_file": tool_edit_file,
    "list_files": tool_list_files,
    "run_bash": tool_run_bash,
}


def run_tool(name: str, tool_input: dict) -> str:
    fn = DISPATCH.get(name)
    if not fn:
        return f"Error: unknown tool '{name}'"
    try:
        return fn(**tool_input)
    except Exception as e:
        return f"Error running {name}: {e}"


# --------------------------------------------------------------------------
# Provider abstraction
#
# A Provider's job: given the conversation so far, call its model API, print
# whatever text the model produced, return the list of tool calls it wants
# to make (empty list = done), and know how to append both its own message
# and the resulting tool outputs back onto the conversation in its native
# format. The agent_loop below only talks to this interface, never to a
# specific SDK — that's what makes switching models a one-line change.
# --------------------------------------------------------------------------

class ToolCall:
    def __init__(self, id: str, name: str, arguments: dict):
        self.id = id
        self.name = name
        self.arguments = arguments


class Provider(ABC):
    label = "provider"

    @abstractmethod
    def send(self, messages: list) -> tuple[list[ToolCall], bool]:
        """Call the model, print any text it wrote, return (tool_calls, done)."""

    @abstractmethod
    def append_tool_results(self, messages: list, tool_calls: list[ToolCall], results: list[str]) -> None:
        """Append the tool outputs to the conversation in this provider's format."""


class AnthropicProvider(Provider):
    label = "Claude"

    def __init__(self, model: str):
        try:
            import anthropic
        except ImportError:
            print("Missing dependency. Install it with:\n  pip install anthropic")
            sys.exit(1)
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Set ANTHROPIC_API_KEY in your environment first.")
            sys.exit(1)
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.tools = [
            {"name": t["name"], "description": t["description"], "input_schema": t["schema"]}
            for t in TOOL_DEFS
        ]

    def send(self, messages):
        response = self.client.messages.create(
            model=self.model, max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT, tools=self.tools, messages=messages,
        )
        for block in response.content:
            if block.type == "text" and block.text.strip():
                print(f"\n\033[94m{self.label}:\033[0m {block.text}")
        messages.append({"role": "assistant", "content": response.content})

        tool_calls = [
            ToolCall(b.id, b.name, b.input) for b in response.content if b.type == "tool_use"
        ]
        self._last_calls = tool_calls
        return tool_calls, response.stop_reason != "tool_use"

    def append_tool_results(self, messages, tool_calls, results):
        messages.append({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": tc.id, "content": r}
                for tc, r in zip(tool_calls, results)
            ],
        })


class OpenAICompatibleProvider(Provider):
    """Works for OpenAI's own API and for any OpenAI-compatible endpoint
    (GLM/Z.ai, and most other third-party model providers)."""

    def __init__(self, label: str, model: str, api_key_env: str, base_url: str | None = None):
        try:
            from openai import OpenAI
        except ImportError:
            print("Missing dependency. Install it with:\n  pip install openai")
            sys.exit(1)
        api_key = os.environ.get(api_key_env)
        if not api_key:
            print(f"Set {api_key_env} in your environment first.")
            sys.exit(1)
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.label = label
        self.tools = [
            {
                "type": "function",
                "function": {"name": t["name"], "description": t["description"], "parameters": t["schema"]},
            }
            for t in TOOL_DEFS
        ]

    def send(self, messages):
        full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
        response = self.client.chat.completions.create(
            model=self.model, max_tokens=MAX_TOKENS,
            messages=full_messages, tools=self.tools,
        )
        msg = response.choices[0].message
        if msg.content and msg.content.strip():
            print(f"\n\033[94m{self.label}:\033[0m {msg.content}")

        # Store the raw assistant message (OpenAI's SDK message object) so
        # we can append it in its native shape below.
        assistant_entry = {"role": "assistant", "content": msg.content or ""}
        tool_calls = []
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": tc.id, "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCall(tc.id, tc.function.name, args))

        messages.append(assistant_entry)
        done = not tool_calls
        return tool_calls, done

    def append_tool_results(self, messages, tool_calls, results):
        for tc, r in zip(tool_calls, results):
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": r})


# Provider registry. AGENT_PROVIDER selects one of these; AGENT_MODEL can
# override the default model string for whichever provider is chosen.
def build_provider(name: str) -> Provider:
    name = name.lower()
    if name == "anthropic":
        return AnthropicProvider(model=os.environ.get("AGENT_MODEL", "claude-fable-5-1"))
    if name == "openai":
        # GPT-6 has not shipped a public API as of this writing; GPT-5.6 Sol
        # is OpenAI's current flagship. Override with AGENT_MODEL=gpt-6-...
        # the day OpenAI publishes its real model string.
        return OpenAICompatibleProvider(
            label="GPT", model=os.environ.get("AGENT_MODEL", "gpt-5.6-sol"),
            api_key_env="OPENAI_API_KEY",
        )
    if name == "zhipu":
        return OpenAICompatibleProvider(
            label="GLM", model=os.environ.get("AGENT_MODEL", "glm-5.3"),
            api_key_env="ZHIPU_API_KEY", base_url="https://api.z.ai/api/paas/v4/",
        )
    print(f"Unknown provider '{name}'. Choose from: anthropic, openai, zhipu")
    sys.exit(1)


def agent_loop(provider: Provider, messages: list, max_turns: int = 40):
    for turn in range(max_turns):
        tool_calls, done = provider.send(messages)

        if done:
            return

        results = []
        for tc in tool_calls:
            print(f"\033[93m→ {tc.name}({json.dumps(tc.arguments)[:200]})\033[0m")
            result = run_tool(tc.name, tc.arguments)
            print(f"\033[90m{result[:500]}\033[0m")
            results.append(result)

        provider.append_tool_results(messages, tool_calls, results)

    print("\n(Stopped: reached max turn limit — the task may be incomplete.)")


def parse_args():
    parser = argparse.ArgumentParser(
        prog="coding-agent",
        description="An autonomous coding agent that can read/write files and run shell commands.",
    )
    parser.add_argument(
        "task", nargs="*",
        help="Task to run and exit (e.g. coding-agent \"fix the bug in utils.py\"). "
             "Omit to start an interactive session instead.",
    )
    parser.add_argument(
        "--provider", choices=["anthropic", "openai", "zhipu"],
        default=os.environ.get("AGENT_PROVIDER", "anthropic"),
        help="Which model backend to use (default: anthropic, or $AGENT_PROVIDER).",
    )
    parser.add_argument(
        "--model", default=os.environ.get("AGENT_MODEL"),
        help="Override the default model string for the chosen provider.",
    )
    parser.add_argument(
        "--workdir", default=os.environ.get("AGENT_WORKDIR"),
        help="Project directory the agent may read/write in (default: current directory).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.model:
        os.environ["AGENT_MODEL"] = args.model  # build_provider() reads this
    configure(workdir=args.workdir)

    provider = build_provider(args.provider)

    print(f"Coding agent ready. Provider: {provider.label} ({provider.model}). Working directory: {WORKDIR}")
    if _PROJECT_RULES:
        print(f"Loaded project rules from {RULES_FILENAME} ({len(_PROJECT_RULES)} chars).")
    else:
        print(f"No {RULES_FILENAME} found — create one in {WORKDIR} to teach the agent your conventions.")
    print("Type a task, or 'exit' to quit.\n")

    if args.task:
        initial_task = " ".join(args.task)
        messages = [{"role": "user", "content": initial_task}]
        agent_loop(provider, messages)
        print()

    while True:
        try:
            task = input("\n\033[92mYou:\033[0m ")
        except (EOFError, KeyboardInterrupt):
            break
        if task.strip().lower() in ("exit", "quit"):
            break
        if not task.strip():
            continue
        messages = [{"role": "user", "content": task}]
        agent_loop(provider, messages)


if __name__ == "__main__":
    main()
