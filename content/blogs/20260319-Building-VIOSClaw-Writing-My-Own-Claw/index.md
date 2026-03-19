---
title: "Building ViosClaw: Writing My Own Claw: First Setup"
date: 2026-03-19
draft: false
tags: ["viosclaw", "openclaw", "agent", "LLM"]
series: ["viosclaw"]
series_order: 1
categories: ["engineering", "ai"]
description: "Why I stopped trusting 500k lines of code I couldn't read, and started building my own minimal agent instead."
---

{{< figure src="viosclaw-logo.png" alt="ViosClaw logo" >}}

{{< lead >}}
OpenClaw is everywhere right now. The news cycle is relentless — new derivatives, new forks,
new "claw" products launching every week. `NanoClaw`, `NemoClaw`, CookingClaw, anything-Claw.
It's gotten to the point where I genuinely can't tell if something is a serious security-focused
agent platform or just someone's weekend project with a lobster emoji slapped on it.

Probably both.
{{< /lead >}}

I got swept up in the hype earlier this year, back when `OpenClaw` was still called `ClawdBot`.
Tried it for two days. The first impression was genuinely impressive — a local AI agent that
actually takes actions on your machine feels like living in the future.

Then something in my brain went *wait a minute*.

Two things I couldn't get past:

- **Security isolation — or the total absence of it.** There's nothing stopping the agent from
touching your system files. No warning, no sandbox, no "are you sure?". Your home directory is
just... exposed. And running a local server means open ports, which on a research machine is
the kind of thing that makes your sysadmin cry.
- **Half a million lines of code I can't read.** OpenClaw is enormous. Integrating something
that large into your daily workflow means extending it complete trust. I'm not comfortable doing
that with software I can't audit — the surface area for things going quietly wrong is just too big.

To be fair, some newer variants address this. **NanoClaw** takes the opposite philosophy:
strip everything down to the minimal viable agent, nothing more. **NemoClaw** goes hard on
security with a triple-layer sandbox (Landlock filesystem controls + seccomp syscall filtering +
network namespace isolation). Both are interesting. Both I also can't fully read.

## Why Build My Own

I'm part of [VIOS Group](https://vios.science), and we've been talking about AI-assisted tooling
for a while. At some point I thought: instead of trying to evaluate which black box to drop into
our workflow, why not just... build one? Minimal, auditable, shaped around how we actually work.

I'm not delusional — I know I'm not going to out-engineer a team that has 500k lines of code.
But that's not the point. The point is to understand the mechanism well enough to build something
I can actually vouch for.

Before starting, I did some reading. The `learn-claude-code` repo — a 12-session walkthrough of
building a nano agent harness from scratch — was useful for understanding the basic mental model:
*"the model is the agent, the code is the harness."* That framing clicked something for me.

This series is my dev log. No promises of a polished product, no roadmap theater.
Just small steps, in the open.

Here's what Phase 1 looks like.

## What I Built

A working terminal AI agent called **ViosClaw** — a Python package that:

- Connects to any LLM via [OpenRouter](https://openrouter.ai)
- Runs a proper agent loop (`while True` + message accumulation)
- Uses `Pydantic` for typed message handling
- Persists config via environment variables
- Ships with linting enforced on every commit

No magic. No frameworks. ~200 lines of Python.

## Project Structure

The first decision was structure, before writing any actual logic. After a couple of iterations,
I landed on a layout that separates the installable package from repo-level config:

```
viosclaw/                  ← repo root
├── pyproject.toml
├── .pre-commit-config.yaml
├── viosclaw/              ← installable Python package
│   ├── config.py
│   ├── main.py
│   └── core/
│       └── agent.py
└── data/
    ├── sessions/
    └── audit/
```

The key insight: `pyproject.toml` at the root, source code inside `viosclaw/`.
This makes it installable with `pip install -e .` and gives you a proper CLI entry point.
Spending 30 minutes on scaffolding upfront genuinely saves hours of painful retrofitting later —
learned that the hard way on previous projects.

## Tooling Choices

### uv over pip

I went with [uv](https://github.com/astral-sh/uv) — Rust-based, 10-100x faster than pip.
There's no good reason to start a new project in 2026 with older tooling:

```bash
uv pip install -e ".[dev]"
```

### Pydantic from the start

Rather than passing raw dicts around, I defined a typed `Message` model upfront:

```python
from pydantic import BaseModel

class Message(BaseModel):
    role: str
    content: str
```

When you're accumulating conversation history across a loop, having runtime validation and
IDE autocomplete on message objects is worth the two extra lines. Future-me will thank me.

## The Agent Loop

This is the core of the whole thing. Five steps, every iteration:

```python
def run_agent() -> None:
    # Initialize with a system message — this is how you set the agent's persona/instructions
    messages: list[Message] = [
        Message(role="system", content="You are a helpful assistant.")
    ]

    while True:
        # 1. Get user input
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        # 2. Append to message history
        messages.append(Message(role="user", content=user_input))

        # cast() here tells mypy the shape of our payload — no runtime cost
        messages_payload = cast(
            list[ChatCompletionMessageParam], [m.model_dump() for m in messages]
        )

        # 3. Send to LLM
        response = client.chat.completions.create(
            model=config.OPENROUTER_DEFAULT_MODEL,
            messages=messages_payload,
            max_tokens=config.MAX_TOKENS,
            temperature=config.TEMPERATURE,
        )

        # 4. Print the response
        assistant_message = response.choices[0].message
        print(f"Assistant: {assistant_message.content}")

        # 5. Append response to history — using the role from the API response, not hardcoded
        messages.append(
            Message(role=assistant_message.role, content=assistant_message.content or "")
        )
```

A few things worth noting here:

**The system message lives at index 0.** It's the agent's permanent instruction — it never
gets appended to mid-loop, just initialized once. This is how you give the agent its personality,
constraints, and context. Right now it's just "helpful assistant", but this is where ViosClaw's
actual character will live eventually.

**`cast()` is doing zero work at runtime.** It's purely for mypy — the OpenAI SDK has strict
typing for `ChatCompletionMessageParam`, and our Pydantic `Message.model_dump()` produces the
right shape. `cast()` just tells the type checker to trust us on that. No overhead, no magic.

**`assistant_message.role` instead of `"assistant"`.** Minor point, but the API tells you what
role the message has. Trusting the response rather than hardcoding is a small habit that
matters more when you add tool calls — the role won't always be `"assistant"`.

Right now this is just a chatbot. The next step — tools — is what makes it an actual agent.

## Config Done Right

Everything configurable lives in `.env`, loaded at startup:

```python
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_DEFAULT_MODEL = os.getenv("OPENROUTER_DEFAULT_MODEL", "anthropic/claude-sonnet-4-6")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
```

One thing that confused me initially: should `MAX_TOKENS` match the model's context window (400K)?
No — `MAX_TOKENS` controls *output length*, not input. 4096 is plenty for conversational replies.
The context window is a different limit entirely.

## What's Next

Phase 1 is a chatbot with good bones. Phase 2 is an actual agent. The difference: tools.

Next up:

- `core/tools.py` — tool registry and dispatch
- `core/sessions.py` — JSONL persistence so context survives restarts
- First tools: `read_file`, `write_file`, maybe `search_arxiv`

Once tools are wired in, the model can decide to *do things* rather than just reply —
and that's when the loop gets interesting.

Full code on [GitHub - VIOSClaw](https://github.com/vios-s/viosclaw). Fork it, break it, tell me what's wrong with it.
