# Prior Authorization Multi-Agent System

A four-agent system built with CrewAI that processes a prior authorization request for an MRI — extracting clinical facts, checking them against a payer's actual policy criteria, drafting an authorization letter, and reviewing that letter for honesty before it's considered final. If the Reviewer catches a letter that overstates the case, it sends the work back to the Writer with specific feedback, and the cycle repeats.

Built as a second hands-on agentic AI project, specifically to learn multi-agent coordination after building a single-agent system first.

## What This Project Does

Given one prior authorization request — a raw clinical note plus the procedure being requested — the system:

1. Extracts structured facts from the note using an LLM (diagnosis, procedure requested, conservative therapy duration, red flags, exam findings, prior imaging history)
2. Checks each of the payer's 4 criteria against those facts — scoring each one Met, Not Met, Insufficient Information, or Waived (only Criterion 2 can be waived, and only if a red flag is documented)
3. Drafts a prior authorization letter based on those results
4. Reviews that letter against the real criteria results — and if the letter overstates the case (claims approval when something is actually missing or failed), rejects it and sends specific feedback back to the Writer
5. Repeats the draft/review cycle up to 3 times, or stops immediately the moment the letter is approved as accurate

This is a genuine multi-agent system — 4 separate agents, each with a distinct role, with one capable of sending work backward to another. That backward loop is the key difference from a single agent with several internal steps.

## Architecture

```
START
  │
  ▼
Extractor Agent        (LLM — parses raw note into structured fields)
  │
  ▼
Policy Checker Agent    (LLM — scores each criterion: Met / Not Met /
  │                      Insufficient / Waived, against the real policy)
  │
  ▼
┌──────────────────────────────────────┐
│         WRITER ↔ REVIEWER LOOP         │
│                                         │
│   Writer drafts a letter                │
│         │                              │
│         ▼                              │
│   Reviewer checks it against the        │
│   real criteria results                  │
│         │                              │
│   Honest and accurate? ── NO ──┐         │
│         │                       │        │
│        YES               feedback sent   │
│         │                back to Writer  │
│         │                       │        │
│         │◄──────────────────────┘        │
│         │      (up to 3 attempts)        │
└─────────┼────────────────────────────────┘
          │
          ▼
   Final letter + approval decision
```

## Tech Stack

- **CrewAI** — multi-agent coordination (Agent, Task, Crew), built specifically around team-of-specialists vocabulary rather than generic graph nodes
- **OpenAI gpt-4o-mini** — every agent's underlying reasoning
- **pypdf** — extracting the payer policy text from PDF
- **Python** — the orchestration loop driving the Writer/Reviewer back-and-forth

## Project Structure

```
prior-authorization-multi-agent/
├── prior_auth_requests.json   # 4 mock requests — raw notes + ground-truth answers
├── payer_policy.pdf            # the 5-section policy document used for review
├── agents.py                   # all 4 agents and their tasks (who, and what each does)
├── run_prior_auth.py           # the orchestration loop (the actual back-and-forth)
└── README.md
```

## Setup

```bash
python -m pip install crewai langchain-openai python-dotenv pypdf
```

Create a `.env` file in the same folder:
```
OPENAI_API_KEY=your-key-here
```

**Important:** run this from a terminal, not a Jupyter notebook. CrewAI deliberately refuses to run synchronously whenever it detects an active event loop, which Jupyter always has running in the background. A terminal has no such loop, so it runs cleanly there.

```bash
python run_prior_auth.py
```

This processes all 4 mock requests in sequence and prints each one's final letter and decision.

## Example Output

Running PA002 — a case deliberately built with no documented conservative therapy duration:

**Attempt 1:** Writer drafts a letter that doesn't disclose the gap. Reviewer rejects it: *"The letter does not disclose that conservative therapy duration is undocumented."*

**Attempt 2:** Writer drafts a new letter, honestly stating the documentation gap and requesting it be provided. Reviewer approves it.

```json
{
  "approved": true,
  "feedback": ""
}
```

The real medical decision never changes across both attempts — Criterion 2 is "Insufficient Information," permanently, decided once in step 2. What changes is only whether the *letter* honestly communicates that, not the underlying facts. The Reviewer's job is enforcing honest communication of a real result, not re-deciding the case.

Of the 4 mock requests, 2 (PA001, PA003) pass review on the first attempt — including PA003, which has no conservative therapy at all but qualifies through a documented red-flag exception. The other 2 (PA002, PA004) each require exactly one rejection and revision before approval.

## Current Limitations

- **The loop only retains the final attempt.** A more complete audit trail would log every rejected draft and the specific feedback at each step, not just the last one.
- **The Policy Checker receives the entire policy document directly**, rather than retrieving relevant sections through vector search the way the first project does. This works fine for one short policy, but wouldn't scale to a payer whose policies cover hundreds of procedures.
- **The system is built around exactly 4 fixed criteria**, specific to lumbar spine MRI. Adapting it to a different procedure (a knee MRI, for example) would require rewriting both the policy document and the extraction schema — not just swapping a number, since a different procedure's criteria may differ entirely in content and even in count.

## Why I Built This

After building a single-agent system first, I wanted to deliberately learn when a problem actually calls for multiple agents instead of one agent with more steps. The real distinction turned out to be about dependency direction, not complexity: a single agent's steps only ever move forward. This system needed a Reviewer that could send work *backward* to the Writer — a genuine loop, not just a longer chain — which is what actually justified 4 separate agents rather than one larger one.

---

*All data in this project is synthetic, created for educational and portfolio purposes only.*
