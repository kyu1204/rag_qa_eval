---
name: omh-loop
description: Set up and start the autonomous loop for a goal. Use when asked to run work as a loop, in a loop, or unattended.
---

# Autonomous loop

You are the architect, not the loop. Set it up, start it, then carry on with
your own work — the loop reports through its event log.

1. Write `WORKPLAN.md`: the goal as verifiable gates, tasks as checkboxes grouped
   by phase, a decisions log and a progress log. Exclude anything needing the
   user's own action (store submission, console access) from the goal.
2. Write a work order per task in `docs/work-orders/<ID>.md`: file paths, signatures,
   thresholds, exact copy, forbidden changes, acceptance criteria and the command
   that verifies it. Include work outside the code the task implies — deploy
   scripts, env, migrations, allowlists — the loop will not infer them.
   Queue several ahead: a loop with no work order left marks BLOCKED and waits.
3. Name the files only you may touch in `loop.architectOnly` (harness.yaml, then
   `omh sync`). An abstract "do not improvise" is not obeyed; a named path is.
4. Start it: `omh loop start`. It returns immediately; the loop runs detached in its own git worktree (branch `omh-loop`) so you can keep working.
5. Attach monitoring immediately — do not wait to be asked for progress:
   `omh loop status` prints the run, then `tail -f .omh/state/loop/runs/<runId>/events.jsonl`.
   Watch for failure kinds too (`blocked`, `idle`, `limit`, `crash`, `error`), not just
   `progress`: otherwise a stalled loop looks exactly like a quiet one.

## Rules the loop follows

- Read `WORKPLAN.md` first; it is the single source of truth.
- Pick the next unchecked task and implement it exactly as its work order in `docs/work-orders/<ID>.md` says. Make no design decisions.
- No work order, no work: mark the task "BLOCKED: no work order" and stop. Never write your own work order.
- Run the work order's acceptance commands before ticking any checkbox.
- Architect-only, never edited by the loop: none declared — add paths to `loop.architectOnly` in harness.yaml.
- If a task needs a human, or after three failed attempts, mark it "BLOCKED: <reason>" and move on. Never idle waiting for a person.
- One task, one commit; update the checkbox and the progress log in the same commit.
- Print exactly OMH_GOAL_COMPLETE as the final line ONLY when every task is done and verified; otherwise never mention that string in any form.

Start: `omh loop start`
Watch: `omh loop status`, then `tail -f .omh/state/loop/runs/<runId>/events.jsonl`
Stop: `omh loop stop` (`--now` to skip the grace period)
