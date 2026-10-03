# Lean ecosystem: fewer tools, git-backed skills

Source: end of session `20261002_141711_389268` (Pi 1.0 / Pi Durable comparison,
"ecosystem + shrink tools/skills + model edits skills itself + trusted sources").
Evidence: code read 2026-10-03 + `sessions.db` tool-call counts since 2026-03-03.

## Verdict on the session's proposals

| Proposal | Verdict | Why (evidence) |
|---|---|---|
| Merge 25 tools -> ~8 action-param tools | **Drop** | Local surface is already ONE `tool` proxy (~150 tok, `registry.rs:834`). Action-param mega-tools caused proxy/native arg confusion before (`registry.rs` CORE_NATIVE_TOOLS note, commit f03c6e8). Zero token gain. |
| Skill manifests (requires_tools/deps, version) | **Drop** | `requires.bins` / `requires.env` already exist (`skills.rs:545-590`). |
| Trusted source index + sha256 + install tool | **Replace with git** | git already does fetch, integrity, versioning, diff, rollback. |
| Model add/remove/change skills | **Mostly exists** | Model writes SKILL.md via `write_file`/`edit_file` and removes skills via `exec`. `create_skill` was never called in 7 months. What's missing is undo, and git provides it. |
| WASM / native plugins | **Drop (YAGNI)** | Skills can already ship scripts that `exec` runs. Add plugins only when a skill can't do something with `exec`. |
| Multi-user / concurrent conversations | **Out of scope** | Not requested for this plan. |

## Usage data (model tool calls, 7 months)

Hot: exec 5188, read_file 2853, inspect_tool_result 1346, search_files 1291,
web_fetch 1249, list_dir 843, web_search 777, recall 454, write_file 303.
Cold (<= 30): find_files 27, browser 27, system_info 24, tool_status 11,
file_info 10, workspace_diff 7, file_preview 7.
Never called by the model: create_skill, apply_patch, send_email, cron_schedule,
execute_code (verify exact registered names in Task 3).
Skills misused as tool names: newsreader 12, weather 2, news_reader, webradio, MEMORY.
Skills actually loaded: newsreader 29, webradio 11, brutalist 11, lightpanda 6,
rss-agent-viewer 3, codebase-investigation 2; the rest 0-1.

## Tasks (in order; each one is small and reversible)

### Task 1: put the skills dir under git (no code)
- `git init` `~/.nanobot/workspace/skills`, commit the current state.
- Every skill the model creates, edits or deletes then shows up as a diff, and undo is `git checkout`/`git revert`.
- Move stray root files (`news.py`, `rss-reader.py`, `rss-mcp-builder.swift`) into their skill dirs, or delete them.
- Done when: `git -C ~/.nanobot/workspace/skills status` is clean.

### Task 2: one `skills` skill = the whole ecosystem (no code)
- Write `skills/skills/SKILL.md` (~30 lines) teaching the model:
  - create/change a skill: `write_file`/`edit_file` on `skills/<name>/SKILL.md`, then `git commit`
  - remove a skill: `git rm -r skills/<name>`, then `git commit`
  - install from a trusted source: `git clone <url>` / `git subtree add`, only for remotes listed in `skills/TRUSTED`, a list that only the user edits
  - update: `git pull`; review: `git log -p`
- `ponytail:` trust enforcement here is advisory: the model could ignore `TRUSTED`.
  The upgrade path is a ~20-line check in the exec guard that rejects `git clone`/`pull` of remotes not listed in `TRUSTED`. Add it only if the model actually ignores the rule.
- Done when: a live session installs one skill from a trusted remote, edits it, and reverts it, and each step shows in `git log`.

### Task 3: delete dead tools (code)
- Candidates: create_skill, apply_patch, send_email, cron_schedule, execute_code;
  then the cold group exec already covers: file_preview (`head`), file_info (`stat`),
  workspace_diff (`git diff`), system_info (`uname`/`sysctl`), tool_status, find_files (`fd`).
- Before deleting each one, confirm it isn't wired into channels, heartbeat, cron or gateway (`rg` the registration). **The user signs off on the final list.**
- Benefit: less code and a smaller proxy catalog. Honest ceiling: **no prompt-token gain on the local path** (the proxy is already one definition), so the win is maintenance, not speed.
- Done when: `cargo test`, `cargo clippy --all-targets` and `./quality-sentinel.sh` show no new failures, and `test_lean_definitions_surface` still passes.

### Task 4: `tool(<skill name>)` loads the skill (code, ~10 lines + 1 test)
- In the `tool` proxy, when the requested name isn't a registered tool but matches a skill, return the skill body, exactly as `get_skills(name)` does.
- Fixes the observed misuse (newsreader/weather/webradio called as tools) without teaching the model a new rule.
- Test: the proxy called with a skill name returns that skill's content, and an unknown name still errors.
- Done when: the test passes, along with the full `cargo test`.

### Task 5: prune skills (no code)
- Keep: newsreader, webradio, brutalist, lightpanda, rss-agent-viewer, codebase-investigation, skills (new).
- Remove via `git rm` (recoverable): evolution, listen, mcp-builder, rss-mcp-builder,
  speed-tools, weather, speak, zhipu-glm. **The user signs off on the list.**

## Explicitly not doing
Action-param mega-tools, manifest schema, signed package index, WASM, MCP client, multi-user.
Revisit a skipped item only when a real session needs something that git + exec + skills can't do.
