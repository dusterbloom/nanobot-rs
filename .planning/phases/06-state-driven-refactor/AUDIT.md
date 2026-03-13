# If/Else & State Audit — Raw Findings

**Date:** 2026-03-13
**Method:** 8 parallel Explore agents across 12 files + grep density analysis

## Density Map (if + else-if/else combined)

| File | `if` | `else` | Combined |
|------|------|--------|----------|
| agent/context.rs | 111 | 33 | **144** |
| voice_pipeline.rs | 100 | 24 | **124** |
| repl/cmd_read.rs | 71 | 41 | **112** |
| cli/mod.rs | 69 | — | **69** |
| agent/lcm.rs | 49 | 13 | **62** |
| server.rs | 60 | — | **60** |
| agent/tool_engine.rs | 32 | 20 | **52** |
| tui.rs | 49 | 17 | **49** |
| cli/core_builder.rs | 47 | — | **47** |
| agent/agent_core.rs | 24 | 22 | **46** |
| agent/agent_shared.rs | — | 30 | **30** |
| repl/cmd_lifecycle.rs | — | 28 | **28** |
| repl/cmd_mutation.rs | 22 | 17 | **39** |
| repl/commands.rs | — | 12 | **12** |
| agent/subagent.rs | — | 19 | **19** |

## Per-File Findings

### context.rs (144 combined) — #1 offender

1. **MIME type guessing** (1630-1644) — 6-branch if/else, should be lookup table
2. **Skill disclosure mode** (326-335, 842-850) — 3-way string normalization DUPLICATED in 2 places
3. **Local vs cloud assembly** (909-979) — 2 paths × 260 lines, the single largest if/else
4. **Context window calc** (506-510, 564-568) — Duplicated with DIFFERENT defaults (16K vs 128K!)
5. **Model identity string** (1110-1127) — 3-branch string prefix dispatch
6. **Bootstrap file loading** (1370-1438) — 7+ nested conditionals, worst nesting depth
7. **PromptBlock::render()** (75-83) — 3-branch implicit 2D state
8. **User content with media** (1459-1493) — 5-branch imperative loop, should be filter_map
9. **Head/tail truncation** (1500-1557) — 80% duplicated code between two functions
10. **display_path()** (1220-1231) — Nested optional chain, minor

### voice_pipeline.rs (124 combined)

1. **strip_thinking_from_buffer()** (384-410) — 3×2 nested, `in_thinking_block: bool` flag
2. **start_tts_playback() synthesis dispatch** (1408-1448) — 3 levels deep, Language×Engine
3. **extract_sentences()** (483-527) — `in_code_block: bool` flag, 3+ branches
4. **select_tts()** (926-946) — Language→engine with fallback, 2×2 matrix
5. **synthesize_to_file()** (1164-1181) — DUPLICATE of select_tts() logic
6. **Sentence batching** (118-127) — 3-branch accumulator in loop
7. **TTS engine selection** (767-819) — Config routing, 2 match arms with nested Ok/Err

### agent_loop.rs + agent_shared.rs

**Already well-structured:**
- `IterationPhase` enum (5 variants) — proper state machine
- `IterationOutcome` enum (4 variants) — proper dispatch
- `PreflightResult` enum — proper routing
- `AhaPriority` enum — proper signal handling

**Refactoring candidates:**
1. **Message dispatch** (agent_loop.rs:476-750) — 6-way if/else, should be `AgentEvent` enum
2. **LCM engine init** (agent_shared.rs:885-1109) — Complex nested DB/legacy/fresh fallback
3. **Tool source selection** (agent_shared.rs:773-872) — 3-way + nested 4-way trio stripping
4. **Proactive grounding** (agent_shared.rs:1181-1215) — 5 nested guard conditions
5. **Delegation health** (agent_shared.rs:1633-1653) — AtomicBool + counter retry logic
6. **Outcome handler** (agent_shared.rs:443-567) — Plan-guided logic nested in each branch

### agent_core.rs (46 combined)

1. **Memory provider selection** (441-491) — 10+ branches across local/cloud
2. **Delegation provider selection** (527-556) — 6+ branches with cascading fallback
3. **Token reserve calculation** (497-501) — 2 branches on is_local
4. **Compaction context size** (503-507) — 2 branches, config default fallback
5. **TrioState transition logging** (256-262) — Observer pattern candidate

**Key boolean flag:** `is_local` parameter cascades through ALL of the above

### tool_engine.rs (52 combined)

1. **Delegation health status** (176-216) — 4 states via AtomicBool + AtomicU64
2. **Tool result formatting** (248-276) — 3 strategies via nested if/else
3. **Provenance recording** (278-287) — 2-way config dispatch
4. **Summary injection** (297-323) — 3-branch nested optional aggregation

### cmd_read.rs (112 combined)

1. **Model dir resolution** (938-950, 1083-1097) — DUPLICATED cfg-gated blocks
2. **Mode/lane detection** (11-40) — 6 branches from boolean chain
3. **Prompt mode display** (341-494) — 150-line if/else (local vs cloud)
4. **Feedback quality** (739-746) — 3-branch string match
5. **Memory display** (566-601) — 3-level nesting
6. **Audit display** (646-681) — 2-level nesting
7. **Verify display** (687-734) — 3-level nesting
8. **Time formatting** (893-905) — 3-branch elapsed display

### cmd_lifecycle.rs (28 else)

1. **/local toggle** (1091-1253) — 10+ nested decisions, CRITICAL complexity
2. **/model source dispatch** (424-620) — 5 match arms, 60-line MLX arm
3. **/ctx apply logic** (146-290) — Platform-specific reload branching
4. **/trio subcommand** (651-693) — 13 slice-pattern arms (already good)

### cmd_mutation.rs (39 combined)

1. **/think toggle** (11-67) — if-match-match-if tree, string alias clutter
2. **/sessions dispatch** (162-214) — Inline arg parsing mixed with dispatch
3. **/replay mode** (216-324) — 4-way if/else display dispatch
4. **/long mode** (93-124) — if/match with special Ok(0) case

### lcm.rs (62 combined)

1. **Escalation Levels 1+2** (925-965) — Identical 3-branch logic DUPLICATED
2. **Compaction action** (430-436) — 3-way + `async_compaction_pending` bool flag
3. **Block parsing** (591-624) — Complex guard logic, 5+ branches
4. **Auto-expand** (759-805) — 4+ nested relevance/budget checks

### subagent.rs (19 else)

1. **Model resolution** (364-381) — 4-way fallback with warning side effects
2. **Cluster routing** (821-848) — 3 nested conditionals
3. **Provider resolution** (851-876) — 2-level fallback
4. **Tool invocation warning** (1112-1124) — 2-level content assembly

### server.rs (60 combined)

1. **Health check state** (847-893) — Multi-level state tracking with side effects
2. **GGUF field matching** (204-214) — 5-branch string key dispatch
3. **Context cap sizing** (270-280) — 5-tier numeric fallthrough

### cli/mod.rs (69 combined)

1. **Email field resolution** (1063-1129) — 4×3 = 12 branches (4 fields × 3 priority levels)
2. **Provider/channel status** (1209-1238, 1340-1389) — 5+4 repeated ternaries
3. **TTS fallback init** (843-876) — 5 nested conditions
4. **MLX provider init** (700-727) — Feature-gated 4-path dispatch
5. **Cron schedule type** (1444-1459) — 3-way mutually exclusive options
6. **Telegram token validation** (1006-1032) — Input/validate/persist nesting
