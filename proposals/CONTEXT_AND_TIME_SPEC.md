# CONTEXT.md and Time Awareness Specification

## Overview

This document specifies the architecture for efficient context management and natural time awareness in nanobot.

**Core Principle:** Keep the system prompt lean (~1.5-2k tokens). Retrieve everything else on-demand.

---

## 1. Lean System Prompt Architecture

### The Problem

Current system prompt includes everything:
- SOUL.md (~400 tokens)
- USER.md (~450 tokens)
- TOOLS.md + schemas (~2,600 tokens)
- MEMORY.md (~1,500 tokens)
- Observations (~1,000 tokens)
- **Total: ~6,000-7,000 tokens before conversation even starts**

This makes small local models (3B) unusable and slows down all models.

### The Solution

**In System Prompt (1.5-2k tokens):**
```
├── SOUL.md (condensed identity, ~200 tokens)
├── USER.md (condensed preferences, ~200 tokens)
└── CONTEXT.md (situational awareness, ~500 tokens)
```

**Retrieved On-Demand:**
```
├── Tools → list names only, load schema when agent decides to use
├── Memory → semantic search pulls relevant bits
└── Observations → search when relevant, don't dump
```

### Benefits

| Model Type | Before | After |
|------------|--------|-------|
| SLM (3B) | Unusable (15k prompt) | Works (2k prompt) |
| LLM (Claude) | Slow first response | Fast |
| Context room | ~25% for conversation | ~85% for conversation |

---

## 2. CONTEXT.md — Situational Awareness (500 tokens)

### Purpose

Bridge between compaction events. Survives context resets. Provides situational awareness without dumping full history.

### Location

```
~/.nanobot/workspace/CONTEXT.md
```

### Structure

```markdown
# Context

## Now
- Focus: [current task/topic]
- Task: [active task from taskboard, if any]
- Blocked: [anything waiting on user/external]

## Time
- Last session: [when, duration, gap since]
- Session started: [timestamp]
- Pattern: [quick check-in / deep work / debugging]

## Recent
- [Decision or outcome from last session]
- [Decision or outcome from session before]
- [Max 3-5 items]

## External
- [Git changes since last session]
- [Files modified outside nanobot]
```

### Example (Under 500 tokens)

```markdown
# Context

## Now
- Focus: Testing NanBeige 3B local model performance
- Task: Optimize context usage for SLMs
- Blocked: None

## Time
- Last session: 2 hours ago (45 min, testing TUI)
- Session started: 2026-02-12 13:49
- Pattern: Deep work session

## Recent
- TUI updated with You/Nano labels and separators
- Flash attention confirmed enabled in local_llm.rs
- 15k token prompt identified as slowness cause

## External
- No git changes since last session
```

---

## 3. On-Demand Retrieval

### Tools

**Before:** All 14 tool schemas in prompt (~2,000 tokens)

**After:**
```
System prompt says: "Available tools: exec, read_file, write_file, 
edit_file, list_dir, web_search, web_fetch, task_board, scratchpad, 
skill_manager, spawn, cron, message"

Agent decides to use task_board → schema loaded for that call only
```

### Memory

**Before:** Full MEMORY.md dump (~1,500 tokens)

**After:**
```
User asks about "LoRA training"
    ↓
Semantic search on MEMORY.md + observations
    ↓
Returns relevant snippets (~200 tokens)
    ↓
Injected as "Relevant Memory:" in that turn only
```

### Observations

**Before:** Last N observations in prompt (~1,000 tokens)

**After:**
```
Only pulled when relevant to current query
Semantic search matches topic
Injected as "Related Context:" when needed
```

---

## 4. Observer/Reflector Integration

### Current System

```
Session ends 
    → Observer saves observation 
    → Reflector crystallizes to MEMORY.md
```

### Extended System

```
Context threshold (80%) OR Session ends
    ↓
Observer captures:
    ├── Observation (what happened) → observations/
    └── CONTEXT.md update (situational state)
    ↓
Reflector (background):
    ├── Crystallizes observations → MEMORY.md
    └── Validates CONTEXT.md freshness
```

### Triggers

| Event | Action |
|-------|--------|
| Context ≥ 80% | Observer updates CONTEXT.md before compaction |
| Session end (graceful) | Observer updates CONTEXT.md |
| User exits (`/quit`, close) | Observer updates CONTEXT.md |
| Reflector cycle | Validates CONTEXT.md not stale |

### Observer Changes

```rust
// observer.rs - extended

pub struct Observer {
    // ... existing fields
}

impl Observer {
    // Existing: called at session end
    pub async fn capture_observation(&self, session: &Session) -> Result<()> {
        // Save observation as before
        self.save_observation(session).await?;
        
        // NEW: Also update CONTEXT.md
        self.update_context_md(session).await?;
        
        Ok(())
    }
    
    // NEW: called when context hits threshold
    pub async fn on_context_threshold(&self, session: &Session) -> Result<()> {
        self.update_context_md(session).await
    }
    
    async fn update_context_md(&self, session: &Session) -> Result<()> {
        let context = ContextMd {
            focus: session.current_focus(),
            task: session.active_task(),
            blocked: session.blocked_items(),
            last_session: Utc::now(),
            recent_decisions: session.decisions().take(5),
            external_changes: self.detect_git_changes()?,
        };
        
        context.write_to(&self.workspace.join("CONTEXT.md"))
    }
}
```

### Reflector Changes

```rust
// reflector.rs - extended

impl Reflector {
    pub async fn run_cycle(&self) -> Result<()> {
        // Existing: crystallize observations to MEMORY.md
        self.crystallize_observations().await?;
        
        // NEW: validate CONTEXT.md freshness
        self.validate_context_md().await?;
        
        Ok(())
    }
    
    async fn validate_context_md(&self) -> Result<()> {
        let context_path = self.workspace.join("CONTEXT.md");
        
        if let Ok(metadata) = fs::metadata(&context_path) {
            let age = Utc::now() - metadata.modified()?;
            
            // If CONTEXT.md is stale (>24h), refresh from observations
            if age > Duration::hours(24) {
                self.refresh_context_from_observations().await?;
            }
        }
        
        Ok(())
    }
}
```

---

## 5. Time Awareness

### Temporal Context

Injected into system prompt (part of CONTEXT.md):

```
## Time
- Current: Thursday, Feb 12, 2026, 1:49 PM
- Last interaction: 3 hours ago
- Session: Just started
- Pattern: Deep work (based on recent session lengths)
```

### Behavioral Hints

| Gap | Suggested Behavior |
|-----|---------------------|
| < 5 min | Continue naturally |
| 5-30 min | Brief "back to it" |
| 30min - 2h | Re-orient, check CONTEXT.md |
| 2h+ | Summarize where we left off |
| Next day | Fresh start, review yesterday |

### Implementation

```rust
fn build_time_context(last_interaction: Option<DateTime<Utc>>) -> String {
    let now = Utc::now();
    let gap = last_interaction.map(|t| now - t);
    
    let gap_str = match gap {
        None => "First session".into(),
        Some(d) if d < Duration::minutes(5) => "Just now".into(),
        Some(d) if d < Duration::hours(1) => format!("{} min ago", d.num_minutes()),
        Some(d) if d < Duration::hours(24) => format!("{} hours ago", d.num_hours()),
        Some(d) => format!("{} days ago", d.num_days()),
    };
    
    format!("- Last interaction: {}", gap_str)
}
```

---

## 6. External Change Detection

On session start, detect changes made outside nanobot:

```rust
fn detect_external_changes(workspace: &Path) -> Vec<String> {
    let mut changes = vec![];
    
    // Git status
    if let Ok(output) = Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(workspace)
        .output() 
    {
        let modified = String::from_utf8_lossy(&output.stdout);
        if !modified.is_empty() {
            changes.push(format!("Modified files: {}", modified.lines().count()));
        }
    }
    
    // Recent commits
    if let Ok(output) = Command::new("git")
        .args(["log", "--oneline", "-5", "--since=12 hours ago"])
        .current_dir(workspace)
        .output()
    {
        let commits = String::from_utf8_lossy(&output.stdout);
        if !commits.is_empty() {
            changes.push(format!("Recent commits: {}", commits.lines().count()));
        }
    }
    
    changes
}
```

---

## 7. Implementation Priority

### Phase 1: Quick Wins
- [ ] Time awareness in system prompt
- [ ] CONTEXT.md template and manual updates

### Phase 2: Core Architecture
- [ ] Lean system prompt builder
- [ ] On-demand tool schema loading
- [ ] Semantic memory retrieval integration

### Phase 3: Automation
- [ ] Observer updates CONTEXT.md on threshold/exit
- [ ] Reflector validates CONTEXT.md freshness
- [ ] External change detection on session start

---

## 8. Migration Path

1. Create CONTEXT.md template
2. Add time awareness to system prompt
3. Gradually move MEMORY.md out of prompt → semantic retrieval
4. Gradually move tool schemas → on-demand loading
5. Wire up Observer/Reflector integration

No breaking changes. Each step improves performance independently.
