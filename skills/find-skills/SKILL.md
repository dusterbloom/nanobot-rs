---
name: find-skills
description: Helps users discover and install agent skills when they ask questions like "how do I do X", "find a skill for X", or express interest in extending capabilities. Activates when the user is looking for functionality that might exist as an installable skill.
---

# Find Skills

This skill helps you discover and install skills from the open agent skills ecosystem.

## When to Use This Skill

Use this skill when the user:

- Asks "how do I do X" where X might be a common task with an existing skill
- Says "find a skill for X" or "is there a skill for X"
- Asks "can you do X" where X is a specialized capability
- Wants to search for tools, templates, or workflows
- Mentions they wish they had help with a specific domain (design, testing, deployment, etc.)

## How to Install Skills

Nanobot can install skills directly from GitHub repositories:

**CLI:**
```bash
nanobot skills add owner/repo          # Install all skills from a repo
nanobot skills add owner/repo@skill    # Install a specific skill
```

**TUI (inside a nanobot session):**
```
/skill add owner/repo
/skill add owner/repo@skill
```

Skills are saved to `~/.nanobot/workspace/skills/{name}/SKILL.md` and become immediately available.

## How to Help Users Find Skills

### Step 1: Understand What They Need

Identify:
1. The domain (e.g., React, testing, design, deployment)
2. The specific task (e.g., writing tests, creating animations, reviewing PRs)
3. Whether this is a common enough task that a skill likely exists

### Step 2: Search for Skills

If `npx` is available, use the skills.sh search:

```bash
npx skills find [query]
```

Or browse skills at: https://skills.sh/

### Step 3: Present Options and Install

When you find relevant skills, present them with:
1. The skill name and what it does
2. The install command

Then install directly:
```bash
/skill add owner/repo@skill-name
```

## Common Skill Sources

| Source | Skills |
|--------|--------|
| `vercel-labs/agent-skills` | React, Next.js, web development best practices |
| `vercel-labs/skills` | find-skills, skill-creator |

## Common Skill Categories

| Category | Example Queries |
|----------|----------------|
| Web Development | react, nextjs, typescript, css, tailwind |
| Testing | testing, jest, playwright, e2e |
| DevOps | deploy, docker, kubernetes, ci-cd |
| Documentation | docs, readme, changelog, api-docs |
| Code Quality | review, lint, refactor, best-practices |
| Design | ui, ux, design-system, accessibility |
| Productivity | workflow, automation, git |

## When No Skills Are Found

If no relevant skills exist:
1. Offer to help with the task directly
2. Suggest the user could create their own skill: place a `SKILL.md` in `~/.nanobot/workspace/skills/{name}/`
