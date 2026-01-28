# TASK.md - Task Management Guidelines for Agentic Development

## Purpose
This document provides guidelines for creating and managing tasks in a fully agentic development environment using Claude Code. Tasks are stored as individual markdown files in the `tasks/` directory.

## Task Structure

### Directory Organization
```
/tasks/
├── TASK-001-implement-feature-x.md
├── TASK-002-fix-authentication-bug.md
├── TASK-003-refactor-database-layer.md
└── archive/
    └── TASK-000-completed-task.md
```

### Task File Naming Convention
`TASK-[ID]-[brief-description].md`
- ID: Sequential number (001, 002, etc.)
- Description: Lowercase, hyphen-separated summary

## Task File Template

Create each task as a markdown file in the `tasks/` directory with this structure:

```markdown
# TASK-[ID]: [Task Title]

## Status
- [ ] Not Started
- [ ] In Progress  
- [ ] Completed
- [ ] Blocked

**Priority**: High | Medium | Low  
**Created**: YYYY-MM-DD  
**Last Updated**: YYYY-MM-DD  

## Task Description
Clear, concise explanation of what needs to be accomplished. Focus on the "what" and "why".

## Requirements
1. Specific requirement or constraint
2. Technical specifications
3. Dependencies or prerequisites
4. Acceptance criteria

## Research & Context
- Background information gathered
- Related documentation or resources
- Previous attempts or similar implementations
- Technical decisions and rationale

## Implementation Notes
- Key files or modules affected
- Suggested approach or algorithm
- Potential challenges identified
- Testing considerations

## Progress Log
- YYYY-MM-DD: Initial task creation
- YYYY-MM-DD: Research completed, found X approach
- YYYY-MM-DD: Implementation started
```

## Workflow for Agentic Development

### 1. Task Creation
When Claude Code receives a new task:
```bash
# Claude creates a new task file
# Example: /tasks/TASK-004-add-user-authentication.md
```

### 2. Task Execution
Claude Code will:
- Read the task file for context
- Update the Research & Context section with findings
- Document progress in the Progress Log
- Update status checkboxes as work progresses

### 3. Task Completion
- Move completed tasks to `/tasks/archive/`
- Keep active and blocked tasks in `/tasks/`
- Maintain task history for future reference

## Best Practices for Agentic Development

### Do's
- Write tasks assuming a new AI agent might pick it up
- Include all necessary context in the task file
- Update the task file after significant progress
- Link to relevant code files using relative paths
- Keep research findings in the task for reuse

### Don'ts
- Don't assume context from previous conversations
- Don't create tasks without clear requirements
- Don't delete task files (archive instead)
- Don't use external references without explanation

## Task Discovery Commands

Agents can use these patterns to find and manage tasks:

```bash
# List all active tasks
ls tasks/TASK-*.md

# Find high priority tasks
grep -l "Priority: High" tasks/TASK-*.md

# Find tasks in progress
grep -l "\- \[x\] In Progress" tasks/TASK-*.md

# Search for specific topics
grep -r "authentication" tasks/
```

## Integration with Claude Code

When starting a session, Claude Code should:
1. Check the `tasks/` directory for active tasks
2. Read task files for context and requirements
3. Update task status and progress
4. Create new task files for incoming requests

This approach ensures continuity across sessions and enables effective collaboration in agentic development.