---
name: "git-operations"
displayName: "Git Operations"
description: "Automate git operations with MCP tools - commit analysis, branch management, and repository insights for development workflows."
keywords: ["git", "version-control", "commits", "branches", "repository"]
author: "Chimera Team"
---

# Git Operations

## Overview

This power provides automated git operations through MCP tools, enabling you to analyze commits, manage branches, search repository history, and generate insights without leaving your development environment.

Perfect for generating release notes, analyzing code changes, understanding repository history, and automating git workflows.

## Available MCP Servers

This power uses the **git** MCP server which provides the following tools:

- `git_status` - Get current repository status
- `git_diff_unstaged` - Show unstaged changes
- `git_diff_staged` - Show staged changes
- `git_commit` - Create a new commit
- `git_log` - View commit history
- `git_show` - Show commit details
- `git_create_branch` - Create a new branch
- `git_checkout` - Switch branches
- `git_list_branches` - List all branches

## Onboarding

### Prerequisites

- Git installed on your system (2.0+)
- A git repository initialized in your workspace
- Basic understanding of git concepts

### Installation

The git MCP server is installed automatically when you add this power. It uses the standard git command-line tool already on your system.

### Configuration

**No additional configuration required** - the MCP server works with any git repository in your workspace.

**Optional Environment Variables:**

- `GIT_AUTHOR_NAME` - Override git author name
- `GIT_AUTHOR_EMAIL` - Override git author email

## Common Workflows

### Workflow 1: Generate Release Notes

**Goal:** Automatically generate release notes from recent commits

**Steps:**

1. Get recent commits:

   ```
   Use git_log tool with parameters:
   - max_count: 20
   - format: "oneline"
   ```

2. Analyze commit messages and categorize:
   - Features (feat:)
   - Bug fixes (fix:)
   - Documentation (docs:)
   - Refactoring (refactor:)

3. Format into release notes with categories

**Example:**

```
Agent: "Generate release notes from the last 20 commits"

I'll use the git_log tool to fetch recent commits and categorize them:

[Tool calls git_log with max_count=20]

Release Notes v1.2.0:

Features:
- feat: add user authentication system
- feat: implement dashboard analytics

Bug Fixes:
- fix: resolve memory leak in cache
- fix: correct API endpoint validation

Documentation:
- docs: update API integration guide
```

### Workflow 2: Analyze Code Changes

**Goal:** Review and understand changes before committing

**Steps:**

1. Check repository status:

   ```
   Use git_status tool
   ```

2. Review unstaged changes:

   ```
   Use git_diff_unstaged tool
   ```

3. Review staged changes:

   ```
   Use git_diff_staged tool
   ```

4. Analyze the changes and provide insights

**Example:**

```
Agent: "What changes are in my working directory?"

[Tool calls git_status]
[Tool calls git_diff_unstaged]

You have 3 modified files:
- backend-api/app/api/endpoints/auth.py (23 lines changed)
- backend-api/tests/test_auth.py (45 lines added)
- README.md (5 lines changed)

Changes include:
- Added JWT token validation in auth endpoint
- Added comprehensive test coverage for auth
- Updated README with new auth documentation
```

### Workflow 3: Smart Commit Creation

**Goal:** Create well-formatted commits with AI assistance

**Steps:**

1. Review current changes with git_diff_staged
2. Analyze the changes to understand intent
3. Generate a conventional commit message
4. Create commit with git_commit tool

**Example:**

```
Agent: "Create a commit for my staged changes"

[Tool calls git_diff_staged]

Based on your changes, I'll create a commit:

Message: "feat: add JWT authentication to API endpoints

- Implement token validation middleware
- Add refresh token support
- Include comprehensive test coverage
- Update API documentation"

[Tool calls git_commit with the message]

✅ Commit created: abc123f
```

### Workflow 4: Branch Management

**Goal:** Create and manage feature branches

**Steps:**

1. List existing branches:

   ```
   Use git_list_branches tool
   ```

2. Create new feature branch:

   ```
   Use git_create_branch tool with branch name
   ```

3. Switch to the branch:

   ```
   Use git_checkout tool
   ```

**Example:**

```
Agent: "Create a new branch for the authentication feature"

[Tool calls git_list_branches]

Current branches:
- main (current)
- develop
- feature/dashboard

[Tool calls git_create_branch with name="feature/auth-system"]
[Tool calls git_checkout with branch="feature/auth-system"]

✅ Created and switched to branch: feature/auth-system
```

### Workflow 5: Commit History Analysis

**Goal:** Search and analyze repository history

**Steps:**

1. Search commits by author, date, or message:

   ```
   Use git_log tool with filters
   ```

2. View detailed commit information:

   ```
   Use git_show tool with commit hash
   ```

3. Analyze patterns and trends

**Example:**

```
Agent: "Show me all commits by John in the last month"

[Tool calls git_log with author="John", since="1 month ago"]

Found 15 commits by John:
- 10 feature additions
- 3 bug fixes
- 2 documentation updates

Most active areas:
- backend-api/app/services/ (8 commits)
- frontend/src/components/ (5 commits)
```

## Tool Usage Examples

### git_status

**Purpose:** Get current repository status

**Parameters:** None

**Example:**

```
Call: git_status()

Response:
On branch main
Your branch is up to date with 'origin/main'.

Changes not staged for commit:
  modified:   backend-api/app/main.py
  modified:   frontend/src/App.tsx

Untracked files:
  powers/git-operations/
```

### git_log

**Purpose:** View commit history

**Parameters:**

- `max_count` (optional): Number of commits to show
- `skip` (optional): Number of commits to skip
- `author` (optional): Filter by author
- `since` (optional): Show commits since date
- `until` (optional): Show commits until date
- `grep` (optional): Search commit messages
- `format` (optional): Output format (oneline, short, full)

**Example:**

```
Call: git_log(max_count=5, format="oneline")

Response:
abc123f feat: add authentication system
def456a fix: resolve memory leak
ghi789b docs: update API guide
jkl012c refactor: simplify error handling
mno345d feat: implement dashboard
```

### git_commit

**Purpose:** Create a new commit

**Parameters:**

- `message` (required): Commit message
- `all` (optional): Stage all changes before committing

**Example:**

```
Call: git_commit(
  message="feat: add user authentication\n\n- JWT token support\n- Refresh tokens\n- Test coverage",
  all=false
)

Response:
[main abc123f] feat: add user authentication
 3 files changed, 150 insertions(+), 10 deletions(-)
```

### git_create_branch

**Purpose:** Create a new branch

**Parameters:**

- `name` (required): Branch name
- `checkout` (optional): Switch to branch after creation

**Example:**

```
Call: git_create_branch(name="feature/new-dashboard", checkout=true)

Response:
Switched to a new branch 'feature/new-dashboard'
```

## Best Practices

### Commit Messages

- Use conventional commit format: `type(scope): description`
- Types: feat, fix, docs, refactor, test, chore
- Keep first line under 72 characters
- Add detailed body for complex changes

### Branch Naming

- Use descriptive names: `feature/auth-system`, `fix/memory-leak`
- Avoid generic names like `temp`, `test`, `wip`
- Use lowercase with hyphens

### Workflow Integration

- Check status before making changes
- Review diffs before committing
- Use meaningful commit messages
- Create feature branches for new work
- Keep commits atomic and focused

### Repository Hygiene

- Commit frequently with clear messages
- Don't commit sensitive data or secrets
- Keep branches up to date with main
- Delete merged branches
- Use .gitignore for build artifacts

## Troubleshooting

### Error: "Not a git repository"

**Cause:** Current directory is not a git repository

**Solution:**

1. Initialize git: `git init`
2. Or navigate to a git repository
3. Verify with: `git status`

### Error: "Nothing to commit"

**Cause:** No changes staged for commit

**Solution:**

1. Check status: Use git_status tool
2. Stage changes: `git add <files>`
3. Or use `all=true` parameter in git_commit

### Error: "Branch already exists"

**Cause:** Trying to create a branch that exists

**Solution:**

1. List branches: Use git_list_branches tool
2. Choose a different name
3. Or checkout existing branch: Use git_checkout tool

### Error: "Merge conflict"

**Cause:** Conflicting changes in branches

**Solution:**

1. View conflicts: Use git_status tool
2. Resolve conflicts manually in files
3. Stage resolved files: `git add <files>`
4. Complete merge: Use git_commit tool

### MCP Server Connection Issues

**Problem:** Git MCP server not responding

**Symptoms:**

- Tools not available
- Connection timeout errors

**Solutions:**

1. Verify git is installed: `git --version`
2. Check workspace has git repository
3. Restart Kiro
4. Check MCP server logs in Kiro

## Configuration

### Environment Variables

**Optional configuration in mcp.json:**

```json
{
  "mcpServers": {
    "git": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-git"],
      "env": {
        "GIT_AUTHOR_NAME": "Your Name",
        "GIT_AUTHOR_EMAIL": "your.email@example.com"
      }
    }
  }
}
```

**Default behavior:** Uses system git configuration from `~/.gitconfig`

## Advanced Usage

### Generating Changelogs

Combine git_log with AI analysis to generate formatted changelogs:

1. Fetch commits between versions
2. Categorize by type (feat, fix, docs)
3. Group by component or module
4. Format with markdown

### Code Review Assistance

Use git tools to assist with code reviews:

1. Analyze diff with git_diff_staged
2. Identify potential issues
3. Suggest improvements
4. Generate review comments

### Repository Insights

Generate insights about repository activity:

1. Analyze commit frequency
2. Identify most active files
3. Track contributor activity
4. Detect patterns and trends

---

**Package:** `@modelcontextprotocol/server-git`
**MCP Server:** git
**Repository:** <https://github.com/modelcontextprotocol/servers>
