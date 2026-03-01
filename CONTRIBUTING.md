# Contributing to DC-COX

Thank you for your interest in contributing to DC-COX! This document provides guidelines for contributing to this project.

## Development Setup

See the [Development section in README.md](README.md#development) for setup instructions.

## Pull Request Process

### 1. PR Title

Your PR title **will appear in release notes**. Make it clear and descriptive:

- ✅ Good: `Add survival curve visualization to worker UI`
- ✅ Good: `Fix convergence error when bootstrap sample is too small`
- ❌ Bad: `Fix bug`
- ❌ Bad: `Update code`

### 2. Labels (Required)

Every PR **must have at least one label** from the categories below. Labels determine how your PR is categorized in release notes.

| Label | When to Use |
|-------|-------------|
| `feature`, `enhancement` | New functionality or improvements |
| `bug`, `fix` | Bug fixes |
| `perf`, `performance` | Performance improvements |
| `refactor`, `chore`, `ci`, `build` | Code refactoring, maintenance, CI/CD changes |
| `documentation`, `docs` | Documentation-only changes |
| `test`, `tests` | Test additions or modifications |
| `security` | Security fixes or improvements |
| `breaking-change` | Changes that break backward compatibility |
| `skip-changelog` | Exclude from release notes (typos, minor fixes) |

> **Note**: Some labels are auto-applied based on changed files (e.g., `documentation` for `*.md` files, `ci` for `.github/**`).

### 3. Breaking Changes

If your PR introduces a breaking change:

1. Add the `breaking-change` label
2. Prefix your PR title with `[BREAKING]` (optional but recommended)
3. Fill out the "Breaking Change Migration Notes" section in the PR template
4. Document what users need to change in their code

Example:
```
[BREAKING] Rename `Projector.project()` to `Projector.transform()`

Migration: Replace all calls to `.project(X, Xanc, events, durations)` 
with `.transform(X, Xanc, events, durations)`.
```

### 4. When to Use `skip-changelog`

Use `skip-changelog` for changes that don't affect users:
- Typo fixes in comments
- Internal refactoring with no API changes
- CI configuration tweaks
- Dependency updates (handled by Dependabot, auto-excluded)

## Release Process

This project uses **GitHub's automatically generated release notes**. There is no `CHANGELOG.md` file to maintain.

### Creating a Release

1. **Ensure `.github/release.yml` exists on main** (it must be in the commit the tag points to)

2. **Go to GitHub → Releases → Draft a new release**

3. **Create or select a tag** (e.g., `v0.1.0`)

4. **Click "Generate release notes"** — GitHub will:
   - Find all PRs merged since the last tag
   - Group them by label into categories
   - Exclude PRs from bots (dependabot, github-actions)
   - Exclude PRs with `skip-changelog` label

5. **Review and edit** the generated notes if needed

6. **Publish the release**

### Troubleshooting Release Notes

If a PR doesn't appear in the expected category:

| Problem | Cause | Solution |
|---------|-------|----------|
| PR missing entirely | PR has `skip-changelog` label | Remove label and regenerate |
| PR in "Other Changes" | PR has no recognized label | Add appropriate label to PR |
| PR in wrong category | Wrong label applied | Fix label on PR |
| Bot PRs appearing | Bot not in exclude list | Add bot to `.github/release.yml` |
| No PRs showing | PRs not merged (only closed) | PRs must be merged, not just closed |

## Code Style

- Follow existing code style (enforced by Ruff)
- Run `uv run ruff check . --fix && uv run ruff format .` before committing
- Type hints are encouraged (checked by mypy)

## Questions?

Open an issue or start a discussion on GitHub.
