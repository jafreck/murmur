---
name: publish
description: >
  Walk through the murmur release and publish flow: bump version, open a PR,
  merge, push a tag, and let the publish workflow build artifacts, create the
  GitHub Release, and push to crates.io.
---

# Publish / Release Flow

Use this guide whenever the user asks to publish, release, or cut a new version
of murmur.

## Overview

```
bump version → PR to main → merge → push vX.Y.Z tag → publish.yml
```

The `publish.yml` workflow triggers on tag push (`v*`). It runs CI, builds
cross-platform binaries, creates the GitHub Release with all artifacts
attached, and publishes to crates.io. The release is not visible until
everything is built, so install scripts always find the binaries they expect.

## Step 1 — Decide the new version

If the user did not specify a target version or bump type, **stop and ask**
before proceeding. Read the current version from `Cargo.toml`, then prompt:

> The current version is **X.Y.Z**. What kind of release is this?
>
> - **patch** (X.Y.Z+1) — bug fixes, small improvements
> - **minor** (X.Y+1.0) — new features, backward-compatible changes
> - **major** (X+1.0.0) — breaking changes

Wait for the user's answer before continuing to Step 2.

## Step 2 — Bump the version

1. Fetch the latest `origin/main` and create a branch from it:
   ```sh
   git fetch origin main
   git checkout -b release/vX.Y.Z origin/main
   ```
2. Update `version` in **Cargo.toml**.
3. Run `cargo check` to regenerate `Cargo.lock`.
4. Commit:
   ```sh
   git add Cargo.toml Cargo.lock
   git commit -m "chore: bump version to X.Y.Z"
   ```

## Step 3 — Open a PR

Push the branch and open a PR targeting `main`:

```sh
git push origin release/vX.Y.Z
gh pr create --title "chore: bump version to X.Y.Z" \
  --body "Version bump for the vX.Y.Z release." \
  --base main
```

Wait for CI to pass (the `ci.yml` workflow runs on PRs to `main`).

## Step 4 — Merge the PR

Once CI is green, merge:

```sh
gh pr merge --squash --delete-branch
```

## Step 5 — Push the tag

Tag the merge commit on `main` and push the tag. This triggers the
`publish.yml` workflow:

```sh
git fetch origin main
git tag vX.Y.Z origin/main
git push origin vX.Y.Z
```

**Do NOT create a GitHub Release manually.** The workflow creates it
automatically after all binaries are built, so the release is never visible
without its artifacts.

## What `publish.yml` does automatically

1. **CI** — runs build, test, fmt, clippy on Linux/macOS/Windows.
2. **Build** — compiles release binaries for four targets:
   - `aarch64-apple-darwin` (macOS ARM, with Metal)
   - `x86_64-apple-darwin` (macOS Intel)
   - `x86_64-unknown-linux-gnu` (Linux)
   - `x86_64-pc-windows-msvc` (Windows)
3. **Create GitHub Release** — creates the release and attaches all binaries
   (`.tar.gz` / `.zip` / `.exe`) and install scripts (`install.sh`,
   `install.ps1`, `uninstall.sh`, `uninstall.ps1`) in a single step.
4. **Publish crate** — runs `cargo publish` to crates.io using the
   `CARGO_REGISTRY_TOKEN` secret.

## Checklist

- [ ] Version bumped in `Cargo.toml`
- [ ] `Cargo.lock` updated (`cargo check`)
- [ ] PR opened, CI green, merged to `main`
- [ ] Tag `vX.Y.Z` pushed to `origin/main`
- [ ] `publish.yml` workflow completed successfully
- [ ] Release assets and crates.io page verified
