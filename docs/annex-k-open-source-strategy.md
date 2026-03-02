# OctoFlow — Annex K: Open Source & Release Strategy

**Parent Document:** OctoFlow Strategic Vision
**Status:** Strategic Plan
**Version:** 0.3
**Date:** February 17, 2026

---

## The Reality

- Solo developer, no budget, no company
- 40+ phases of compiler engineering (777 tests)
- Want community growth NOW
- Want to protect the compiler binary from being renamed and redistributed
- Everything else should be genuinely open source

**The model is dead simple:**

```
BINARY (protected):    The compiled OctoFlow compiler
OPEN SOURCE (Apache 2.0): Everything else
```

That's it. No enterprise tiers. No SaaS pricing. No trademark budget. Just a protected binary and an open ecosystem.

---

## 1. What's Open, What's Protected

### Open Source (Apache 2.0) — Everything Except the Compiler Binary

```
OPEN SOURCE:
├── Parser source code (flowgpu-parser crate)
├── Standard library (.flow modules)
├── Pre-flight validation source
├── Lint source
├── Language specification (all docs/)
├── Examples and tutorials
├── Test suite
├── REPL source (when separated)
├── Build scripts, CI config
└── Module SDK
```

Anyone can read, modify, fork, redistribute, teach, sell courses about, build tools with — no restrictions beyond Apache 2.0 attribution.

### Protected Binary — The Compiler Only

```
PROTECTED (OctoFlow Binary License):
├── octoflow compiler binary (Linux x64)
├── octoflow compiler binary (macOS arm64)
├── octoflow compiler binary (macOS x64)
└── octoflow compiler binary (Windows x64)
```

The compiler is distributed as a **free pre-built binary** with a simple license:

```
OctoFlow Compiler Binary License v1.0

Copyright (c) 2026 Mike D. (OctoFlow)

Permission is granted, free of charge, to any person obtaining
a copy of this binary, to use it for any purpose — personal,
educational, or commercial — subject to these conditions:

1. You may NOT redistribute this binary under a different name.
2. You may NOT reverse-engineer, decompile, or extract source
   code from this binary.
3. You may NOT bundle this binary into another product and sell
   it as your own.
4. You MAY use this binary to compile and run any OctoFlow
   program for any purpose, including commercial use.
5. You MAY distribute programs and outputs created by this
   binary without restriction.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

For the open source components (parser, stdlib, docs), see the
Apache 2.0 license in the respective directories.
```

**What this means:**
- Free to use for anything
- Free to build commercial products WITH it
- Cannot rename the binary and redistribute it
- Cannot claim you built it
- Cannot extract the source and build a competing compiler

### Why Binary, Not Source-Available

Source-available is great in theory. In practice, for a solo dev:

- If the compiler source is public, anyone with `cargo build` can compile it and strip the name
- A license violation is hard to enforce when you're one person with no legal budget
- Binary distribution is simpler: the source stays private, the binary is free

**The compiler source stays in a private repo. The binary goes on GitHub Releases.**

---

## 2. Protection Without Money

### 2.1 Common Law Trademark (Free)

You don't need to register a trademark to have one. In most countries, **first use in commerce establishes common law trademark rights.**

**What to do (costs $0):**
- Use "OctoFlow™" (with the ™ symbol) in all public materials
- The ™ symbol means "I claim this as my trademark" — no registration needed
- Document your first public use (date, URL, screenshot)
- If someone later tries to use "OctoFlow" for a programming language, your first-use date proves priority

**When you have $250-400 later:**
- File for registered trademark (®) with your country's IP office
- Registration makes enforcement easier but is NOT required for protection

### 2.2 First-Mover Advantage (Free)

The best protection is being the canonical source:
- **Your** GitHub repo is where stars accumulate
- **Your** releases are what people install
- **Your** docs are what tutorials reference
- **Your** name is on every commit

A clone has to fight uphill against an established community. By the time someone tries to copy, if you have 100+ stars and active issues, the clone is irrelevant.

### 2.3 The Binary License (Free)

The license text above is all you need. It's simple enough that anyone can understand it. It clearly prohibits the one thing you're worried about: rename-and-redistribute.

### 2.4 Git History (Free)

Your git history is timestamped proof that you built this. Every commit from Phase 0 to Phase 40 proves authorship. If anyone claims they built it, 40 phases of git history says otherwise.

---

## 3. Repository Structure

### Public Repo: `github.com/octoflow/octoflow`

```
octoflow/
├── LICENSE-APACHE          # Apache 2.0 for open components
├── LICENSE-BINARY           # OctoFlow Binary License for compiler
├── README.md
├── CONTRIBUTING.md
├── arms/
│   ├── octoflow-parser/    # OPEN SOURCE (Apache 2.0)
│   │   └── src/
│   ├── [preflight source]  # OPEN SOURCE (Apache 2.0) — when split out
│   └── [lint source]       # OPEN SOURCE (Apache 2.0) — when split out
├── stdlib/                 # OPEN SOURCE (Apache 2.0)
├── examples/               # OPEN SOURCE (Apache 2.0)
├── docs/                   # OPEN SOURCE (Apache 2.0)
├── tests/                  # OPEN SOURCE (Apache 2.0)
└── releases/               # BINARY downloads (GitHub Releases)
    ├── octoflow-linux-x64
    ├── octoflow-macos-arm64
    └── octoflow-windows-x64.exe
```

### Private Repo: (your local machine / private GitHub)

```
flowgpu-private/
├── arms/
│   ├── flowgpu-spirv/      # SPIR-V codegen (private)
│   ├── flowgpu-vulkan/     # Vulkan runtime (private)
│   └── flowgpu-cli/        # Compiler core (private)
│       └── src/
│           ├── compiler.rs  # The engine (private)
│           ├── lib.rs       # Overrides, types (private)
│           └── main.rs      # CLI entry point (private)
└── build scripts to produce release binaries
```

### How Releases Work

1. You build the compiler locally (private repo)
2. Run all 777+ tests
3. Cross-compile binaries for Linux, macOS, Windows
4. Upload binaries to GitHub Releases on the public repo
5. Users download the binary + clone the open source parts

```bash
# User installation (future)
# Option A: Download binary from GitHub Releases
curl -L https://github.com/octoflow/octoflow/releases/latest/download/octoflow-linux-x64 -o octoflow
chmod +x octoflow

# Option B: Install script
curl -fsSL https://octoflow.dev/install.sh | sh

# The parser/stdlib/docs are in the repo
git clone https://github.com/octoflow/octoflow
```

---

## 4. Community From Day One

### What to Release First

Don't wait for perfection. The language has 777 tests and handles real use cases. That's more than most languages have at launch.

**Soft Launch (after Phase 42, ~807 tests):**
- Make the public repo visible
- Upload first binaries to GitHub Releases
- Write a clear README explaining what OctoFlow is
- Post on personal social media
- No big announcement — just accessible

**Community Launch (after Phase 44, ~837 tests):**
- Post on Hacker News: "Show HN: OctoFlow — GPU-native language where LLMs write the code"
- Post on Reddit: r/programming, r/rust, r/ProgrammingLanguages
- Open GitHub Discussions for questions
- Create Discord server when there are 10+ people interested

### Contributor Experience

```markdown
# CONTRIBUTING.md

OctoFlow welcomes contributions to the open source components:
parser, stdlib, docs, examples, and tests.

## How to Contribute

1. Fork the repo
2. Make your changes to any Apache 2.0 licensed component
3. Run tests (instructions in README)
4. Submit a Pull Request
5. I'll review and merge

## What You Can Contribute

- Bug reports (GitHub Issues)
- Parser improvements
- New stdlib modules (.flow files)
- Documentation improvements
- Example programs
- Test cases
- IDE extensions (using the open parser)

## The Compiler

The compiler binary is distributed under the OctoFlow Binary License.
The compiler source is not in this repo. If you find a compiler bug,
please file an issue and I'll fix it.
```

### Community Spaces (Start Small)

```
GitHub Issues       — bugs, feature requests
GitHub Discussions   — questions, ideas, showcases
Discord              — when 10+ interested people (not before)
```

Don't create infrastructure nobody uses. Scale up when there's demand.

---

## 5. Release Timeline

### Now → Phase 42: Prepare

- [x] OctoFlow branding on README
- [x] Strategic vision document
- [x] Domain audit and foundation map
- [ ] Write LICENSE-BINARY text
- [ ] Write CONTRIBUTING.md
- [ ] Complete Phase 41 (stats, paths, encoding)
- [ ] Complete Phase 42 (date/time)
- [ ] Build first cross-platform binaries
- [ ] Decide on public repo structure (what moves, what stays private)

### Phase 42 → Phase 44: Soft Launch

- [ ] Public repo goes live (parser source + compiler binary + docs + examples)
- [ ] First GitHub Release with binaries
- [ ] Mention on social media
- [ ] Accept first community contributions
- [ ] Use "OctoFlow™" consistently (common law trademark)

### Phase 44+: Community Launch

- [ ] Hacker News / Reddit post
- [ ] Discord server
- [ ] First community-written modules
- [ ] Blog posts explaining architecture
- [ ] "OctoFlow in 5 Minutes" demo

---

## 6. Future Revenue (Only When It Makes Sense)

Don't build revenue infrastructure for 0 users. But when the time comes:

### When You Have Users (100+)

- **GitHub Sponsors** — people support projects they use
- **Open Collective** — transparent project funding
- **Donations** — simple, no strings attached

### When You Have Real Demand (1000+)

- **OctoFlow Cloud** — hosted GPU execution (people without GPUs pay to run .flow programs on your GPU cluster)
- **Sponsored features** — companies pay to prioritize features they need
- **Workshops/training** — paid sessions for teams adopting OctoFlow

### When You Have Scale (10,000+)

- **Premium binary** — advanced profiler, optimization reports
- **Hosted registry** — premium module hosting
- **Consulting** — when you can afford to spend time on it

**The rule:** Build community first. Revenue follows genuine adoption.

---

## 7. Realistic Risks

| Threat | Likelihood | What To Do |
|--------|-----------|------------|
| Nobody cares | High | Build things people actually need. Show real demos. |
| Someone redistributes binary | Low | License prohibits it. DMCA takedown if needed ($0). |
| Someone reverse-engineers compiler | Very Low | SPIR-V codegen is complex. By the time they rebuild it, you're 20 phases ahead. |
| Big company builds competing GPU language | Low | Validates the space. Your community stays if you're authentic. |
| Burnout | Medium | Pace yourself. Accept contributions. You're not a company. |

**The biggest risk is obscurity, not theft.** Focus on being seen, not on being protected.

---

## Summary

```
COMPILER:     Free binary, protective license (can't rename/redistribute)
EVERYTHING ELSE: Open source, Apache 2.0

TRADEMARK:    Use "OctoFlow™" everywhere (free common law rights)
              Register ® later when you have $250-400

COMMUNITY:    Public repo after Phase 42
              Hacker News launch after Phase 44

REVENUE:      GitHub Sponsors / donations first
              Cloud/training later when demand exists

PROTECTION:   Binary license + first-mover + git history + ™
```

No enterprise tiers. No SaaS pricing. No VC deck. Just a developer sharing their work with the world while keeping the engine protected.

---

*"Share everything except the engine. Protect the name. Build the community. The rest will follow."*
