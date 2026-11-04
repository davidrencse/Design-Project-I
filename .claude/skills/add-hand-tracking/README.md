# Hand Tracking Cursor — Claude Code Skill

A Claude Code skill that adds **hand-gesture cursor control** to any existing website using MediaPipe Hand Landmarker.

Point your index finger to move the cursor. Make a fist to click. Move your hand to the edges to scroll.

## What it does

When you run `/add-hand-tracking` in Claude Code, it will:

1. **Detect your framework** (React, Vue, Svelte, Next.js, vanilla HTML, etc.)
2. **Install** `@mediapipe/tasks-vision`
3. **Add** all the tracking modules, adapted to your project's idioms
4. **Inject** an intro overlay ("Mouse or Hand Tracking?" choice screen)
5. **Wire up** cursor UI, fist-click, and edge-zone scrolling
6. **Verify** everything works

## Features

- **Index finger cursor** — your finger tip controls the cursor position
- **Fist = click** — make a fist to click on whatever's under the cursor
- **Edge-zone scrolling** — move hand to top/bottom edges to scroll
- **High sensitivity** — small palm movements cover the full screen (no arm waving)
- **Buttery smooth** — One Euro Filter kills jitter while staying responsive
- **Mouse fallback** — always available, auto-activates if webcam fails
- **Privacy first** — all processing runs locally in the browser, zero network requests
- **Framework agnostic** — works with React, Vue, Svelte, Next.js, Astro, vanilla HTML

## Installation

Copy the skill into your Claude Code skills directory:

```bash
# Global (available in all projects)
mkdir -p ~/.claude/skills
cp -r add-hand-tracking ~/.claude/skills/

# Or clone directly
git clone https://github.com/sam-vish/hand-tracking-claude-skill.git ~/.claude/skills/add-hand-tracking
```

Then **restart Claude Code** (close and reopen the chat) for the skill to be discovered.

## Usage

Open any website project in Claude Code, then:

```
/add-hand-tracking
```

Or just say: *"add hand tracking to this website"* — Claude will auto-detect the skill.

## File Structure

```
add-hand-tracking/
├── SKILL.md                          # Master instructions (9-step playbook)
└── references/
    ├── one-euro-filter.js            # Smoothing filter with tuning guide
    ├── hand-tracker.js               # MediaPipe wrapper, fist detection, sensitivity
    ├── cursor-pipeline.js            # State manager, fist→click dispatch, mouse fallback
    ├── edge-scroller.js              # Edge-zone scroll with zone diagram
    ├── cursor-ui.html                # Cursor dot/ring with fist+scroll visual states
    └── intro-overlay.html            # "Mouse or Hand Tracking?" choice screen
```

## Tech

- **MediaPipe Hand Landmarker** — 21 hand landmarks at 30fps via GPU
- **One Euro Filter** — adaptive low-pass filter for signal smoothing
- **Fist detection** — all 4 fingertips below MCP joints = closed fist
- **Sensitivity remapping** — camera range 0.30–0.70 → full screen 0–1

## License

MIT
