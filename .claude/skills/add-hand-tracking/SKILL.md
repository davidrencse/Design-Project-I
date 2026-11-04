---
name: add-hand-tracking
description: >
  Add hand-gesture cursor control to any existing website. Replaces or augments
  the mouse cursor with MediaPipe Hand Landmarker tracking via webcam.
  Index finger controls cursor, fist gesture clicks, edge-zone scrolling.
  Works with any framework (React, Vue, Svelte, Next.js, vanilla HTML, etc.).
  TRIGGER: user wants hand tracking, gesture control, webcam cursor, or
  touchless navigation on their website.
user-invocable: true
argument-hint: "[path-to-project]"
effort: high
---

# Add Hand-Tracking Cursor to Any Website

You are adding gesture-based cursor control to an existing website using
**Google MediaPipe Hand Landmarker**. The user's index finger controls the
cursor, making a fist triggers a click, and moving the hand to screen edges
scrolls the page.

## Step 1: Analyze the Target Project

Before writing any code, understand what you're working with:

1. **Detect the framework**: Read `package.json`, config files, and entry points.
   Common stacks: React/Next.js, Vue/Nuxt, Svelte/SvelteKit, Astro, vanilla HTML.
2. **Find the root layout/shell**: The file that wraps all pages (e.g., `_app.tsx`,
   `+layout.svelte`, `App.vue`, `index.html`).
3. **Check for existing cursor styles**: Search for `cursor:` in CSS files.
   You'll need to set `cursor: none` globally when hand tracking is active.
4. **Check for COEP/COOP headers**: MediaPipe needs these for SharedArrayBuffer.
   Look in server config, `next.config.js`, `vite.config.*`, `nuxt.config.*`, etc.

## Step 2: Install the Dependency

```bash
npm install @mediapipe/tasks-vision
```

If using a bundler that has issues with WASM (common with Vite), add to the
bundler config:

```js
// Vite: vite.config.ts
optimizeDeps: { exclude: ['@mediapipe/tasks-vision'] }
```

## Step 3: Add Required Headers

MediaPipe's WASM runtime performs best with these headers. Add them to the
dev server config:

```
Cross-Origin-Embedder-Policy: credentialless
Cross-Origin-Opener-Policy: same-origin
```

Framework-specific locations:
- **Vite/SvelteKit**: `vite.config.ts` → `server.headers`
- **Next.js**: `next.config.js` → `headers()` or middleware
- **Nuxt**: `nuxt.config.ts` → `nitro.routeRules` or `devServer`
- **Vanilla**: meta tags or server config

## Step 4: Create the Hand Tracking Module

Create these files in the project. Adapt the file extension and module syntax
to match the project's conventions (`.ts`, `.js`, `.tsx`, etc.).

### 4a. One Euro Filter (smoothing)

Read the reference implementation:
**File: `${CLAUDE_SKILL_DIR}/references/one-euro-filter.js`**

This is a signal processing filter that makes cursor movement buttery smooth.
Copy this into the project as a utility module. Key parameters:
- `minCutoff = 0.4` — heavy smoothing when hand is still (kills jitter)
- `beta = 0.002` — stays smooth even during movement

### 4b. Hand Tracker (MediaPipe wrapper)

Read the reference implementation:
**File: `${CLAUDE_SKILL_DIR}/references/hand-tracker.js`**

This wraps MediaPipe Hand Landmarker. Key design decisions:
- **Index finger tip (landmark 8)** controls the cursor position
- **Fist detection**: all 4 fingertips below their MCP joints = fist
- **Sensitivity**: raw coords 0.30–0.70 remapped to 0–1 (small palm movements
  cover the full screen — no arm waving)
- **Mirror flip**: webcam x is inverted so moving hand right → cursor right
- Runs at **30fps** detection rate

### 4c. Cursor Pipeline (state management)

Read the reference implementation:
**File: `${CLAUDE_SKILL_DIR}/references/cursor-pipeline.js`**

This is the central state manager. Adapt it to the project's state management:
- **React**: use `useState`/`useRef` in a context provider or zustand store
- **Vue**: use `reactive()`/`ref()` in a composable
- **Svelte**: use `$state` runes in a `.svelte.ts` module
- **Vanilla**: use a simple event emitter or global object

Key behaviors:
- OneEuroFilter smoothing on every frame
- Fist-to-click: on open→closed transition, dispatch a real `MouseEvent('click')`
  at the cursor's screen position via `document.elementFromPoint()`
- 500ms cooldown between fist clicks
- Mouse fallback always available

### 4d. Edge-Zone Scroller

Read the reference implementation:
**File: `${CLAUDE_SKILL_DIR}/references/edge-scroller.js`**

Scrolls the page based on vertical cursor position:
- **Top zone (0–35%)**: scroll up, speed ramps quadratically
- **Dead zone (35–65%)**: no scrolling
- **Bottom zone (65–100%)**: scroll down, speed ramps quadratically
- `MAX_SCROLL_SPEED = 45` px/frame

The scroller needs a reference to the scrollable container element. In most
sites this is `document.documentElement` or a specific wrapper div.

## Step 5: Create the Cursor UI Component

Read the reference implementation:
**File: `${CLAUDE_SKILL_DIR}/references/cursor-ui.html`**

The cursor UI is a fixed-position overlay with:
- **White dot** (10px) with glow — follows hand position
- **Ring** (28px) around the dot — subtle border
- **Fist feedback**: dot shrinks to 60%, turns blue, ring contracts
- **Scroll feedback**: dot scales up 130%, ring expands and glows blue
- **Scroll arrows**: appear at top/bottom edges when scrolling

Position: `left: {x * 100}vw; top: {y * 100}vh` with `transform: translate(-50%, -50%)`

**IMPORTANT**: Set `pointer-events: none` on the cursor so it doesn't block clicks.

## Step 6: Create the Intro/Choice Overlay

Read the reference implementation:
**File: `${CLAUDE_SKILL_DIR}/references/intro-overlay.html`**

Before the website loads, show a fullscreen overlay asking the user to choose:
- **"ENTER WITH MOUSE"** (primary button) — enables mouse fallback, skips tracking
- **"USE HAND TRACKING"** (secondary button) — initializes MediaPipe, requests
  webcam permission, starts tracking

Design:
- Dark semi-transparent backdrop (`rgba(0,0,0,0.92)`)
- Centered card with hand icon SVG, title, tagline, description
- Description text: "Your cursor follows your index finger. Move your hand to the
  edges to scroll. Make a fist to click."
- Small note: "Hand tracking is optional. Mouse works as a fallback."
- Loading state on the hand tracking button while MediaPipe initializes

The overlay should be rendered as the **first thing** in the app, before any
other content. When a choice is made, remove the overlay and show the site.

## Step 7: Wire It All Together

In the root layout/shell component:

1. **On mount**: enable mouse fallback by default, set up the scroll container
2. **Start rAF loop**: on every frame, call `updateScroll(gazeY)` to drive scrolling
3. **"Enter with mouse"**: just dismiss the overlay
4. **"Use hand tracking"**: call `initGaze()`, on success dismiss overlay
5. **Show toggle button**: in the bottom-left corner during experience, let user
   switch between mouse and hand tracking at any time

## Step 8: Global CSS

Add to the global stylesheet:

```css
/* Hide system cursor when hand tracking is active */
/* Only apply this via a class on <body> or <html> when tracking starts */
.hand-tracking-active * {
  cursor: none !important;
}
```

Do NOT hide the cursor by default — only when hand tracking is actually running.
The mouse fallback should show the normal cursor.

## Step 9: Verify

After integration:
1. Run the dev server
2. Check that the intro overlay appears
3. Click "ENTER WITH MOUSE" — verify normal mouse works, cursor follows mouse,
   edge scrolling works
4. Reload, click "USE HAND TRACKING" — verify webcam permission prompt appears,
   hand detection starts, cursor follows index finger
5. Make a fist — verify click fires on the element under cursor
6. Move hand to bottom edge — verify page scrolls down
7. Check that all existing site functionality (buttons, links, animations) still works

## Important Notes

- **HTTPS required in production**: Webcam API requires secure context. localhost works for dev.
- **Privacy**: All processing is local. No video data leaves the browser.
- **Performance**: MediaPipe runs on GPU via WebGL. Should not affect site performance.
- **Fallback**: If webcam is denied or MediaPipe fails to load, auto-switch to mouse.
- **Don't break existing functionality**: The hand tracking is an overlay on top of
  the existing site. Don't modify existing components unless necessary.
- **Framework adaptation**: The reference code is vanilla JS. You MUST adapt the
  patterns to match the project's framework idioms (React hooks, Vue composables,
  Svelte runes, etc.). Don't just dump vanilla JS into a React project.
