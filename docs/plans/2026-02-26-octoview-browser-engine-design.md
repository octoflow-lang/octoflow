# OctoView Browser Engine: CSS + JS + DOM

## Goal

Transform OctoView from a static HTML viewer into a real dual-mode browser engine capable of rendering dynamic websites. Web mode uses a traditional CSS/JS/DOM pipeline (CPU, correct), while the Loom Engine intercepts the OVD output to learn GPU-native rendering patterns.

## Architecture

### Dual Pipeline

```
HTML → fetch → parse (html5ever)
  → CSS cascade (cssparser + selectors)
  → JS execute (Boa) ↔ Mutable DOM
  → Layout (box model → flexbox → grid)
  → OVD semantic tree
  ├→ CPU render (correct, immediate)
  └→ Loom intercept (learns patterns, eventually GPU-native)
```

The web rendering pipeline serves two purposes:
1. **Functional browser** — renders web content correctly for users
2. **Training ground** — the Loom Engine observes OVD→pixel mappings and learns to replicate them via GPU compute shaders

### Phase 1: CSS Engine

**Goal:** Parse and apply stylesheets — biggest visual impact, no JS needed.

**Components:**
- Parse `<style>` blocks during HTML extraction
- Fetch and parse external `<link rel="stylesheet">` CSS files
- CSS selector engine: tag, `.class`, `#id`, descendant, child, attribute selectors
- Specificity calculation and cascade ordering
- Style inheritance (color, font-size, font-family inherit; background doesn't)
- Computed styles applied to OVD nodes before layout

**Crates:**
- `cssparser` (from Servo) — tokenizer and parser for CSS syntax
- `selectors` (from Servo) — selector parsing, matching, specificity

**What improves:** Most websites look dramatically better. Wikipedia, HN, GitHub all render with correct colors, fonts, spacing.

### Phase 2: Boa + Mutable DOM

**Goal:** Execute JavaScript and expose DOM APIs for dynamic content.

**Components:**
- Embed Boa JS engine (pure Rust, ES2024)
- Mutable DOM tree replacing flat OVD Vec for web mode
- DOM APIs exposed to Boa:
  - `document.getElementById()`, `document.querySelector()`, `document.querySelectorAll()`
  - `document.createElement()`, `element.appendChild()`, `element.removeChild()`
  - `element.innerHTML`, `element.textContent`, `element.setAttribute()`
  - `element.style.*` (inline style manipulation)
  - `element.classList.add/remove/toggle`
- Timer APIs: `setTimeout()`, `setInterval()`, `clearTimeout()`
- `console.log()`, `console.error()`
- DOM mutation triggers re-layout and re-render

**What improves:** Simple dynamic sites work — dropdown menus, tab panels, accordion widgets, dynamically loaded content.

### Phase 3: Events + Network

**Goal:** Handle user interaction and async data loading.

**Components:**
- Event system: `addEventListener()`, `removeEventListener()`
- Event bubbling and capturing phases
- Mouse events: click, mousedown, mouseup, mouseover, mouseout
- Keyboard events: keydown, keyup, keypress
- `fetch()` API with Promise support (Boa has Promises)
- `XMLHttpRequest` (legacy support)
- `JSON.parse()` / `JSON.stringify()` (Boa built-in)
- Cookie jar (reqwest cookie support)
- Form submission (GET/POST)

**What improves:** AJAX-heavy sites start working. SPAs that load content via fetch(). Interactive forms.

### Phase 4: Advanced Layout

**Goal:** CSS layout modes beyond vertical flow.

**Components:**
- Full CSS box model (margin, padding, border, width, height)
- Flexbox (`display: flex`, `justify-content`, `align-items`, `flex-wrap`, etc.)
- CSS Grid (basic: `grid-template-columns`, `grid-template-rows`, `gap`)
- Positioning: `position: absolute/relative/fixed/sticky`
- Float and clear
- Z-index and stacking contexts
- `overflow: hidden/scroll/auto`
- Media queries (`@media` — viewport width breakpoints)

**What improves:** Modern website layouts render correctly. Sidebars, navbars, card grids, responsive designs.

## Technology Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| CSS parsing | `cssparser` (Servo) | Battle-tested, maintained, pure Rust |
| Selector matching | `selectors` (Servo) | Specificity, matching, pseudo-classes |
| JS engine | `boa_engine` | Pure Rust, ES2024, active development |
| HTML parsing | `html5ever` | Already using, Servo ecosystem |
| HTTP | `reqwest` | Already using, cookie jar support |

## Loom Engine Integration

The Loom Engine sits between OVD generation and final pixel output:

1. **Intercept phase:** After layout produces positioned boxes, Loom receives the same data
2. **Observe phase:** Loom records CSS property → visual output mappings
3. **Replicate phase:** Loom generates GPU compute shaders that reproduce the CPU rendering
4. **Takeover phase:** When Loom's output matches CPU output within tolerance, it takes over

This happens transparently — users see correct rendering from day one via CPU, with GPU acceleration phasing in as Loom learns.

## Success Criteria

- **Phase 1:** Wikipedia, Hacker News, and GitHub render with correct styles
- **Phase 2:** Pages with simple JS (show/hide, dynamic content insertion) work
- **Phase 3:** Sites using fetch() for data loading render content
- **Phase 4:** Flexbox-based layouts (most modern sites) render correctly
