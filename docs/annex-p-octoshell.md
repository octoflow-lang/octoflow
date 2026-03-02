# OctoFlow — Annex P: OctoShell — GPU-First Desktop Environment

**Parent Document:** OctoFlow Blueprint & Architecture
**Status:** Draft
**Version:** 0.1
**Date:** February 16, 2026

---

## Table of Contents

1. What OctoShell Is (and Isn't)
2. Why a GPU-First Desktop
3. Architecture
4. OctoShell as Wayland Compositor
5. Window Management
6. Taskbar, Launcher & System UI
7. Core Apps (9 Apps, ~1,700 Lines)
8. GPU Terminal
9. Live Wallpapers & Desktop Effects
10. Theming & Customization
11. App Distribution & OctoStore
12. Performance Comparison
13. Raspberry Pi & ARM
14. OctoLinux — The Distribution
15. Security Model (Desktop Context)
16. Implementation Roadmap
17. The Growth Narrative

---

## 1. What OctoShell Is (and Isn't)

### 1.1 What It Is

OctoShell is a GPU-first desktop environment where every application is a `.flow` file and the entire user interface is rendered by a single Vulkan instance. It runs on top of Linux as a Wayland compositor, replacing GNOME, KDE, Sway, or any other desktop environment.

```
OctoShell IS:
  A desktop environment (like GNOME, KDE, Xfce)
  A Wayland compositor (like Mutter, KWin, Sway)
  An application runtime (like Electron, but GPU-native)
  A window manager (tiling, floating, stacking)
  A complete user experience built from .flow apps

OctoShell IS NOT:
  An operating system (it runs ON Linux)
  A kernel (Linux kernel handles hardware)
  A hypervisor or VM
  A replacement for Linux — it's a replacement for GNOME/KDE
```

### 1.2 The Elevator Pitch

```
CURRENT LINUX DESKTOP:
  Linux kernel → GNOME/KDE → GTK/Qt apps (each with own renderer)
  5 apps open = 5 different GUI frameworks = 1+ GB RAM
  Desktop compositor runs on CPU. Apps render on CPU. GPU idles.

OCTOSHELL:
  Linux kernel → OctoShell → .flow apps (one shared GPU renderer)
  5 apps open = 1 OctoFlow runtime = ~150 MB RAM
  Everything GPU-rendered. CPU free for actual work.
```

### 1.3 The Chromebook Analogy

```
CHROMEOS:
  Linux kernel + Chrome browser = entire user experience.
  Every app is a web app. Chrome is the desktop.
  It works. Chromebooks sell millions.
  BUT: everything is CPU-rendered web tech. Slow, heavy, janky.

OCTOSHELL:
  Linux kernel + OctoFlow runtime = entire user experience.
  Every app is a .flow app. OctoView is the browser.
  BUT: everything is GPU-rendered. Fast, light, 60fps always.

  ChromeOS reimagined with GPU-native performance.
```

---

## 2. Why a GPU-First Desktop

### 2.1 The CPU Rendering Problem

Every major desktop environment renders its user interface primarily on the CPU. The GPU sits idle while the CPU paints buttons, text, window decorations, and animations.

```
WHAT HAPPENS WHEN YOU OPEN 5 APPS ON GNOME:

  Firefox:      Skia (CPU) + WebGL (GPU for some content)     ~300 MB
  VS Code:      Chromium (CPU) + Electron                      ~400 MB
  Nautilus:     GTK4 (CPU)                                     ~80 MB
  Terminal:     VTE (CPU)                                      ~50 MB
  Settings:     GTK4 (CPU)                                     ~60 MB

  GNOME Shell:  Clutter/Mutter (CPU compositor)                ~200 MB

  TOTAL: ~1.1 GB RAM, CPU at 15-30% just rendering UI
  GPU: ~0% utilization (doing nothing)
```

```
SAME 5 APPS ON OCTOSHELL:

  Browser:      ext.ui (GPU) — one .flow app                   ~30 MB
  Editor:       ext.ui (GPU) — one .flow app                   ~25 MB
  Files:        ext.ui (GPU) — one .flow app                   ~20 MB
  Terminal:     ext.ui (GPU) — one .flow app                   ~15 MB
  Settings:     ext.ui (GPU) — one .flow app                   ~15 MB

  OctoShell:    Vulkan compositor (GPU)                         ~40 MB
  OctoFlow:     Shared runtime                                  ~20 MB

  TOTAL: ~165 MB RAM, CPU at ~2% (idle), GPU at ~5% (rendering)
```

### 2.2 The Electron Problem

The modern desktop has a hidden crisis: most "native" apps are web apps in disguise.

```
APPS THAT ARE ACTUALLY CHROMIUM BROWSERS:
  VS Code         (~400 MB)
  Slack            (~300 MB)
  Discord          (~350 MB)
  Spotify          (~300 MB)
  Teams            (~400 MB)
  Notion           (~250 MB)
  Figma desktop    (~300 MB)

  7 "apps" = 7 copies of Chromium = ~2.3 GB just for browser engines.
  Each has: V8 JS engine, Blink renderer, network stack, IPC layer.

  A modern developer's laptop runs 5-10 Electron apps daily.
  That's 2-4 GB of RAM just for duplicate browser infrastructure.
```

OctoShell eliminates this entirely. Every app shares one runtime:

```
OCTOSHELL EQUIVALENT:
  Editor:      editor.flow      (~25 MB, shared OctoFlow runtime)
  Chat:        chat.flow        (~20 MB, shared runtime)
  Music:       music.flow       (~15 MB, shared runtime)
  Dashboard:   dashboard.flow   (~20 MB, shared runtime)
  Design:      design.flow      (~25 MB, shared runtime)

  5 apps + shared runtime = ~125 MB total
  vs 5 Electron apps = ~2 GB

  16x less memory. Same functionality.
```

### 2.3 The Composition Advantage

When everything runs on one GPU through one Vulkan instance, operations that are impossible or expensive in traditional desktops become trivial:

```
IMPOSSIBLE IN GNOME/KDE, TRIVIAL IN OCTOSHELL:

  1. GPU-ACCELERATED WINDOW BLUR
     Frosted glass behind every window.
     GNOME: KWin blur extension, 15% CPU hit, stutters.
     OctoShell: GPU Gaussian blur, <0.1ms per frame, free.

  2. CROSS-APP GPU PIPELINES
     Video player feeds frames to image editor in real-time.
     GNOME: Impossible without IPC, shared memory, frame copies.
     OctoShell: Both apps share GPU buffers. Zero-copy.

  3. ML-POWERED DESKTOP FEATURES
     "Smart search" indexes files using OctoServe embedding model.
     GNOME: Requires separate ML runtime, Python, massive overhead.
     OctoShell: ext.ml runs on same GPU as UI. Already loaded.

  4. REAL-TIME DASHBOARD WALLPAPERS
     Stock ticker, system monitor, weather — live on desktop background.
     GNOME: Requires special desktop widgets, separate rendering path.
     OctoShell: Wallpaper is a .flow app. Same as any other app.

  5. 60FPS EVERYTHING, ALWAYS
     Window animations, scrolling, resizing — always smooth.
     GNOME: CPU compositor drops to 30fps under load.
     OctoShell: GPU has <5% utilization rendering UI. Never drops.
```

---

## 3. Architecture

### 3.1 The Stack

```
LAYER 4: APPLICATIONS (.flow)
  ├── Core apps (files, notes, terminal, settings, player, ...)
  ├── User apps (trading terminal, dashboard, editor, ...)
  ├── oct:// apps (web apps rendered by OctoView)
  └── Legacy apps (X11/Wayland apps composited as textures)

LAYER 3: OCTOSHELL (desktop environment)
  ├── Window manager (tiling, floating, stacking modes)
  ├── Compositor (Vulkan render pass for all windows)
  ├── Taskbar + app launcher + notification system
  ├── System dialogs (file picker, permission prompts)
  └── All implemented in .flow using ext.ui

LAYER 2: OCTOFLOW RUNTIME
  ├── OctoFlow compiler (parse .flow → SPIR-V → Vulkan dispatch)
  ├── ext.ui (widget library + GPU rendering pipeline)
  ├── ext.media (video/audio/image via OctoMedia)
  ├── ext.ml (ML inference via OctoServe)
  ├── ext.crypto (GPU-accelerated security)
  ├── ext.net (networking + oct:// protocol)
  └── std.os (file system, process management, hardware info)

LAYER 1: SYSTEM
  ├── Linux kernel (hardware, drivers, networking, processes)
  ├── Vulkan driver (GPU access — mesa or nvidia-open)
  ├── libinput (input devices — keyboard, mouse, touch)
  ├── DRM/KMS (display output — resolution, refresh rate)
  └── PipeWire (audio — playback, capture, routing)
```

### 3.2 Rendering Architecture

```
ONE VULKAN INSTANCE. ONE RENDER PASS. ALL WINDOWS.

PER FRAME (16.6ms budget at 60fps, OctoShell uses <2ms):

  INPUT (0.02ms)
    libinput events: keyboard, mouse, touch
    Determine: which window is focused/hovered
    Route events to correct app

  APP UPDATE (0.1-0.5ms)
    Each app processes its events
    Updates state (reactive variables)
    Regenerates widget tree (diff against previous frame)

  LAYOUT (0.1ms)
    Window positions and sizes (window manager)
    Widget layout within each window (ext.ui flexbox)

  COMPOSITE (0.5-1.0ms)
    Single Vulkan render pass:
      Draw call 1: Desktop background (wallpaper texture or shader)
      Draw call 2: Window backgrounds (instanced rectangles)
      Draw call 3: Window content (app widget geometry)
      Draw call 4: Window decorations (title bars, borders, shadows)
      Draw call 5: Text (all text across all windows, one glyph atlas)
      Draw call 6: Overlays (taskbar, notifications, launcher)
      Draw call 7: Cursor

    Total: 7-10 draw calls per frame for ENTIRE desktop.

    GNOME/KDE: Each window = separate render target → composite.
    OctoShell: All windows in ONE render pass. No compositing overhead.

  PRESENT (0.01ms)
    Vulkan present to swapchain → appears on display.

TOTAL: ~1-2ms per frame
BUDGET: 16.6ms (60fps) or 8.3ms (120fps)
HEADROOM: 85-95% of frame time available for app compute
```

### 3.3 Legacy App Support

Not every app will be `.flow` from day one. OctoShell composites traditional Linux apps:

```
.flow APPS (native):
  Render directly into OctoShell's Vulkan render pass.
  Zero-copy. Maximum performance. Full integration.

WAYLAND APPS (GTK, Qt, Electron):
  App renders to its own Wayland buffer.
  OctoShell imports buffer as Vulkan texture.
  Composited into the desktop render pass.
  Performance: same as any Wayland compositor (good).
  Integration: window management works, but no ext.ui theming.

X11 APPS (legacy):
  Run through XWayland compatibility layer.
  OctoShell composites as texture (same as Wayland path).
  Performance: acceptable. Some features may not work perfectly.
  Integration: minimal. Window management works.

PRIORITY:
  .flow apps for everything new.
  Wayland for existing Linux apps (Firefox, LibreOffice, GIMP).
  X11 for truly legacy software.

  Over time: community rewrites popular apps in .flow.
  300-line .flow file manager replaces 50,000-line Nautilus.
```

---

## 4. OctoShell as Wayland Compositor

### 4.1 Two Launch Modes

```
MODE A: STANDALONE COMPOSITOR (production)
  Linux boots → login manager → OctoShell starts
  OctoShell IS the Wayland compositor.
  Replaces GNOME/KDE entirely.

  User sees: OctoFlow desktop from login to logout.
  Equivalent to: selecting "GNOME" or "KDE" at login screen.

  Implementation: Smithay (Rust Wayland compositor library)
  or custom Wayland protocol handler on DRM/KMS.

MODE B: WINDOWED (development / trial)
  User runs GNOME/KDE normally.
  OctoShell runs as a window inside the existing desktop.
  Like a "desktop within a desktop."

  User can: try OctoShell without commitment.
  Then: switch to Mode A when comfortable.

  Implementation: OctoShell as a regular Wayland/X11 window
  with Vulkan rendering inside it.

SHIP ORDER:
  Mode B first (Month 12-13): easy adoption, no risk.
  Mode A later (Month 14-15): full experience, daily driver.
```

### 4.2 Compositor Implementation

```
USING SMITHAY (Rust Wayland compositor library):

Smithay provides:
  Wayland protocol handling
  DRM/KMS display management
  libinput integration
  XWayland support
  Buffer management (wl_shm, dma-buf)

OctoShell adds:
  Vulkan rendering pipeline (replace wlroots renderer)
  .flow app integration (native GPU rendering)
  ext.ui widget rendering in compositor context
  Window management logic (tiling, floating)
  Desktop UI (.flow taskbar, launcher, notifications)

ALTERNATIVE: wlroots (C library, used by Sway/Hyprland)
  More mature, but C-based.
  Smithay is Rust-native → better integration with OctoFlow.

ALTERNATIVE: Custom from scratch on DRM/KMS
  Maximum control, minimum dependencies.
  Most work. Consider only if Smithay proves insufficient.
```

### 4.3 Display Management

```
MULTI-MONITOR:
  OctoShell handles multiple displays natively.
  Each display = separate Vulkan swapchain.
  Workspaces can span or be per-monitor.

  $ octo-shell displays
  Display 1: HDMI-1 (2560x1440 @ 144Hz)
  Display 2: DP-1 (1920x1080 @ 60Hz)

  Window dragging between monitors: <1ms response.
  GPU renders both displays in same Vulkan instance.

HIGH REFRESH RATE:
  60fps: 16.6ms budget, OctoShell uses <2ms (88% headroom)
  120fps: 8.3ms budget, OctoShell uses <2ms (76% headroom)
  144fps: 6.9ms budget, OctoShell uses <2ms (71% headroom)
  240fps: 4.1ms budget, OctoShell uses <2ms (51% headroom)

  OctoShell can drive 240Hz displays without dropping frames.
  GNOME struggles at 144Hz.

HDR:
  Vulkan supports HDR output (VK_EXT_swapchain_colorspace).
  OctoShell can render 10-bit color, HDR10, Dolby Vision.
  ext.ui themes with HDR-aware color management.

  GNOME: HDR support is experimental/broken (as of 2026).
  OctoShell: HDR from day one (Vulkan handles it).

VARIABLE REFRESH RATE (FreeSync / G-Sync):
  Vulkan FIFO_RELAXED presentation mode.
  Frame delivered when ready, display syncs to it.
  No tearing, no vsync stutter.
```

---

## 5. Window Management

### 5.1 Three Modes

```
FLOATING (default):
  Traditional windows. Drag to move, resize edges.
  Like GNOME/KDE/Windows/macOS.
  Best for: casual use, creative work, mixed window sizes.

TILING (power user):
  Windows automatically tile to fill screen.
  Like i3/Sway/Hyprland.
  Best for: development, trading terminals, multitasking.

  Keyboard-driven: Super+H (split horizontal), Super+V (split vertical)
  New window takes half the focused window's space.

STACKING (presentation):
  Windows stack on top of each other, tabs along top.
  Like browser tabs but for any app.
  Best for: limited screen space, focused work.

SWITCH: Super+F (floating), Super+T (tiling), Super+S (stacking)
Per-workspace mode: Workspace 1 = tiling, Workspace 2 = floating.
```

### 5.2 Window Management Code

```flow
// octoshell/wm.flow — Window Manager core

import ext.ui
import std.os

fn main():
    let mut windows = []
    let mut focused = null
    let mut mode = "floating"
    let mut workspaces = [[], [], [], []]
    let mut current_workspace = 0

    ui.app("OctoShell", fullscreen=true, theme=ui.theme.dark):
        // Desktop background
        desktop_background()

        // Windows layer
        ui.layout.stack():
            for win in workspaces[current_workspace]:
                if mode == "floating":
                    floating_window(win, focused == win.id)
                else if mode == "tiling":
                    tiled_window(win, focused == win.id)
                else:
                    stacked_window(win, focused == win.id)

        // Taskbar (always on top)
        taskbar(workspaces, current_workspace, focused)

        // App launcher overlay
        if show_launcher:
            app_launcher()

        // Notification stack
        notification_stack()

fn floating_window(win, is_focused):
    ui.layout.column(
        x=win.x, y=win.y,
        width=win.width, height=win.height,
        background=ui.theme.surface,
        border=if is_focused:
            ui.border(2, ui.theme.primary)
        else:
            ui.border(1, ui.gray(60)),
        shadow=if is_focused:
            ui.shadow(0, 4, 16, ui.rgba(0, 0, 0, 0.3))
        else:
            ui.shadow(0, 2, 8, ui.rgba(0, 0, 0, 0.15)),
        border_radius=8,
        on_mousedown=fn(): focused = win.id
    ):
        // Title bar
        title_bar(win)

        // App content (rendered by the app's .flow)
        ui.layout.column(flex=1, clip=true):
            win.app.render()

fn title_bar(win):
    ui.layout.row(
        height=32, padding_h=8, gap=8,
        background=ui.gray(25),
        border_radius_top=8,
        on_drag=fn(dx, dy): win.x += dx; win.y += dy
    ):
        ui.icon(win.icon, size=14)
        ui.text(win.title, size=13, color=ui.gray(200))
        ui.layout.spacer()
        ui.icon_button(ui.icon.minimize, size=12,
            on_click=fn(): minimize(win))
        ui.icon_button(ui.icon.maximize, size=12,
            on_click=fn(): toggle_maximize(win))
        ui.icon_button(ui.icon.close, size=12,
            color=ui.rgb(220, 60, 60),
            on_click=fn(): close(win))
```

### 5.3 Keyboard Shortcuts

```
WINDOW MANAGEMENT:
  Super + Enter       Open terminal
  Super + Space       App launcher
  Super + Q           Close focused window
  Super + F           Floating mode
  Super + T           Tiling mode
  Super + M           Maximize / restore
  Super + Arrow       Snap to half-screen (left/right/top/bottom)
  Super + 1-4         Switch workspace
  Super + Shift + 1-4 Move window to workspace
  Alt + Tab           Window switcher
  Alt + F4            Close window

TILING MODE:
  Super + H           Split horizontal
  Super + V           Split vertical
  Super + J/K/L/;     Focus left/down/up/right
  Super + Shift + J/K/L/; Move window left/down/up/right
  Super + R           Resize mode (arrows to resize, Esc to exit)

SYSTEM:
  Super + L           Lock screen
  Super + E           File manager
  Super + B           Browser (OctoView)
  Super + P           Display settings
  Ctrl + Alt + T      Terminal (alternative)
  Print Screen        Screenshot (GPU capture — instant)
```

---

## 6. Taskbar, Launcher & System UI

### 6.1 Taskbar

```flow
fn taskbar(workspaces, current_ws, focused):
    ui.layout.row(
        height=48,
        background=ui.rgba(20, 20, 20, 0.9),
        blur=16,            // frosted glass — GPU-computed Gaussian blur
        position=bottom,
        dock=true,          // always visible
        padding_h=8
    ):
        // App launcher button
        ui.icon_button(ui.icon.grid, size=24,
            tooltip="Applications",
            on_click=fn(): show_launcher = !show_launcher)

        ui.layout.divider(vertical=true, color=ui.gray(40))

        // Workspace indicators
        for i in 0..4:
            ui.layout.column(
                width=32, height=32, align=center, justify=center,
                border_radius=6,
                background=if i == current_ws:
                    ui.theme.primary_dim else: ui.transparent,
                on_click=fn(): current_ws = i
            ):
                ui.text(str(i + 1), size=12,
                    weight=if i == current_ws: ui.bold else: ui.normal,
                    color=if i == current_ws:
                        ui.theme.primary else: ui.gray(120))
                if len(workspaces[i]) > 0:
                    ui.badge_dot(color=ui.theme.primary)

        ui.layout.divider(vertical=true, color=ui.gray(40))

        // Running apps
        for win in workspaces[current_ws]:
            ui.layout.column(
                width=44, height=44, align=center, justify=center,
                border_radius=8,
                background=if focused == win.id:
                    ui.rgba(255, 255, 255, 0.1) else: ui.transparent,
                border_bottom=if focused == win.id:
                    ui.border(3, ui.theme.primary) else: none,
                tooltip=win.title,
                on_click=fn(): focus(win)
            ):
                ui.icon(win.icon, size=24)
                if win.has_notification:
                    ui.badge(win.notification_count)

        ui.layout.spacer()

        // System tray
        system_tray()

fn system_tray():
    ui.layout.row(gap=12, align=center):
        // Network
        ui.icon(network_icon(), size=16, color=ui.gray(180),
            tooltip=network_tooltip())

        // Volume
        ui.icon(volume_icon(), size=16, color=ui.gray(180),
            tooltip="Volume: {volume}%",
            on_click=fn(): show_volume_slider = !show_volume_slider)

        // Battery (if laptop)
        if has_battery:
            ui.icon(battery_icon(), size=16,
                color=if battery_pct < 20: ui.red else: ui.gray(180),
                tooltip="Battery: {battery_pct}%")

        // Clock
        ui.text(time_now("%H:%M"), size=14, color=ui.gray(200))
        ui.text(time_now("%a %b %d"), size=11, color=ui.gray(140))
```

### 6.2 App Launcher

```flow
fn app_launcher():
    let mut search = ""
    let apps = std.os.list_apps("~/.octoflow/apps/")
    let filtered = apps
        |> filter(fn(a): contains(lower(a.name), lower(search)))
        |> sort_by(fn(a): a.launch_count, descending=true)

    // Dimmed backdrop
    ui.layout.stack(
        fullscreen=true,
        background=ui.rgba(0, 0, 0, 0.5),
        on_click=fn(): show_launcher = false
    ):
        // Launcher panel
        ui.layout.column(
            center=true,
            max_width=600, max_height=500,
            background=ui.rgba(25, 25, 25, 0.95),
            blur=24,
            border_radius=16,
            padding=24,
            shadow=ui.shadow(0, 8, 32, ui.rgba(0, 0, 0, 0.5))
        ):
            // Search bar
            ui.search(
                placeholder="Search applications...",
                value=search,
                on_change=fn(v): search = v,
                autofocus=true,
                size=18
            )

            // App grid
            ui.layout.grid(
                cols=4, gap=16, padding_top=16,
                max_height=400, scroll=true
            ):
                for app in filtered:
                    ui.layout.column(
                        align=center, padding=12,
                        border_radius=12,
                        background=ui.transparent,
                        hover_background=ui.rgba(255, 255, 255, 0.08),
                        on_click=fn(): launch(app); show_launcher = false
                    ):
                        ui.icon(app.icon, size=48)
                        ui.text(app.name, size=12, color=ui.gray(180),
                            max_lines=1, ellipsis=true)
```

### 6.3 Notification System

```flow
fn notification_stack():
    ui.layout.column(
        position=top_right,
        margin=12,
        gap=8,
        max_width=360
    ):
        for notif in notifications |> take(5):
            ui.layout.row(
                padding=12, gap=12,
                background=ui.rgba(30, 30, 30, 0.95),
                blur=12,
                border_radius=12,
                border=ui.border(1, ui.gray(50)),
                shadow=ui.shadow(0, 4, 12, ui.rgba(0, 0, 0, 0.3)),
                transition=ui.transition(duration=300, easing=ui.ease.out),
                on_click=fn(): focus_app(notif.source)
            ):
                ui.icon(notif.icon, size=24)
                ui.layout.column(flex=1):
                    ui.text(notif.title, size=13, weight=ui.bold)
                    ui.text(notif.body, size=12, color=ui.gray(160),
                        max_lines=2, ellipsis=true)
                ui.icon_button(ui.icon.close, size=12,
                    on_click=fn(e): e.stop(); dismiss(notif))
```

---

## 7. Core Apps (9 Apps, ~1,700 Lines Total)

Every core app ships with OctoShell. All are `.flow` files. All are GPU-rendered. The total codebase for 9 apps is smaller than a single GTK widget.

### 7.1 App Inventory

```
APP             LINES    DESCRIPTION
──────────────────────────────────────────────────────────────
Files            ~200    File manager with GPU thumbnails
Notes            ~150    Markdown editor with live preview
Gallery          ~100    Image viewer with OctoMedia filters
Player           ~35     Video player (from Annex M)
Monitor          ~120    System monitor with GPU charts
Settings         ~200    System configuration panel
OctoView         ~500    Browser (oct:// + HTTPS bridge)
Terminal         ~150    GPU-rendered terminal emulator
OctoStore        ~250    App store for .flow apps
──────────────────────────────────────────────────────────────
TOTAL           ~1,705

COMPARISON:
  GNOME core apps:   ~500,000+ lines (C, Vala, JS)
  KDE core apps:    ~1,000,000+ lines (C++, QML)
  OctoShell apps:    ~1,700 lines (.flow)
```

### 7.2 Files (File Manager) — ~200 Lines

```flow
import ext.ui
import ext.media
import std.os

fn main():
    let mut path = std.os.home_dir()
    let mut entries = std.os.list_dir(path)
    let mut view_mode = "grid"   // grid, list, columns
    let mut selected = null
    let mut preview = null

    ui.app("Files", 900, 600, theme=ui.theme.dark):
        // Toolbar
        ui.layout.row(height=44, padding_h=12, gap=8,
            background=ui.gray(25)):
            ui.icon_button(ui.icon.arrow_left, on_click=fn(): go_back())
            ui.icon_button(ui.icon.arrow_up, on_click=fn(): go_up())
            ui.text(path, size=13, flex=1, color=ui.gray(180))
            ui.search(placeholder="Search...", width=200)
            ui.icon_button(ui.icon.grid,
                active=view_mode == "grid",
                on_click=fn(): view_mode = "grid")
            ui.icon_button(ui.icon.list,
                active=view_mode == "list",
                on_click=fn(): view_mode = "list")

        // Content area
        ui.layout.row(flex=1):
            // Sidebar (bookmarks)
            ui.layout.column(width=180, padding=8, background=ui.gray(20)):
                ui.text("Favorites", size=11, color=ui.gray(120))
                sidebar_item("Home", ui.icon.home, std.os.home_dir())
                sidebar_item("Documents", ui.icon.folder, home + "/Documents")
                sidebar_item("Downloads", ui.icon.download, home + "/Downloads")
                sidebar_item("Pictures", ui.icon.image, home + "/Pictures")

            // File grid/list
            ui.layout.column(flex=1, padding=8):
                if view_mode == "grid":
                    file_grid(entries, selected)
                else:
                    file_list(entries, selected)

            // Preview panel
            if preview != null:
                preview_panel(preview)

fn file_grid(entries, selected):
    ui.layout.grid(cols=auto(120), gap=4):
        for entry in entries:
            ui.layout.column(
                width=110, padding=8, align=center,
                border_radius=8,
                background=if selected == entry.name:
                    ui.theme.primary_dim else: ui.transparent,
                on_click=fn(): selected = entry.name,
                on_dblclick=fn(): open_entry(entry)
            ):
                // GPU-rendered thumbnail for images
                if entry.is_image:
                    ui.media.image(entry.path, width=80, height=80,
                        fit=cover, border_radius=4)
                else:
                    ui.icon(entry_icon(entry), size=48)
                ui.text(entry.name, size=11, max_lines=2,
                    ellipsis=true, align=center)
```

Image thumbnails are rendered by the OctoMedia pipeline — the same GPU that renders the UI. No separate thumbnail daemon. No cache warming. Instant.

### 7.3 Player (Video Player) — ~35 Lines

The complete video player from Annex M:

```flow
import ext.ui
import ext.ui.media

fn main(args):
    let file = if len(args) > 0: args[0]
        else: ui.dialog.file_open(["*.mp4", "*.mkv", "*.avi"])
    let video = media.open(file)
    let mut playing = true
    let mut pos = 0.0
    let mut vol = 0.8

    ui.app(file, 960, 640, theme=ui.theme.dark):
        ui.layout.column():
            ui.media.video(video, playing=playing, position=pos,
                volume=vol, on_time_update=fn(t): pos = t)

            ui.layout.row(height=52, padding=8, gap=8):
                ui.icon_button(
                    if playing: ui.icon.pause else: ui.icon.play,
                    on_click=fn(): playing = !playing)
                ui.slider(0.0, video.duration, value=pos,
                    on_change=fn(v): pos = v)
                ui.text(ftime(pos) + " / " + ftime(video.duration),
                    size=12)
                ui.slider(0.0, 1.0, value=vol,
                    on_change=fn(v): vol = v, width=80)

fn ftime(s):
    format("{:02}:{:02}", floor(s / 60), floor(mod(s, 60)))
```

**35 lines. GPU-decoded, GPU-rendered. ~8 MB binary. ~30 MB RAM. vs VLC: 150 MB install, 80 MB RAM, 500+ source files.**

### 7.4 Monitor (System Monitor) — ~120 Lines

```flow
import ext.ui
import ext.ui.chart
import std.os

fn main():
    let mut cpu_history = []
    let mut gpu_history = []
    let mut mem_history = []
    let mut net_history = []

    // Poll system stats every 500ms
    std.os.interval(500, fn():
        cpu_history = cpu_history |> append(std.os.cpu_usage()) |> tail(120)
        gpu_history = gpu_history |> append(std.os.gpu_usage()) |> tail(120)
        mem_history = mem_history |> append(std.os.mem_usage()) |> tail(120)
        net_history = net_history |> append(std.os.net_throughput()) |> tail(120)
    )

    ui.app("System Monitor", 800, 600, theme=ui.theme.dark):
        ui.layout.column(padding=16, gap=16):
            ui.heading("System Monitor")

            ui.layout.grid(cols=2, gap=16):
                // CPU chart
                ui.layout.column(padding=12, background=ui.gray(22),
                    border_radius=8):
                    ui.text("CPU", size=12, color=ui.gray(140))
                    ui.text("{cpu_history[-1]:.0}%", size=24, weight=ui.bold)
                    ui.chart.line(cpu_history, height=120,
                        color=ui.rgb(66, 133, 244),
                        fill=ui.rgba(66, 133, 244, 0.1),
                        y_min=0, y_max=100)

                // GPU chart
                ui.layout.column(padding=12, background=ui.gray(22),
                    border_radius=8):
                    ui.text("GPU", size=12, color=ui.gray(140))
                    ui.text("{gpu_history[-1]:.0}%", size=24, weight=ui.bold)
                    ui.chart.line(gpu_history, height=120,
                        color=ui.rgb(234, 67, 53),
                        fill=ui.rgba(234, 67, 53, 0.1),
                        y_min=0, y_max=100)

                // Memory chart
                ui.layout.column(padding=12, background=ui.gray(22),
                    border_radius=8):
                    ui.text("Memory", size=12, color=ui.gray(140))
                    let mem_gb = mem_history[-1] * std.os.total_mem() / 100
                    ui.text("{mem_gb:.1} GB", size=24, weight=ui.bold)
                    ui.chart.line(mem_history, height=120,
                        color=ui.rgb(52, 168, 83),
                        fill=ui.rgba(52, 168, 83, 0.1),
                        y_min=0, y_max=100)

                // Network chart
                ui.layout.column(padding=12, background=ui.gray(22),
                    border_radius=8):
                    ui.text("Network", size=12, color=ui.gray(140))
                    ui.text("{net_history[-1]:.1} MB/s", size=24, weight=ui.bold)
                    ui.chart.line(net_history, height=120,
                        color=ui.rgb(251, 188, 4),
                        fill=ui.rgba(251, 188, 4, 0.1))
```

All four charts update at 2Hz data rate, render at 60fps. GPU charts handle this trivially — it's just updating a few vertices in a GPU buffer. The same charts in GNOME System Monitor use Cairo (CPU) and stutter visibly.

### 7.5 Terminal (GPU-Rendered) — ~150 Lines

See section 8 for detailed specification.

### 7.6 Other Core Apps

```
NOTES (~150 lines):
  Split-pane markdown editor.
  Left: ext.ui.textarea (editing)
  Right: ext.ui.markdown (GPU-rendered preview)
  Updates live on every keystroke.
  Saves to ~/.octoflow/notes/ as .md files.

GALLERY (~100 lines):
  Image viewer with thumbnail grid.
  GPU zoom/pan (same as ext.ui.media.image).
  OctoMedia filter presets accessible from toolbar.
  Apply cinematic, warm, cool grading to any photo.

SETTINGS (~200 lines):
  Theme selection (dark, light, custom)
  Display settings (resolution, refresh rate, scaling)
  Network configuration
  Account management (OctoName identity)
  Security settings (permissions, trusted publishers)
  About (OctoFlow version, GPU info, system info)

OCTOSTORE (~250 lines):
  Browse ext.* modules and community .flow apps.
  Search, install, update, remove.
  Publisher verification (Ed25519 signatures).
  Permission review before install.
  Like a minimal app store built entirely in .flow.
```

---

## 8. GPU Terminal

### 8.1 Why GPU Terminal Matters

Every mainstream terminal emulator renders text on the CPU. Even "GPU-accelerated" terminals (Alacritty, Kitty) use OpenGL for text rendering but still do ANSI parsing and layout on the CPU.

```
TERMINAL RENDERING PIPELINE:

TRADITIONAL (gnome-terminal, iTerm2):
  PTY output → ANSI parser (CPU) → text layout (CPU) →
  glyph rasterize (CPU) → cairo/pango paint (CPU) →
  copy to GPU → composite

ALACRITTY/KITTY (GPU-assisted):
  PTY output → ANSI parser (CPU) → text layout (CPU) →
  glyph atlas lookup (CPU) → OpenGL quad generation (CPU) →
  GPU renders quads

OCTOTERMINAL (fully GPU-integrated):
  PTY output → ANSI parser (CPU) →
  glyph atlas + quad generation (ext.ui text pipeline) →
  Vulkan render (same pass as rest of desktop)

  The terminal IS an ext.ui widget. Same rendering path
  as every other .flow app. No separate renderer.
```

### 8.2 Performance

```
OPERATION              GNOME-TERM    ALACRITTY    OCTOTERMINAL
───────────────────────────────────────────────────────────────
cat large_file.txt     Visible lag    Fast         Instant
  (100,000 lines)      (CPU paint)   (GPU quads)  (GPU fill)

Scroll speed           Stutters       Smooth       Smooth
  (page up/down)       at speed       60fps        60fps

Resize                 Reflow lag     Fast         Instant
                       (CPU layout)   (GPU)        (GPU)

Transparency           CPU composite  GL blend     Vulkan blend
                       Slow           Fast         Free

Blur behind            Not possible   Not possible GPU Gaussian
                                                   <0.1ms

Memory                 ~50 MB         ~30 MB       ~15 MB

Scrollback             CPU memory     CPU memory   GPU buffer
  (100K lines)         ~100 MB        ~50 MB       ~20 MB
                                                    (compressed)
```

### 8.3 Terminal Code

```flow
import ext.ui
import std.os

fn main(args):
    let shell = std.os.env("SHELL", "/bin/bash")
    let pty = std.os.spawn_pty(shell)
    let mut buffer = terminal_buffer(rows=50, cols=120)
    let mut scroll_offset = 0

    pty.on_output(fn(data):
        buffer.write(data)  // ANSI parser updates buffer
    )

    ui.app("Terminal", 800, 500, theme=ui.theme.dark):
        ui.layout.column():
            // Terminal viewport
            ui.layout.column(
                flex=1,
                padding=4,
                background=ui.rgba(15, 15, 15, 0.95),
                font=ui.font.mono("JetBrains Mono", 13),
                on_scroll=fn(dy): scroll_offset = clamp(
                    scroll_offset + dy, 0, buffer.history_len()),
                on_keydown=fn(key): pty.write(key.to_ansi()),
                on_paste=fn(text): pty.write(text),
                focusable=true
            ):
                // Render visible lines from buffer
                let visible = buffer.lines(scroll_offset, scroll_offset + 50)
                for line in visible:
                    terminal_line(line)

            // Scrollbar (thin, auto-hide)
            if buffer.history_len() > 50:
                ui.scrollbar(
                    total=buffer.history_len(),
                    visible=50,
                    position=scroll_offset,
                    on_scroll=fn(pos): scroll_offset = pos
                )

fn terminal_line(line):
    ui.layout.row(height=18):
        for span in line.spans:
            ui.text(span.text,
                color=ansi_color(span.fg),
                background=ansi_color(span.bg),
                weight=if span.bold: ui.bold else: ui.normal,
                italic=span.italic,
                underline=span.underline)
```

### 8.4 Terminal Integration Features

```
FEATURES ONLY POSSIBLE IN GPU-INTEGRATED TERMINAL:

  1. INLINE IMAGE RENDERING
     $ octo-media preview photo.jpg
     → Image rendered inline in terminal (GPU texture)
     Same GPU, zero copy. Kitty protocol compatible.

  2. INLINE CHART RENDERING
     $ octo-media stats data.csv
     → ext.ui.chart rendered inline in terminal
     Interactive: zoom, pan, crosshair — IN the terminal.

  3. GPU-FILTERED OUTPUT
     $ cat log.txt | octo-media apply highlight_errors
     → GPU highlights error lines in red, warnings in yellow
     → 10x faster than grep --color for large files

  4. LIVE OctoMedia PREVIEW
     $ octo-media apply cinematic photo.jpg --preview
     → Split terminal: command output left, image preview right
     → Preview updates in real-time as you adjust parameters
```

---

## 9. Live Wallpapers & Desktop Effects

### 9.1 GPU-Powered Wallpapers

Because the desktop background is rendered by the same Vulkan instance as everything else, live wallpapers cost essentially nothing:

```flow
// wallpapers/gradient.flow — Animated gradient wallpaper

import ext.ui

fn main():
    let mut time = 0.0

    ui.canvas(fullscreen=true, render=fn(ctx):
        for y in 0..ctx.height:
            for x in 0..ctx.width:
                let r = sin(x * 0.01 + time) * 0.5 + 0.5
                let g = cos(y * 0.01 + time * 0.7) * 0.3 + 0.2
                let b = sin((x + y) * 0.005 + time * 0.5) * 0.4 + 0.3
                ctx.pixel(x, y, rgb(r * 255, g * 255, b * 255))
        time += 0.016
    )
```

```
CPU COST OF THIS WALLPAPER:
  GNOME with animated wallpaper extension: 5-15% CPU, frequent stutters
  OctoShell: 0% CPU (GPU computes sin/cos for all pixels in <0.5ms)
```

### 9.2 Data-Driven Wallpapers

```flow
// wallpapers/market.flow — Live market data wallpaper

import ext.ui
import ext.ui.chart
import ext.net

fn main():
    let mut prices = []

    net.websocket("wss://stream.example.com/xauusd",
        on_message=fn(tick):
            prices = prices |> append(tick.price) |> tail(500)
    )

    ui.canvas(fullscreen=true, render=fn(ctx):
        // Dark background with subtle gradient
        ctx.fill(ui.rgb(12, 12, 18))

        // Large semi-transparent price chart
        chart.line(prices,
            x=0, y=ctx.height * 0.3,
            width=ctx.width, height=ctx.height * 0.6,
            color=ui.rgba(66, 133, 244, 0.3),
            fill=ui.rgba(66, 133, 244, 0.05),
            line_width=2)

        // Current price (large, subtle)
        if len(prices) > 0:
            ctx.text(format("{:.2}", prices[-1]),
                x=ctx.width / 2, y=ctx.height / 2,
                size=120, color=ui.rgba(255, 255, 255, 0.08),
                align=center)
    )
```

Your XAUUSD price streaming live as your desktop wallpaper. GPU-rendered. Zero CPU usage. Impossible in any other desktop environment.

### 9.3 Desktop Effects

```
ALL FREE (GPU-COMPUTED, <0.1ms EACH):

  Window shadows:    Gaussian blur beneath each window
  Transparency:      Alpha blending on window backgrounds
  Frosted glass:     Gaussian blur of content behind translucent panels
  Rounded corners:   SDF (Signed Distance Field) rendering
  Animations:        GPU-interpolated position, size, opacity
  Window glow:       Bloom shader on focused window border
  Workspace switch:  3D cube rotation or slide animation
  Alt-Tab:           GPU-rendered window thumbnails (live preview)
  Screenshot:        GPU pixel readback (instant, no flicker)
  Screen recording:  Vulkan frame capture (zero-copy to encoder)
```

---

## 10. Theming & Customization

### 10.1 Theme System

```flow
// themes/momentum.flow — Custom Momentum FX theme

let momentum_theme = ui.theme.create(
    // Colors
    primary=ui.rgb(138, 43, 226),         // Momentum purple
    primary_dim=ui.rgba(138, 43, 226, 0.15),
    secondary=ui.rgb(0, 200, 200),         // Cyan accent
    background=ui.rgb(12, 12, 18),         // Deep dark
    surface=ui.rgb(22, 22, 30),            // Card surface
    text=ui.rgb(230, 230, 240),            // Light text
    text_dim=ui.rgb(140, 140, 160),        // Muted text
    error=ui.rgb(220, 50, 50),
    warning=ui.rgb(240, 180, 40),
    success=ui.rgb(50, 200, 80),

    // Typography
    font_family="Inter",
    font_mono="JetBrains Mono",
    font_size_base=14,

    // Spacing
    border_radius=8,
    spacing_unit=4,

    // Effects
    shadow_color=ui.rgba(0, 0, 0, 0.3),
    blur_amount=16,

    // Window decorations
    titlebar_height=32,
    titlebar_font_size=13,
    window_border_width=1,
    window_border_color=ui.gray(40),
    window_border_focused=ui.rgb(138, 43, 226)
)
```

### 10.2 Theme Applies Everywhere

```
A SINGLE THEME CONTROLS:
  Window decorations (title bar, borders, shadows)
  Taskbar appearance
  App launcher
  Notifications
  All core apps (files, terminal, settings, ...)
  All community .flow apps (they inherit theme)
  Syntax highlighting in terminal and editor
  Chart colors in dashboard and monitor

  ONE THEME FILE. EVERYTHING MATCHES.

  GNOME: Theme applies to GTK apps. Qt apps look different.
         Electron apps ignore the theme entirely.
         VS Code has its own theme. Firefox has its own.
         Nothing matches.

  OctoShell: Everything is ext.ui. One theme. Everything matches.
```

---

## 11. App Distribution & OctoStore

### 11.1 App Format

```
.flow APP BUNDLE:
  app_name/
  ├── oct.manifest        App metadata, permissions, dependencies
  ├── main.flow           Entry point
  ├── assets/             Images, fonts, data files
  │   ├── icon.svg
  │   └── data/
  └── modules/            Local .flow modules
      └── helpers.flow

oct.manifest:
{
    "name": "Trading Terminal",
    "id": "terminal.momentum-fx.oct",
    "version": "1.0.0",
    "author": "oct:ed25519:7f3a...",
    "entry": "main.flow",
    "icon": "assets/icon.svg",
    "permissions": ["network", "gpu_compute", "storage"],
    "dependencies": {
        "ext.ui": "^1.0",
        "ext.ui.chart": "^1.0",
        "ext.net": "^1.0"
    },
    "category": "Finance",
    "description": "XAUUSD real-time trading terminal with GPU-rendered charts"
}
```

### 11.2 Install & Launch

```bash
# Install from OctoStore
$ octo install terminal.momentum-fx.oct

# Install from URL
$ octo install oct://momentum-fx.oct/apps/terminal

# Install from local file
$ octo install ./my-app/

# Launch
$ octo run terminal.momentum-fx.oct
# or: click the icon in the app launcher

# Update
$ octo update terminal.momentum-fx.oct

# Remove
$ octo remove terminal.momentum-fx.oct
```

### 11.3 OctoStore App

```flow
// octostore/main.flow — App Store (~250 lines)

import ext.ui
import ext.net

fn main():
    let mut category = "featured"
    let mut search = ""
    let mut apps = fetch_apps(category)

    ui.app("OctoStore", 900, 650, theme=ui.theme.dark):
        ui.layout.row():
            // Sidebar (categories)
            ui.layout.column(width=200, padding=12,
                background=ui.gray(18)):
                ui.heading("OctoStore", size=18)
                ui.layout.spacer(height=12)
                store_category("Featured", "featured")
                store_category("Productivity", "productivity")
                store_category("Development", "development")
                store_category("Finance", "finance")
                store_category("Creative", "creative")
                store_category("Utilities", "utilities")
                store_category("Games", "games")

            // Main content
            ui.layout.column(flex=1, padding=16):
                ui.search(placeholder="Search apps...",
                    value=search, on_change=fn(v):
                        search = v; apps = search_apps(v))

                ui.layout.grid(cols=2, gap=12, padding_top=12):
                    for app in apps:
                        app_card(app)

fn app_card(app):
    ui.layout.row(padding=12, gap=12,
        background=ui.gray(22), border_radius=10,
        hover_background=ui.gray(28)):
        ui.icon(app.icon, size=48, border_radius=10)
        ui.layout.column(flex=1):
            ui.text(app.name, weight=ui.bold)
            ui.text(app.description, size=12,
                color=ui.gray(150), max_lines=2)
            ui.layout.row(gap=8):
                ui.text("* {app.rating:.1}", size=11)
                ui.text(app.download_count + " installs",
                    size=11, color=ui.gray(120))
        ui.button(
            if app.installed: "Open" else: "Install",
            on_click=fn():
                if app.installed: launch(app) else: install(app))
```

---

## 12. Performance Comparison

### 12.1 Desktop Environment Benchmarks

```
METRIC              GNOME 46    KDE 6      OCTOSHELL
──────────────────────────────────────────────────────────
Frame time          8-16ms      8-16ms     <2ms
Render method       CPU+GL      CPU+GL     GPU (Vulkan)
RAM (idle desktop)  ~400 MB     ~350 MB    ~80 MB
RAM per app         50-200 MB   50-200 MB  15-30 MB
App startup         500ms-2s    500ms-2s   <100ms
Window animation    CPU tween   CPU tween  GPU shader
Transparency        CPU blend   CPU blend  GPU blend (free)
Blur                ~15% CPU    ~10% CPU   <0.1ms (GPU)
4K rendering        Lag visible  Lag visible Same <2ms
144Hz support       Stutters    OK         Smooth
Window count limit  ~20 (lag)   ~20 (lag)  ~100+ (GPU)
Total codebase      ~3M lines   ~5M lines  ~10K lines
```

### 12.2 Memory Comparison

```
SCENARIO: Desktop with file manager, terminal, browser,
          text editor, system monitor open.

GNOME:
  Shell (Mutter):     200 MB
  Nautilus (GTK):     80 MB
  Terminal (VTE):     50 MB
  Firefox:            500 MB
  gedit (GTK):        60 MB
  System Monitor:     40 MB
  TOTAL:              ~930 MB

KDE PLASMA:
  Plasmashell:        180 MB
  Dolphin (Qt):       70 MB
  Konsole (Qt):       40 MB
  Firefox:            500 MB
  Kate (Qt):          50 MB
  System Monitor:     35 MB
  TOTAL:              ~875 MB

OCTOSHELL:
  OctoShell:          40 MB
  OctoFlow runtime:   20 MB (shared)
  Files (.flow):      20 MB
  Terminal (.flow):   15 MB
  OctoView (.flow):   30 MB (browser engine much smaller)
  Notes (.flow):      15 MB
  Monitor (.flow):    15 MB
  TOTAL:              ~155 MB

  6x less RAM than GNOME. Same functionality.
```

### 12.3 Why the Difference

```
GNOME/KDE OVERHEAD:
  GTK4 library:      ~30 MB loaded per app
  Qt6 library:       ~40 MB loaded per app
  D-Bus daemon:      ~10 MB
  PulseAudio:        ~15 MB
  GVFS:              ~20 MB
  Tracker (indexer):  ~30 MB
  Background services: ~50 MB

  Each app loads its own copy of the toolkit library.
  5 GTK apps = 5 x 30 MB = 150 MB just for GTK overhead.

OCTOSHELL EFFICIENCY:
  OctoFlow runtime: loaded ONCE, shared by ALL apps.
  ext.ui: loaded ONCE, shared by ALL apps.
  No D-Bus (apps communicate via oct:// if needed).
  No separate indexer (file search is a .flow app using GPU).
  No background services (everything is event-driven .flow).

  The runtime is ~20 MB. Shared by every app. Period.
```

---

## 13. Raspberry Pi & ARM

### 13.1 Pi 5 Capabilities

```
RASPBERRY PI 5:
  CPU:    Broadcom BCM2712, Cortex-A76 (quad-core)
  GPU:    VideoCore VII (Vulkan 1.2 support)
  RAM:    4 GB or 8 GB
  Price:  $60-80

  Vulkan 1.2 = OctoFlow runs natively.
  SPIR-V compute shaders on VideoCore VII = verified working.
```

### 13.2 Pi Performance Comparison

```
OPERATION              GNOME ON PI 5    OCTOSHELL ON PI 5
──────────────────────────────────────────────────────────
Desktop render         15-30 fps        60 fps
Window switching       500ms delay      <50ms
File manager           2s to open       <200ms
Terminal scroll        Visibly laggy    Smooth
Browser                Barely usable    Usable (OctoView)
Video playback         VLC struggles    GPU decode (smooth)
RAM (idle desktop)     ~600 MB          ~100 MB
Available for apps     ~1.4 GB (4GB Pi) ~3.9 GB (4GB Pi)
```

GNOME on Pi 5 uses ~600 MB just for the desktop, leaving only 1.4 GB for applications on a 4 GB Pi. OctoShell uses ~100 MB, leaving 3.9 GB. That's nearly 3x more available memory for actual work.

### 13.3 Pi Use Cases

```
TRADING KIOSK ($80):
  Pi 5 + OctoShell + trading terminal .flow app
  Fullscreen, 60fps, GPU-rendered candlestick charts
  Connected to momentum-fx.oct via oct:// (encrypted)

  Cost: $80 hardware
  Replaces: $500+ Windows PC running MT5

DIGITAL SIGNAGE ($80):
  Pi 5 + OctoShell + dashboard .flow app
  Display live data on office TV
  GPU-rendered charts, auto-updating

  Cost: $80 per display
  Replaces: $200+ signage players

EDUCATION ($80):
  Pi 5 + OctoShell + OctoFlow IDE
  Students learn GPU programming on $80 hardware
  Complete development environment

  Schools can deploy 30 OctoFlow workstations for $2,400
  vs 30 Windows PCs for $15,000+

HOME SERVER ($80):
  Pi 5 + OctoShell (headless mode) + OctoServe
  Run small LLMs on Pi's GPU
  Serve OctoView apps to local network

  $80 AI server for home automation, smart home, local LLM
```

---

## 14. OctoLinux — The Distribution

### 14.1 What OctoLinux Is

OctoLinux is not a new Linux distribution. It's a configuration of existing Linux with OctoShell as the desktop environment. Think "Ubuntu is Debian + GNOME" — OctoLinux is "Arch/Void + OctoShell."

### 14.2 Base System Options

```
OPTION A: ARCH-BASED (recommended)
  Arch Linux + OctoShell
  Rolling release. Latest Vulkan drivers. Large community.
  AUR for additional packages.

OPTION B: VOID-BASED (minimalist)
  Void Linux + OctoShell
  Independent, musl libc option, runit init.
  Smallest possible base system.

OPTION C: ALPINE-BASED (container/embedded)
  Alpine Linux + OctoShell
  musl libc, ~50 MB base, container-friendly.
  Best for Pi, kiosks, embedded.
```

### 14.3 Install Size

```
COMPONENT               SIZE
────────────────────────────────
Linux kernel            ~100 MB
Base system (Arch)      ~300 MB
Vulkan driver (mesa)    ~50 MB
OctoFlow runtime        ~20 MB
OctoShell               ~10 MB
Core apps               ~5 MB
Fonts + assets          ~30 MB
PipeWire (audio)        ~15 MB
NetworkManager          ~10 MB
Base utilities          ~45 MB
────────────────────────────────
TOTAL                   ~585 MB

COMPARISON:
  Ubuntu Desktop:     ~5,000 MB
  Fedora KDE:         ~4,000 MB
  Arch + GNOME:       ~2,000 MB
  Windows 11:        ~25,000 MB
  macOS:             ~15,000 MB

  OctoLinux:            ~585 MB  (8.5x smaller than Ubuntu)
```

### 14.4 First Boot Experience

```
1. BOOT (5 seconds)
   Linux kernel → systemd → OctoShell starts

2. WELCOME SCREEN (.flow app)
   "Welcome to OctoLinux"
   Language selection
   Keyboard layout
   Display configuration (resolution, scaling)
   Theme choice (dark / light / custom)

3. IDENTITY SETUP
   "Create your OctoFlow identity"
   $ octo identity create
   Public key generated (this is your permanent identity)
   Optional: register an .oct name

4. NETWORK
   WiFi selection (if applicable)

5. DESKTOP
   OctoShell desktop appears.
   App launcher shows 9 core apps.
   Welcome tour (.flow app) explains basics.

   Total time from power on to usable desktop: ~15 seconds.

   Ubuntu: ~45 seconds.
   Windows 11: ~60-120 seconds.
```

---

## 15. Security Model (Desktop Context)

### 15.1 App Permissions (Desktop Extensions to Annex O)

```
LOCAL .flow APPS (installed via OctoStore):
  GPU compute
  ext.ui rendering
  Read/write within app data directory
  Network: requires manifest declaration + first-run approval
  File system (outside app dir): requires user file picker
  Camera/mic: permission prompt
  Other apps' data: denied
  System modification: denied
  Background execution (unless declared + approved): denied

PROCESS ISOLATION:
  Each .flow app runs in a separate process (Tier 3 remote apps).
  Local trusted apps can share the OctoFlow runtime process.
  Crash in one app doesn't take down the desktop.

  OctoShell itself runs as the compositor process.
  If an app hangs: Super+Q kills it. Desktop stays responsive.
```

### 15.2 System Security

```
FULL DISK ENCRYPTION: Standard LUKS (Linux default)
SECURE BOOT: Signed kernel + signed OctoFlow binary
IDENTITY: GPU-generated Ed25519 keypair (stored encrypted)
UPDATES: Signed by OctoFlow release key, verified before install
FIREWALL: nftables configured by OctoShell settings panel
SSH: Optional, off by default
```

---

## 16. Implementation Roadmap

### 16.1 Build Phases

```
PHASE 1: WINDOWED MODE (Month 12-13)                    ~2,000 lines
  OctoShell runs as a window inside GNOME/KDE.
  Window management (floating only).
  Basic taskbar + app launcher.
  3 core apps: terminal, notes, monitor.
  → DEMO: "look at this desktop running inside GNOME"

PHASE 2: FULL COMPOSITOR (Month 13-14)                   ~3,000 lines
  Smithay-based Wayland compositor.
  DRM/KMS direct rendering.
  Wayland app support (Firefox, etc.).
  Tiling + floating window modes.
  All 9 core apps.
  → DEMO: boot to OctoShell, use as daily driver

PHASE 3: POLISH (Month 14-15)                           ~2,000 lines
  Multi-monitor support.
  Fractional scaling.
  Display settings GUI.
  Theme system finalized.
  Live wallpapers.
  Screen lock.
  → RELEASE: OctoShell 1.0

PHASE 4: OCTOLINUX (Month 15-18)                        ~1,000 lines
  Installer ISO generation.
  Arch-based configuration.
  Hardware compatibility testing (Intel, AMD, NVIDIA, Pi 5).
  Welcome wizard.
  Documentation.
  → RELEASE: OctoLinux 1.0 ISO

TOTAL IMPLEMENTATION: ~8,000 lines
  (Shell: ~3,000, Compositor: ~2,500, Core apps: ~1,700, Installer: ~800)

COMPARISON:
  GNOME Shell:     ~300,000 lines (JavaScript + C)
  KDE Plasma:      ~500,000 lines (C++ + QML)
  Sway:            ~50,000 lines (C)
  Hyprland:        ~80,000 lines (C++)
  OctoShell:       ~8,000 lines (.flow + Rust)
```

### 16.2 Dependencies on Other OctoFlow Components

```
COMPONENT              REQUIRED FOR OCTOSHELL     STATUS
───────────────────────────────────────────────────────────
OctoFlow compiler      Everything                  Done (149 tests)
ext.ui core            All UI rendering            Phase 14-15
ext.ui.chart           Monitor, dashboards         Phase 16
ext.media              Player, Gallery             Phase 17
OctoView               Browser app                 Phase 18
ext.net                OctoStore, oct:// apps      Phase 10-11
ext.crypto             Identity, signing           Phase 6-7
std.os                 Files, Terminal, Monitor     New (file system + process API)
Smithay integration    Wayland compositor           New
```

### 16.3 Critical Path

```
ext.ui MUST exist before OctoShell.
OctoShell IS ext.ui used as a compositor.

The build order:
  1. ext.ui core (Phase 14-15): windows, text, rectangles, events
  2. ext.ui widgets (Phase 15): buttons, inputs, sliders, layout
  3. ext.ui.chart (Phase 16): line, bar, candlestick
  4. OctoShell windowed mode: WM + taskbar + 3 apps
  5. OctoShell compositor: Smithay + DRM/KMS
  6. Core apps: remaining 6 apps
  7. OctoLinux: installer + configuration
```

---

## 17. The Growth Narrative

### 17.1 The Viral Moments

```
MONTH 3:   "5 lines of OctoFlow = cinematic photo grading"
           Reddit r/programming: 500 upvotes
           → Creators install OctoFlow

MONTH 8:   "Run Llama on your AMD GPU. Finally."
           Reddit r/LocalLLaMA: 2,000 upvotes
           → ML developers install OctoFlow

MONTH 12:  "1 million data points at 60fps in a chart"
           HN: 300 points
           → Data engineers notice OctoFlow

MONTH 15:  "My entire Linux desktop is 1,700 lines of code"
           HN #1: 1,000+ points
           "OctoShell uses 155 MB RAM where GNOME uses 930 MB"
           Reddit r/linux: 5,000 upvotes
           → Linux community adopts OctoShell

           Phoronix benchmark: "OctoShell outperforms GNOME
           and KDE on every metric we tested"

           Pi community: "Finally, a usable desktop on Pi 5"

MONTH 18:  "OctoLinux: The GPU-First Operating System"
           Not a new OS. Just Linux that uses your GPU properly.
           585 MB install. 15 second boot. 60fps always.
           → Mainstream tech press coverage
```

### 17.2 The Ecosystem Flywheel

```
OctoMedia ships (Month 3)
  → Creators discover OctoFlow
    → They want a GUI for their tools
      → ext.ui ships (Month 12)
        → Developers build .flow apps
          → Apps need a desktop
            → OctoShell ships (Month 15)
              → Users want more apps
                → OctoStore grows
                  → More developers build apps
                    → OctoShell becomes daily-driver quality
                      → OctoLinux ships (Month 18)
                        → Self-sustaining ecosystem

Each product creates demand for the next.
OctoShell is where the flywheel reaches escape velocity.
Because once you have a desktop, you have an ECOSYSTEM.
```

### 17.3 The End State

```
ONE LANGUAGE:
  Every app, every widget, every animation, every shader,
  every filter, every ML model, every network connection,
  every chart, every terminal character — written in .flow.

ONE RUNTIME:
  OctoFlow compiler → SPIR-V → Vulkan → GPU.
  20 MB shared across all apps.
  No GTK, no Qt, no Electron, no Chromium.

ONE GPU:
  Desktop rendering, video processing, ML inference,
  cryptography, chart visualization, terminal rendering —
  all on the same GPU, same Vulkan instance, same frame.

ONE IDENTITY:
  oct:ed25519:your_key
  Your apps, your data, your name — all one cryptographic identity.
  Portable across devices. Self-sovereign.

THAT'S OCTOSHELL.
  The desktop where your GPU finally does its job.
```

---

## Summary

OctoShell is a GPU-first desktop environment that runs on Linux, replacing GNOME/KDE with a Vulkan-rendered compositor where every application is a `.flow` file sharing a single OctoFlow runtime. Nine core apps in ~1,700 lines of code replace hundreds of thousands of lines of traditional desktop software, while using 6x less RAM and delivering consistent 60fps performance.

Key differentiators: single Vulkan render pass for entire desktop (<2ms per frame), shared runtime eliminates per-app overhead (no Electron duplication), GPU-accelerated effects are free (blur, transparency, animation), and the same `.flow` language powers everything from the window manager to the terminal to the browser to trading terminals.

OctoLinux packages this as a ~585 MB Linux configuration — 8.5x smaller than Ubuntu — with a 15-second boot time and support for hardware from Raspberry Pi 5 ($80) to high-end workstations with 240Hz displays. OctoShell is where the OctoFlow ecosystem reaches critical mass: once developers have a desktop, they have a platform.

---

*This annex specifies OctoShell, the GPU-first desktop environment built on OctoFlow. Implementation begins at Month 12 after ext.ui core is complete, with OctoLinux shipping at Month 18. The build order follows the ext.ui → OctoShell → OctoLinux dependency chain established in the platform roadmap.*
