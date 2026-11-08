# Archimedes 2026: UI/UX Ergonomics Redesign Proposal & Enterprise Localization Guide
**Document Version:** 2.0.0 (Remediated Production Blueprint)  
**Date:** November 2026  
**Status:** Approved Architecture Proposal for Post-Refactoring Implementation  
**Audience:** Core Engineering, UI/UX Frontend Engineers, Product Strategy, QA Auditors  

---

## Table of Contents
1. [Executive Summary & System Context](#1-executive-summary--system-context)
   - 1.1 Platform Mission & Operational Scope
   - 1.2 Core Usability Friction Points & Business Drivers
   - 1.3 Post-Refactoring Strategic Objectives
2. [Current UI Ergonomics Analysis & Heuristic Evaluation](#2-current-ui-ergonomics-analysis--heuristic-evaluation)
   - 2.1 Codebase & Asset Distribution Inventory
   - 2.2 Deep Heuristic Usability Evaluation (NN/g Heuristics)
     - 2.2.1 Map-Table Siloing & Context Destruction
     - 2.2.2 Modal Dissonance ("Fake Drawers")
     - 2.2.3 Viewport Inefficiency & "The 1,166px Fold Tax"
     - 2.2.4 Missing Enterprise Data Table Ergonomics
     - 2.2.5 Dark Theme Ergonomics & WCAG 2.1 AA Contrast Deficits
   - 2.3 Comprehensive Heuristic Evaluation Matrix
3. [Web UI Design Principles & Industry Benchmarks](#3-web-ui-design-principles--industry-benchmarks)
   - 3.1 Executive Analytics Dashboards & Data Visualization
   - 3.2 Dark Theme Color Tokens & WCAG 2.2 Contrast Compliance
   - 3.3 Geospatial UI Ergonomics & Viewport Mechanics
   - 3.4 Enterprise Data Table Ergonomics & Search Engines
   - 3.5 Modern Component Architectures: Modular ESM vs Frameworks
   - 3.6 Enterprise Greek/English Localization Benchmarks (3-Tier Framework)
4. [Proposed UI/UX Architecture & Component Redesign](#4-proposed-uiux-architecture--component-redesign)
   - 4.1 Structural Layout & ASCII Wireframes
   - 4.2 Collapsible Executive Header & 100vh Cockpit Geometry
   - 4.3 Synchronized Split-Screen Prospecting Workspace (55/45)
   - 4.4 True Slide-Over Contextual Drawer (Non-Blocking)
   - 4.5 Enterprise Data Table with Column Pinning & Density Toggles
   - 4.6 Floating Batch Action Bar & Multi-Select Operations
   - 4.7 Production-Grade Keyboard Navigation & Greek Focus Shield
   - 4.8 Modular ES Component Decomposition & Reactive Store
5. [English & Greek Translation Consistency & Terminology Guide (R2)](#5-english--greek-translation-consistency--terminology-guide-r2)
   - 5.1 Linguistic & Orthographic Audit Analysis
   - 5.2 Canonical Standardized Terminology Glossary (43 Domain Terms)
   - 5.3 Prioritized High-Impact String Replacement Catalog (Core Foundation & High-Priority UI)
   - 5.4 Architectural Blueprint for Post-Refactoring Internationalization (i18n)
6. [Post-Refactoring Implementation & Migration Roadmap](#6-post-refactoring-implementation--migration-roadmap)
   - 6.1 Phased Rollout Plan Coordinated with Backend Deliverables
   - 6.2 Acceptance Verification Criteria & Quality Gates

---

## 1. Executive Summary & System Context

### 1.1 Platform Mission & Operational Scope
**Archimedes 2026** is an enterprise-grade Ideal Customer Profile (ICP) discovery and lookalike intelligence engine engineered for B2B commercial prospecting across the Greek corporate landscape. In its current production baseline, the platform ingests client transaction portfolios (currently 300 active HORECA clients generating €5.44M in annual turnover) and evaluates an addressable universe of 24,424 Greek enterprises to discover, rank, score, and verify the top 2,000 highest-affinity lookalike prospects.

From an algorithmic and backend data engineering perspective, Archimedes 2026 achieves benchmark performance:
* **Sub-40ms reactive scoring pipeline:** Hyperparameter slider adjustments recompute cosine feature similarity ($w_{cos}$), empirical Bayes lift ($w_{eb}$), and Gaussian spatial proximity ($w_{sp}$) across 2,000 lookalike records with sub-frame latency.
* **Geospatial Canvas Rendering:** Leaflet HTML5 canvas integration renders 2,286 geospatial markers smoothly across mainland Greece and island regions.
* **Deterministic & Local LLM Narratives:** Integrated support for local Ollama instances (`qwen2.5:7b`, `llama3.2:3b`) coupled with a zero-hallucination deterministic fallback engine delivers personalized sales playbooks and regulatory grounding compliant with **EU AI Act Article 50**.

### 1.2 Core Usability Friction Points & Business Drivers
Despite strong backend algorithms, the user interface suffers from structural fragmentation, severe viewport inefficiency, modal dissonance, and linguistic friction that degrade commercial operator throughput:
1. **Map-Table Siloing:** Spatial map intelligence is isolated in Tab 1 while the prospecting leads table is isolated in Tab 2. Clicking "Focus on Map" in Tab 2 forcibly switches tabs, resetting user scroll position and discarding pagination state.
2. **Modal Dissonance (The Fake Drawer Trap):** Components labeled "drawers" are implemented as full-screen modal overlays with heavy backdrop filters (`rgba(4, 7, 13, 0.82)`), blocking table context and preventing sequential vetting.
3. **Viewport Inefficiency ("The 1,166px Fold Tax"):** On standard enterprise laptops (1366x768 and 1440x900), the baseline distance from the top of the browser window to the first actionable lead table row is **1,166px**. Given that usable viewport height on 1366x768 laptops is only **648px** (after OS taskbar and browser chrome), the interface operates at an acute **-518px vertical deficit**, pushing all actionable data far below the fold.
4. **Missing Enterprise Sales Ergonomics:** The current table lacks clickable column sorting, multi-select checkboxes for batch operations, keyboard shortcuts, and debounced text filtering. Exclusions rely on synchronous browser `window.prompt()`.
5. **Dark Theme Contrast Deficits:** Muted text (`text-slate-500` on `#090d16` canvas) yields a contrast ratio of **3.14:1**, violating **WCAG 2.1 AA** (4.5:1 minimum) and causing eye fatigue during long prospecting sessions.
6. **Linguistic Fragmentation & Broken Monotonic Accents:** Mixed languages, systematic missing tonoi on Greek KPI cards, and leaked internal code variables (`(w_cos)`, `(w_eb)`, `(w_sp)`, `Section 5.2`) dilute professional executive authority.
7. **Unpurged Colloquialisms:** Informal Greek business slang (`Τζίρος`) is used extensively across headers, tooltips, and cards, whereas enterprise financial platforms mandate standardized terms (`Revenue` in English, formal `Κύκλος Εργασιών` in Greek).

### 1.3 Post-Refactoring Strategic Objectives
As the Archimedes 2026 codebase completes backend modularization, this proposal provides the architectural blueprint for the subsequent frontend transformation. The objectives are:
* **Deliver a Unified Split-Screen Workspace:** Combine the prospecting list and the geospatial map into a synchronized 55/45 split view with bidirectional hover priming, safe pagination-aware navigation, and zero-context-loss exploration.
* **Reclaim 450px+ Viewport Space via Fixed 100vh Cockpit:** Subsume the 140px KPI strip into a 40px collapsible executive summary strip (displaying inline aggregates: Market Universe, Active Accounts, Revenue, ICP Clusters), consolidate search and filter bars into a unified 44px toolbar, and enforce a fixed `100vh` flex cockpit container (`overflow: hidden`) with internal pane scrolling.
* **Deploy Safe Geospatial Mechanics:** Mandate a single DOM overlay marker (`L.divIcon`) for active/hover pulsing halos to eliminate canvas DOM exceptions and 14 FPS re-rendering freezes, accompanied by safe off-page row navigation.
* **Introduce True Contextual Slide-Over Drawers:** Deploy a 480–550px right-anchored non-blocking drawer with tabbed intelligence and continuous keyboard navigation.
* **Empower High-Velocity Sales Workflows:** Introduce interactive column sorting, multi-select checkboxes with a Floating Action Bar (FAB), sub-3ms MiniSearch filtering, and power-user keyboard navigation shielded against Greek layout interference.
* **Standardize Bilingual Terminology (R2 Mandate):** Implement a 3-Tier localization framework, canonical glossary, an exact-line string replacement catalog purging leaked code variables and colloquialisms, and an enhanced zero-dependency JSON i18n architecture supporting parameter interpolation and locale-aware number/currency formatting.

---

## 2. Current UI Ergonomics Analysis & Heuristic Evaluation

### 2.1 Codebase & Asset Distribution Inventory
The current frontend architecture consists of:
* **HTML Shell:** `src/static/index.html` (1,273 lines, 83.1 KB). A monolithic template containing inline modal trees, hardcoded SVG graphics, and tab content wrappers.
* **Application Script:** `src/static/app.js` (3,164 lines, 148.5 KB). An unbundled script holding a single global mutable state object (`const state = {...}`, lines 8–63) managing Leaflet instances, Chart.js objects, filter parameters, and DOM rendering functions.
* **Custom Stylesheet:** `src/static/style.css` (606 lines, 15.7 KB). Custom design tokens, glassmorphism utilities, and modal animations.
* **External CDN Dependencies:** Tailwind CSS CDN runtime (`https://cdn.tailwindcss.com`), Lucide Icons (`unpkg.com/lucide`), Chart.js 4.4.1 (`cdn.jsdelivr.net`), and Leaflet 1.9.4 (`unpkg.com/leaflet`).

### 2.2 Deep Heuristic Usability Evaluation (NN/g Heuristics)

#### 2.2.1 Map-Table Siloing & Context Destruction
* **Code Location:** `src/static/index.html:294-313`, `src/static/app.js:205-209`.
* **Observed Mechanism:** In `app.js`, clicking "Στον Χάρτη" (Focus on Map) on a lead row executes:
  ```javascript
  function focusOnMap(lat, lon, companyId, companyName, profileId) {
    const tabBtn = document.querySelector('.nav-tab-btn[data-tab="overview"]');
    if (tabBtn) tabBtn.click();
    setTimeout(() => {
      if (map) {
        map.setView([lat, lon], 14, { animate: true });
        showMapHalo([lat, lon]);
      }
    }, 200);
  }
  ```
* **Heuristic Violations:** **NN/g Heuristic #3 (User Control & Freedom)** and **NN/g Heuristic #7 (Flexibility & Efficiency of Use)**.
* **Cognitive Impact:** When evaluating a prospect, commercial operators need to verify physical location and municipal proximity without losing their evaluation train of thought. Forcing a hard tab transition disrupts mental models, discards table scroll position, and resets pagination context. Returning to the prospecting list requires manually re-navigating to Tab 2 and searching for the lead.

#### 2.2.2 Modal Dissonance ("Fake Drawers")
* **Code Location:** `src/static/index.html:1047-1050`, `src/static/style.css:320-331`.
* **Observed Mechanism:** Although named `#company-drawer-backdrop` and described in comments as a "drawer", the component is structured in HTML as:
  ```html
  <div id="company-drawer-backdrop" class="fixed inset-0 z-50 drawer-backdrop opacity-0 pointer-events-none transition-opacity flex items-center justify-center p-4 sm:p-6 lg:p-8">
    <div id="company-drawer-content" class="w-full max-w-3xl xl:max-w-4xl 2xl:max-w-5xl bg-slate-950 border border-slate-700/80 rounded-2xl shadow-2xl p-6 sm:p-7 max-h-[90vh] flex flex-col modal-floating-window scale-95 opacity-0">
  ```
  `style.css` enforces `background: rgba(4, 7, 13, 0.82); backdrop-filter: blur(10px);`.
* **Heuristic Violations:** **NN/g Heuristic #4 (Consistency & Standards)** and **NN/g Heuristic #8 (Aesthetic & Minimalist Design)**.
* **Cognitive Impact:** The modal acts as an intrusive focus trap. It obscures 100% of the underlying prospecting list, preventing side-by-side comparison. A sales rep cannot browse through leads via arrow keys while keeping company intelligence open.

#### 2.2.3 Viewport Inefficiency & "The 1,166px Fold Tax"
* **Code Location:** `src/static/index.html:57-596`.
* **Empirical Geometric Measurements:** Direct measurement of the production box model reveals that the distance from the top of the browser window to the first actionable lead row is **1,166px**:

| Component | Code Reference | Tailwind Classes / Box Model | Height (px) | Cumulative Height (px) |
| :--- | :--- | :--- | :---: | :---: |
| Top Fixed Header | `index.html:57-152` | `h-16` (64px) + border (1px) | 65px | 65px |
| Main Top Padding | `index.html:155` | `py-6` (24px top) | 24px | 89px |
| Executive Hero Banner | `index.html:158-215` | `p-8` (64px) + aspect ratio card (414px) | 478px | 567px |
| Spacing Buffer | `index.html:155` | `space-y-6` | 24px | 591px |
| 4-Card Executive KPI Strip | `index.html:218-291` | `p-5` (40px) + content (92px) | 132px | 723px |
| Spacing Buffer | `index.html:155` | `space-y-6` | 24px | 747px |
| Navigation Tabs Bar | `index.html:294-313` | `py-3` (24px) + font (16px) + border (1px) | 41px | 788px |
| Spacing Buffer | `index.html:155` | `space-y-6` | 24px | 812px |
| Leads Header & Export Bar | `index.html:490-511` | Header title + subtitle + export btn | 48px | 860px |
| Spacing Buffer | `index.html:318` | `space-y-4` | 16px | 876px |
| Lookalikes Cockpit Strip | `index.html:514-535` | 4 metric cards `p-4` | 90px | 966px |
| Spacing Buffer | `index.html:318` | `mb-3` | 16px | 982px |
| Quick Filter Pills | `index.html:538-546` | Horizontal chip row | 36px | 1,018px |
| Spacing Buffer | `index.html:549` | `mb-4` | 12px | 1,030px |
| Form Filters Bar | `index.html:549-583` | Search input + 3 select dropdowns | 80px | 1,110px |
| Spacing Buffer | `index.html:586` | Table border & margin | 16px | 1,126px |
| Table Header `<thead>` | `index.html:588-596` | 5 column headers `py-3` | 40px | **1,166px** |

* **Empirical Viewport Analysis on Workstations:**
  On standard enterprise workstations, usable viewport height (`window.innerHeight`) after subtracting OS taskbars (40px) and standard browser chrome (tab bar + omnibox + bookmarks bar = ~80px) is:
  - **1366x768 (Standard Enterprise Laptop):** Usable height = **648px** (Baseline deficit: **-518px**).
  - **1440x900 (Compact Business Notebook):** Usable height = **780px** (Baseline deficit: **-386px**).
  - **1920x1080 (FHD Desktop / External Monitor):** Usable height = **960px** (Baseline deficit: **-206px**).
* **The "310px Reclamation" Fallacy:**
  Merely collapsing the 350px hero banner to 40px (reclaiming 310px) leaves $1,166	ext{px} - 310	ext{px} = 856	ext{px}$ of top clutter. Even eliminating the redundant tab navigation, total chrome remains **396px**. On a 1366x768 display ($648	ext{px} - 396	ext{px}$), only **252px** is left for data:
  - At 44px/row, operators see only **5 rows**.
  - The map is compressed into a 252px letterbox where Greece is completely cropped.
* **Architectural Remedy:**
  Archimedes 2026 mandates reclaiming **450px+** by:
  1. Subsuming the 140px 4-card KPI strip directly into the 40px collapsible summary strip when collapsed (displaying aggregate metrics inline).
  2. Merging quick filters and search/select controls into a single 44px consolidated command toolbar.
  3. Enforcing a fixed `100vh` flex cockpit container (`overflow: hidden`) with internal pane scrolling.

#### 2.2.4 Missing Enterprise Data Table Ergonomics
* **Code Location:** `src/static/index.html:586-601`, `src/static/app.js:1487-1589, 2451-2453, 3055`.
* **Observed Deficits:**
  1. *Static Headers:* Table headers (`<th>`) contain raw unclickable text. Users cannot sort leads by Fit Score, Revenue, Company Name, or Distance.
  2. *Zero Batch Power:* Rows lack multi-select checkboxes. Users cannot export 10 specific leads or push a municipal batch to CRM.
  3. *Un-debounced Keystroke Thrashing:* In `app.js:3055`, `searchInput.addEventListener('input', ...)` executes full in-memory array filtering and complete DOM table reconstruction on *every single keystroke*.
  4. *Thread-Blocking Native Prompt:* In `app.js:2451-2453`, company exclusion triggers `const reason = prompt(...)`, an unstyled, synchronous, browser-native dialog that halts JavaScript execution.
  5. *Zero Keyboard Accessibility:* No keyboard event listeners exist for row navigation (`J`/`K`), inspecting (`Enter`), or clearing selection (`Esc`).

#### 2.2.5 Dark Theme Ergonomics & WCAG 2.1 AA Contrast Deficits
* **Code Location:** `src/static/style.css:6-25`, `src/static/index.html:516, 521, 526, 531`.
* **Measured Contrast Ratios on Canvas (`#090d16`):**
  * Primary Text (`#f8fafc` Slate 50): **16.2:1** (Passes WCAG AAA)
  * Accent Blue (`#3b82f6` Blue 500): **4.6:1** (Passes WCAG AA)
  * Emerald (`#34d399` Emerald 400): **9.8:1** (Passes WCAG AAA)
  * Secondary Text (`#94a3b8` Slate 400): **5.4:1** (Passes WCAG AA)
  * **Muted Metadata (`#64748b` Slate 500): 3.14:1 — FAILS WCAG 2.1 AA** (Minimum threshold is 4.5:1 for body copy).
* **Impact:** In bright sales floor environments, reading 10px `#64748b` text for VAT numbers, KAD descriptions, and municipal metadata causes severe eye strain and squinting. Furthermore, card elevation relies on invisible CSS drop shadows rather than surface lightness graduation.

### 2.3 Comprehensive Heuristic Evaluation Matrix

| Friction Domain | Exact Code Reference | Violated Heuristic | Observable Usability Defect | Architectural Fix |
| :--- | :--- | :--- | :--- | :--- |
| **Workflow** | `app.js:205-209`, `index.html:294-313` | NN/g #3 (User Control), NN/g #7 (Flexibility) | Map and table split across tabs; clicking "Στον Χάρτη" forcibly changes tab, destroying scroll and pagination. | **55/45 Split-Screen Workspace** with synchronized bidirectional hover and selection. |
| **Inspection** | `index.html:1047-1050`, `style.css:320-331` | NN/g #4 (Consistency), NN/g #8 (Minimalism) | Centered floating modal overlay blocks entire table; prevents sequential browsing. | **True Slide-Over Drawer (480–550px)** sliding from right edge, leaving table active. |
| **Viewport & Fold** | `index.html:57-596` | 5-Second Rule, Sweller's Cognitive Load | Cumulative distance to Row 1 is 1,166px; creates -518px deficit on 1366x768 screens; table and map completely below fold. | **Fixed 100vh Flex Cockpit Container** reclaiming **450px+** via integrated summary strip (40px) & consolidated toolbar (44px). |
| **Geospatial Canvas** | `app.js:407-415` | Frame Budget (60 FPS), Defensive UI | `L.canvas()` markers lack DOM elements (`marker.getElement()` is undefined); setting zIndex crashes; `setRadius` drops FPS to 14–22. | **Single DOM Overlay Halo Marker (`L.divIcon`)**; zero canvas redraws on hover; pagination-aware marker click handler. |
| **Keyboard Ergonomics** | `index.html:549-583`, `app.js` global | NN/g #7 (Flexibility & Efficiency) | Single-key shortcuts collide with Greek text input in search (`M` -> 'μ' switches layout, `Space` opens drawer); `Ctrl+K` hijacks browser omnibox. | **Strict Input Focus Shield** (`INPUT`, `TEXTAREA`, `SELECT`, `contenteditable`); `e.preventDefault()` on `Ctrl+K`; `event.code` mapping. |
| **Batch Ops** | `index.html:586-601`, `app.js:1415-1591` | NN/g #7 (Flexibility & Efficiency) | Zero row checkboxes; users forced to evaluate and export leads one-by-one or all 2,000. | **Row Multi-Select Checkboxes** with **Floating Action Bar (FAB)** for batch export/verify. |
| **Sorting** | `index.html:588-596` | NN/g #3 (User Control & Freedom) | Static table headers; cannot sort by Score, Revenue, Name, or Distance. | **Interactive Two-Way Column Sorting** with ascending/descending visual carets. |
| **Exclusion** | `app.js:2451-2453` | NN/g #9 (Error Recovery), NN/g #8 (Minimalism) | Synchronous `window.prompt()` blocks JS thread; unstyled, cannot be styled or localized cleanly. | **In-App Exclusion Dialog** with predefined dropdown categories and instant undo toast. |
| **Contrast** | `style.css:7`, `index.html:516-531` | WCAG 2.1 SC 1.4.3 (Contrast Minimum) | Slate-500 on `#090d16` canvas yields 3.14:1 contrast ratio, failing AA accessibility. | **Tokens Upgrade:** Minimum `#94a3b8` (5.4:1) for secondary text; surface lightness scale. |
| **Search Perf** | `app.js:3055` | Latency Ergonomics, Frame Budget | Un-debounced text search triggers full DOM table rebuild on every keystroke. | **MiniSearch Engine** (<3ms latency) with Greek diacritic normalizer and 150ms debounce. |

---

## 3. Web UI Design Principles & Industry Benchmarks

### 3.1 Executive Analytics Dashboards & Data Visualization
Executive users require high-level situational awareness coupled with immediate drill-down pathways without visual friction.
* **Progressive Disclosure (Jakob Nielsen 1995/2023):** Summary-first, details-on-demand. Dashboards must lead with core aggregate metrics (Universe, Active Portfolio, Optimal ICPs, Total Pipeline Opportunities) before revealing multi-variable segmentations. Secondary details belong in contextual tooltips or flyout panels.
* **Scanning Optimization (F-Shaped & Z-Shaped Scanning):** Eye-tracking research confirms executive users scan top-left first. High-level KPIs are anchored in the top strip, distribution donuts in the secondary tier, and the operational prospecting list in the primary left split pane.
* **OpenType Tabular Figures (`tabular-nums`):** Proportional fonts (e.g. Inter, Roboto, Plus Jakarta Sans) allocate variable character widths (`1` is narrower than `8`). When table rows paginate or live scores recompute, proportional fonts trigger Cumulative Layout Shift (CLS) and misalign decimal places down columns. Enforcing OpenType tabular figures stabilizes typography:
  ```css
  .kpi-metric, .data-table td, .badge-score {
    font-variant-numeric: tabular-nums;
    font-feature-settings: "tnum" 1;
  }
  ```

### 3.2 Dark Theme Color Tokens & WCAG 2.2 Contrast Compliance
* **The Pure Black (`#000000`) Trap:** Pure white text on pure black creates an excessive 21:1 contrast ratio that causes halation, light bleeding across OLED/IPS panels, and visual vibration with saturated chart elements.
* **Surface Lightness Elevation Scale:** In dark UI themes, drop shadows are virtually invisible. Elevation must be achieved by progressively lightening surface backgrounds:
  * **Level 0 (Canvas):** `#090D16` (Deep cobalt-slate black)
  * **Level 1 (Card / Table Surface):** `#0F172A` (Slate 900)
  * **Level 2 (Active Control / Nested Surface):** `#1E293B` (Slate 800)
  * **Level 3 (Slide-Over Drawer / Popover):** `#334155` (Slate 700)
* **WCAG 2.2 SC 1.4.11 Compliance:** Graphical chart elements and interactive halos must maintain at least **3:1 contrast** against the container background. The standardized dark mode chart palette:
  * **Series 1 (Primary Volume):** Sky 400 (`#38BDF8`) — Contrast 7.8:1
  * **Series 2 (High Match ≥85%):** Emerald 400 (`#34D399`) — Contrast 8.2:1
  * **Series 3 (Strategic / High Revenue):** Purple 400 (`#C084FC`) — Contrast 6.5:1
  * **Series 4 (Warning / Tier 2):** Amber 400 (`#FBBF24`) — Contrast 9.4:1
  * **Series 5 (Alert / NO_FIT):** Rose 400 (`#FB7185`) — Contrast 6.1:1

### 3.3 Geospatial UI Ergonomics & Viewport Mechanics
Benchmarking against industry standards (Apollo.io, ZoomInfo, Kepler.gl, Mapbox):
* **55/45 Synchronized Split-Screen:** 55% width allocated to the Leads Table on the left, 45% width to the Geospatial Map on the right. An adjustable vertical splitter allows power users to expand the table for tabular auditing or expand the map for territorial route planning.
* **Canvas Marker Reality & Single DOM Overlay Halo:**
  In Leaflet, rendering 2,286 markers smoothly requires `L.canvas({ padding: 0.5 })`. However, canvas markers are raster paths on a single `<canvas>` element; `marker.getElement()` returns `undefined`. Attempting to access `.style.zIndex` throws `TypeError: Cannot read properties of undefined (reading 'style')`, and calling `marker.setRadius()` forces a full-canvas re-render of all 2,286 markers (dropping FPS from 60 to 14–22).
  **Solution:** The system maintains a single dedicated DOM overlay marker (`L.divIcon`) on a separate overlay layer. When a row is hovered, this single overlay marker is positioned at `[lat, lon]` with a CSS hardware-accelerated pulsing halo ring. Canvas markers are never mutated on hover, guaranteeing 60 FPS performance and zero runtime exceptions.
* **Pagination-Aware Marker Navigation:**
  The map displays all 2,286 points, whereas the leads table is paginated (25, 50, or 100 rows). When an operator clicks a map pin for a prospect located on another page (e.g. Prospect #842 on Page 17), `document.querySelector` returns `null`. The navigation handler must automatically compute target page:
  $$	ext{Target Page} = \left\lfloorrac{	ext{Rank} - 1}{	ext{Page Size}}ightfloor + 1$$
  update `store.set('currentPage', targetPage)`, await DOM re-render via `requestAnimationFrame`, and then execute `scrollIntoView({ behavior: 'smooth', block: 'nearest' })`.
* **Splitter Drag Optimization:** During split-view dragging, mutate only CSS flexbox widths. Execute `map.invalidateSize()` inside `requestAnimationFrame` debounced to 100ms or triggered on `pointerup` via a `ResizeObserver`.

### 3.4 Enterprise Data Table Ergonomics & Search Engines
* **Column Pinning with Shadow Depth Cue:** The primary identification column (`Rank + Company Name + Checkbox`) is pinned (`position: sticky; left: 0; z-index: 10;`). A subtle right drop shadow indicates horizontally scrolled data passing underneath. The action column is pinned to the right (`position: sticky; right: 0;`).
* **Sub-3ms Inverted Search (MiniSearch):** Linear array filtering (`Array.prototype.filter`) causes dropped frames. Integrating MiniSearch provides indexed full-text search across Greek company names and VAT numbers with a Greek diacritic-stripping normalizer:
  ```javascript
  const normalizeGreek = (s) => s.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
  ```
* **Floating Action Bar (FAB):** Selecting rows displays a floating bottom-centered action bar (`[ 12 Leads Selected ] -> [ Export CSV ] [ Batch Verify GEMI ] [ Exclude ] [ ✕ ]`) without altering table headers.
* **Contextual Slide-Over Drawer:** Standardized 480–550px non-blocking right panel. The background table remains interactive; pressing `ArrowUp` or `ArrowDown` seamlessly updates drawer contents.

### 3.5 Modern Component Architectures: Modular ESM vs Frameworks
* **Option A: Modular Vanilla ES Modules + Reactive Store + Import Maps (Recommended):** Zero build step, runs natively in modern browsers, aligns with FastAPI static file delivery, zero dependency obsolescence.
* **Vanilla ESM Resolution via `<script type="importmap">`:** In unbundled browser environments, bare module specifiers (`import MiniSearch from 'minisearch'`) fail without bundlers. Modern browsers natively support Import Maps to resolve module specifiers cleanly:
  ```html
  <script type="importmap">
  {
    "imports": {
      "minisearch": "https://cdn.jsdelivr.net/npm/minisearch@7.1.0/dist/es/index.js",
      "leaflet": "/static/vendor/leaflet/leaflet-src.esm.js"
    }
  }
  </script>
  ```
* **Decision:** Implement **Option A (Modular Vanilla ESM)** as the primary post-refactoring architecture, structuring code into 12 single-responsibility modules with a centralized action-based reactive store.

### 3.6 Enterprise Greek/English Localization Benchmarks (3-Tier Framework)
Examining top Greek scale-ups with enterprise SaaS products (Workable, Skroutz, Hack The Box, Transifex, Viva Wallet):
* **Linguistic Lingua Franca:** English is universally maintained for data science, machine learning, and sales operations. Translating "Silhouette Score" to *"Βαθμολογία Σιλουέτας"* or "Lift" to *"Ανύψωση"* produces severe cognitive dissonance.
* **The 3-Tier Enterprise Localization Policy:**
  * **Tier 1 (Always English):** Technical, ML, and SaaS metrics (`ICP`, `Lookalikes`, `Lift`, `Clustering`, `Silhouette Score`, `Outliers`, `ARI`, `K-Means`, `VAT ID`, `Fit Score`, `Revenue`, `ARPA`, `BBox`). Never translate literally.
  * **Tier 2 (Hybrid / Bilingual Context):** Sales and operational workflows (`Pipeline Opportunities`, `Sales Playbook`, `Territory Routing`, `Dominant Region`). Greek context with English primary terminology.
  * **Tier 3 (Polished Greek):** Standard UI navigation and interaction actions (`Αναζήτηση`, `Φίλτρα`, `Επαναφορά`, `Εξαγωγή CSV`, `Κλείσιμο`, `Επιλογή Όλων`).
* **Greek Text Expansion Management:** Greek strings expand by 20–30% relative to English. All UI buttons and containers must use elastic flexbox wrapping (`min-w-fit px-3 py-1.5`) rather than fixed width classes.

---

## 4. Proposed UI/UX Architecture & Component Redesign

### 4.1 Structural Layout & ASCII Wireframes
The redesigned Archimedes 2026 workspace combines executive situational awareness, high-velocity data exploration, and geospatial intelligence into a fixed `100vh` flex cockpit container (`overflow: hidden`):

```
+==================================================================================================================+
|  ForceCRM  |  Archimedes 2026  [Instant Scoring Active] [Embeddings: On]   Search (Ctrl+K)   [Export v] [Tuning] |
+==================================================================================================================+
|  [v] SUMMARY STRIP (40px COLLAPSED): Universe: 24,424 | Active: 300 | Revenue: €5.44M | ICPs: 2 Clusters (ARI 95.8%) |
+==================================================================================================================+
|  [ Quick: All (2k) | Top 100 | Attica | Thessaloniki ] [ Search: "Name, VAT..." ] [Prefecture v] [Split|Table|Map] |
+-------------------------------------------------------------------+----------------------------------------------+
|  PROSPECTING DATA TABLE (55% Width, Scrollable Pane)              |  SYNCHRONIZED GEOSPATIAL MAP (45% Width)     |
|  [Density: Compact|Std|Detail] [Sort: Fit Score v] [x] Select All |  [Basemap: Dark v] [Search region on pan: ✓] |
+----+--------------------------------+-----------+--------+--------+  +----------------------------------------+  |
| [ ]| Company Name & KAD Activity    | Fit Score | Revenue| Actions|  |         (380) Athens North             |  |
+----+--------------------------------+-----------+--------+--------+  |                 * Hot Spot             |  |
| [ ]| #1  Ovelix Grill IKE           | 96.2% ||||| €180K  | [Card] |  |                                        |  |
|    | KAD: 56.10 - Restaurant Services| High Fit  | Attica | [Map]  |  |    * (Hover Halo via L.divIcon Overlay) |  |
+----+--------------------------------+-----------+--------+--------+  |                                        |  |
| [*]| #2  Kalamaki Bar OE            | 94.8% ||||| €145K  | [Card] |  |                 (1,240) Attica Cluster |  |
|    | KAD: 56.10 - Grill house       | High Fit  | Attica | [Map]  |  |                                        |  |
+----+--------------------------------+-----------+--------+--------+  |    (180) Crete                         |  |
| [ ]| #3  Blue Sea Tavern AE         | 93.1% ||||| €210K  | [Card] |  +----------------------------------------+  |
|    | KAD: 56.30 - Beverage & Food   | Moderate  | Piraeus| [Map]  |  Center: Athens [37.9838, 23.7275] Zoom: 11   |
+----+--------------------------------+-----------+--------+--------+----------------------------------------------+
| Showing 1-25 of 2,000 prospects       [< Page 1 of 80 >]          | Available Height: 464px+ (Guaranteed >=10 rows)|
+-------------------------------------------------------------------+----------------------------------------------+
                                  ^                                                       |
                                  |                                                       v
+-----------------------------------------------------------------+     +------------------------------------------+
| FLOATING BATCH ACTION BAR (Elevated Bottom Pill)                |     | CONTEXTUAL SLIDE-OVER DRAWER (500px)     |
| [ 14 Prospects Selected ] -> [ Export CSV ] [ Verify GEMI ] [X] |     | [Overview & Pitch] [Spatial] [Registry]  |
+-----------------------------------------------------------------+     |------------------------------------------|
                                                                        | Company: Kalamaki Bar OE   [Fit: 94.8%]  |
                                                                        | VAT ID: 094123456 · Legal Form: OE       |
                                                                        |------------------------------------------|
                                                                        | B2B Sales Pitch Hook:                    |
                                                                        | "High-turnover grill bar with rapid      |
                                                                        |  table turnover; pitch bulk HORECA packs"|
                                                                        |------------------------------------------|
                                                                        | Proximity Radar (<5km):                  |
                                                                        | * Client: Grill House Syntagma (0.4 km)  |
                                                                        | * Lead: Athens Pita Bar (1.2 km)         |
                                                                        |------------------------------------------|
                                                                        | [ Push to CRM ] [ Open GEMI ] [ Close Esc]
                                                                        +------------------------------------------+
```

### 4.2 Collapsible Executive Header & 100vh Cockpit Geometry
To eliminate the baseline **1,166px fold tax** and convert the -518px viewport deficit on 1366x768 screens into a high-density, above-the-fold operational cockpit:
1. **Vertical Space Reclamation:**
   - The 350px static hero banner is converted into an expandable executive drawer defaulting to a compact **40px summary strip**.
   - The separate 140px 4-card KPI strip is **subsumed directly into the 40px summary strip** when collapsed, rendering inline aggregate metrics:
     `Market Universe: 24,424 | Active Accounts: 300 | Portfolio Revenue: €5.44M | Discovered ICPs: 2 Clusters (ARI: 95.8%)`
   - The quick filter pills and form filter dropdowns are consolidated into a **single 44px command toolbar**.
   - Clicking the summary chevron expands the full executive metrics, algorithm diagrams, and ICP breakdowns on demand. Preference is stored in `localStorage.getItem('archimedes_hero_expanded')`.
   - **Total Space Reclaimed:** **450px+ of vertical height**.
2. **Fixed 100vh Cockpit Mechanics:**
   Instead of an unbounded scrolling `<body>` that forces operators to scroll the entire document, the UI is wrapped in a strict fixed-height flex container with independent inner pane scrolling:
   ```css
   .app-cockpit {
     height: 100vh;
     display: flex;
     flex-direction: column;
     overflow: hidden;
   }
   .workspace-split {
     flex: 1 1 0%;
     min-height: 0; /* Essential for child flex scrolling */
     display: flex;
     overflow: hidden;
   }
   .leads-table-pane {
     flex: 0 0 55%;
     overflow-y: auto;
     height: 100%;
   }
   .map-pane {
     flex: 1 1 0%;
     height: 100%;
     position: relative;
   }
   ```
3. **Guaranteed Viewport Geometry on 1366x768 Workstations:**
   - Usable Viewport Height: **648px**
   - Top Header Chrome: 65px
   - Collapsed Summary Strip: 40px
   - Unified Command Toolbar: 44px
   - Margins/Borders: 35px
   - **Total Top Chrome:** **184px**
   - **Guaranteed Usable Workspace:** $648	ext{px} - 184	ext{px} = \mathbf{464	ext{px}}$ ($\ge 10$ Standard 44px rows or 16 Compact 28px rows, with a 464px map canvas above the fold).

### 4.3 Synchronized Split-Screen Prospecting Workspace (55/45)
* **Screen Partitioning:** Left pane (55% width) hosts the enterprise leads table; right pane (45% width) hosts the interactive Leaflet map canvas.
* **Layout Mode Controller:** A 3-segment control in the toolbar enables instant switching:
  * `[ ◫ Split 50/50 ]`: Default synchronized prospecting cockpit.
  * `[ ▤ Table Only ]`: 100% width table for high-volume data auditing and bulk sorting.
  * `[ 🗺️ Map Only ]`: 100% width map for territorial field sales planning.
* **Bidirectional Synchronization Engine:**
  - *Table-to-Map Priming via Single DOM Overlay Halo:*
    Hovering any table row primes the corresponding map pin using a dedicated overlay marker (`L.divIcon`), avoiding `L.canvas()` DOM manipulation errors:
    ```javascript
    let hoverMarker = null;
    export function primeMapPin(map, lat, lon, color = '#38bdf8') {
      if (!hoverMarker) {
        hoverMarker = L.marker([lat, lon], {
          icon: L.divIcon({
            className: 'map-pulse-halo',
            html: `<div class="pulse-ring" style="--halo-color: ${color}"></div>`,
            iconSize: [32, 32],
            iconAnchor: [16, 16]
          }),
          zIndexOffset: 1000
        }).addTo(map);
      } else {
        hoverMarker.setLatLng([lat, lon]);
        hoverMarker.setOpacity(1);
      }
    }
    ```
  - *Map-to-Table Safe Navigation (Pagination-Aware):*
    Clicking a map pin for a prospect located on another table page calculates the target page, re-renders the table page via the reactive store, and smoothly scrolls the row into view before opening the drawer:
    ```javascript
    export function onMapMarkerClick(store, companyId, rank) {
      const pageSize = store.getState().pageSize || 25;
      const targetPage = Math.floor((rank - 1) / pageSize) + 1;
      if (store.getState().currentPage !== targetPage) {
        store.set('currentPage', targetPage);
        requestAnimationFrame(() => {
          const row = document.querySelector(`[data-company-id="${companyId}"]`);
          if (row) row.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        });
      } else {
        const row = document.querySelector(`[data-company-id="${companyId}"]`);
        if (row) row.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
      store.set('activeCompanyId', companyId);
      store.set('drawerOpen', true);
    }
    ```

### 4.4 True Slide-Over Contextual Drawer (Non-Blocking)
* **Dimensions & Anchor:** 480px (desktop) to 540px (ultrawide) panel anchored to the right edge (`position: fixed; right: 0; top: 0; bottom: 0; z-index: 40;`).
* **Non-Blocking Operation:** Slides out over the right-hand map pane. The left-hand data table remains 100% visible and interactive.
* **Continuous Keyboard Navigation:** The user can navigate prospects using `ArrowUp` or `ArrowDown` (or `J`/`K`); the drawer content updates instantly without closing or re-animating.
* **Tabbed Information Architecture:**
  1. **Tab 1: Overview & Sales Pitch:** Personalized B2B pitch hook, recommendation rationale, KAD activity hierarchy with Empirical Bayes Lift indicators, and corporate age.
  2. **Tab 2: Geospatial Radar & Mini-Map:** Leaflet street-level mini-map with exact address coordinates, OpenStreetMap verification tier, Transit Hub distance, and proximity radar displaying existing clients (<5km) in green and lookalikes in blue.
  3. **Tab 3: Corporate Registry & GEMI Verification:** Official GEMI commercial status, Modulo-11 VAT checksum verification status, Google Places operational check, and corporate branch hierarchy.

### 4.5 Enterprise Data Table with Column Pinning & Density Toggles
* **Sticky Column Pinning:** The rank, company name, and selection checkbox columns are pinned to `left: 0` with a dynamic right shadow depth cue when scrolled horizontally. Quick actions are pinned to `right: 0`.
* **Two-Way Column Sorting:** Clicking headers toggles `Ascending -> Descending -> Default Rank` with visual sorting carets on:
  * Rank & ICP Profile
  * Fit Score (Percentage)
  * Company Name (A–Z / Z–A)
  * Annual Revenue (Numeric)
  * Distance from Active Territory Center (km)
* **3-Way Information Density Toggle:**
  * **Compact (28px row height):** Single-line company name, score pill, icon actions. Displays ~16 prospects simultaneously on a 768p laptop display.
  * **Standard (44px row height):** Default mode. Company name, KAD description, fit score bar, location. Displays ~10 prospects.
  * **Detailed (64px row height):** Multi-line enrichment tags, recommendation justification snippet, street address.

### 4.6 Floating Batch Action Bar & Multi-Select Operations
* **Row Checkboxes:** Checkbox column with "Select All on Page" and "Select All 2,000 Filtered".
* **Floating Action Bar (FAB):** When $\ge 1$ prospect is selected, an elevated bar slides up from the viewport bottom:
  ```
  [ 14 Prospects Selected ]  |  [ 📥 Export CSV ]  [ 🔍 Batch Verify GEMI ]  [ 🏷️ Assign ICP ]  [ ✕ Clear Selection ]
  ```
* Non-destructive batch exclusions trigger a bottom-right toast with an explicit `[ Undo ]` affordance.

### 4.7 Production-Grade Keyboard Navigation & Greek Focus Shield
High-velocity sales reps can navigate the platform entirely from the keyboard. To eliminate typing collisions on Greek keyboard layouts (where physical key `M` emits Greek `μ`, `E` emits `ε`, `T` emits `τ`, and `Space` triggers page scroll), the platform implements an exhaustive input focus shield and `preventDefault` guards:

```javascript
export function initKeyboardNavigation(store) {
  window.addEventListener('keydown', (e) => {
    // 1. Omnibox / MiniSearch Global Intercept (Prevent browser address bar hijack)
    if ((e.metaKey || e.ctrlKey) && e.code === 'KeyK') {
      e.preventDefault();
      document.getElementById('leads-search-input')?.focus();
      return;
    }

    // 2. Comprehensive Input Focus Shield (Protects Greek text input)
    const activeEl = document.activeElement;
    const isEditing = activeEl && (
      activeEl.tagName === 'INPUT' ||
      activeEl.tagName === 'TEXTAREA' ||
      activeEl.tagName === 'SELECT' ||
      activeEl.isContentEditable
    );
    if (isEditing) {
      if (e.code === 'Escape') activeEl.blur();
      return; // Absolute shield against typing collisions
    }

    // 3. Modal / Drawer Dismissal Precedence
    if (e.code === 'Escape') {
      if (store.getState().activeModal) { store.set('activeModal', null); return; }
      if (store.getState().drawerOpen) { store.set('drawerOpen', false); return; }
      if (store.getState().selectedIds?.length > 0) { store.set('selectedIds', []); return; }
    }

    // 4. Scoped Table & Action Shortcuts (Mapped cleanly via event.code)
    switch (e.code) {
      case 'KeyJ':
      case 'ArrowDown':
        e.preventDefault();
        store.dispatch('SELECT_NEXT_PROSPECT');
        break;
      case 'KeyK':
      case 'ArrowUp':
        e.preventDefault();
        store.dispatch('SELECT_PREV_PROSPECT');
        break;
      case 'Enter':
        e.preventDefault();
        store.set('drawerOpen', !store.getState().drawerOpen);
        break;
      case 'KeyM':
        store.dispatch('TOGGLE_LAYOUT_MODE', 'map');
        break;
      case 'KeyT':
        store.dispatch('TOGGLE_LAYOUT_MODE', 'table');
        break;
      case 'KeyE':
        e.preventDefault();
        store.dispatch('EXPORT_CSV');
        break;
      case 'KeyD':
        store.dispatch('CYCLE_DENSITY');
        break;
    }
  });
}
```

| Shortcut | Context / Scope | Action Triggered | Safety Guard |
| :--- | :--- | :--- | :--- |
| `J` or `ArrowDown` | Data Table | Select and highlight next prospect row; updates open drawer. | Shielded when editing |
| `K` or `ArrowUp` | Data Table | Select and highlight previous prospect row; updates open drawer. | Shielded when editing |
| `Enter` | Data Table | Open / close contextual slide-over drawer for active row. | Shielded when editing |
| `Esc` | Global | Close modal -> close drawer -> clear selection (in precedence order). | Safe blur active element |
| `Ctrl+K` or `Cmd+K`| Global | Focus global MiniSearch input instantly. | `e.preventDefault()` stops browser hijack |
| `M` | Global | Toggle layout mode between Split-View and Map-Only. | Shielded when editing |
| `T` | Global | Toggle layout mode between Split-View and Table-Only. | Shielded when editing |
| `E` | Global | Trigger CSV export of current filtered view. | Shielded when editing |
| `D` | Global | Cycle table density (Compact -> Standard -> Detailed). | Shielded when editing |

### 4.8 Modular ES Component Decomposition & Reactive Store
The 3,164-line monolithic `src/static/app.js` is decomposed into 12 single-responsibility ES modules under `src/static/js/`:

```
src/static/
├── index.html                      # Semantic HTML shell with native <script type="importmap">
├── style.css                       # Design tokens, surface lightness, tabular figures
└── js/
    ├── app.js                      # Application bootstrap coordinator & lifecycle init
    ├── store.js                    # Centralized action-based reactive store with equality checks
    ├── api.js                      # Typed REST client (/api/run, /api/verify-company, etc.)
    ├── components/
    │   ├── kpiCockpit.js           # Executive KPI summary strip & progressive drawer
    │   ├── splitLayout.js          # Synchronized split-screen controller & draggable divider
    │   ├── leadsTable.js           # Pinned data table with sorting, density, and checkboxes
    │   ├── mapView.js              # Leaflet canvas manager, L.divIcon overlay, Supercluster
    │   ├── drawer.js               # Contextual right slide-over inspector with tabs
    │   ├── charts.js               # Chart.js wrappers with dark tokens & tabular figures
    │   ├── searchFilter.js         # MiniSearch engine with Greek diacritic normalizer & chips
    │   └── batchActions.js         # Floating contextual action bar & bulk operations
    └── utils/
        └── i18n.js                 # Zero-dependency translation engine & Intl formatters
```

#### Production Reactive Store (`src/static/js/store.js`)
To avoid shallow proxy traps on nested properties (`store.state.filters.search`), silent array push failures, and infinite recursion call stack overflows between table and map, the architecture employs an action-based reactive store with strict value equality checks:

```javascript
export function createStore(initialState) {
  const subscribers = new Map();
  let state = { ...initialState };

  function notify(prop, newValue, oldValue) {
    if (subscribers.has(prop)) {
      [...subscribers.get(prop)].forEach(cb => cb(newValue, oldValue));
    }
    if (subscribers.has('*')) {
      [...subscribers.get('*')].forEach(cb => cb(prop, newValue, oldValue));
    }
  }

  return {
    getState() { return state; },
    set(prop, updater) {
      const oldValue = state[prop];
      const newValue = typeof updater === 'function' ? updater(oldValue) : updater;
      // Strict equality guard against infinite update recursion
      if (oldValue === newValue) return;
      state[prop] = newValue;
      notify(prop, newValue, oldValue);
    },
    dispatch(action, payload) {
      // Coordinated action dispatcher for multi-property state transitions
      switch (action) {
        case 'SELECT_PROSPECT':
          this.set('activeCompanyId', payload.id);
          this.set('hoveredPinId', payload.id);
          break;
        case 'CYCLE_DENSITY': {
          const densities = ['compact', 'standard', 'detailed'];
          const current = densities.indexOf(state.density || 'standard');
          this.set('density', densities[(current + 1) % densities.length]);
          break;
        }
      }
    },
    subscribe(prop, callback) {
      if (!subscribers.has(prop)) subscribers.set(prop, new Set());
      subscribers.get(prop).add(callback);
      return () => subscribers.get(prop).delete(callback);
    }
  };
}
```

---

## 5. English & Greek Translation Consistency & Terminology Guide (R2)

### 5.1 Linguistic & Orthographic Audit Analysis
An exhaustive code and token audit across all six application files identified four primary defect categories:
1. **Broken Monotonic Accents in KPI Cards:** Eight prominent metric cards in Tab 2 and Tab 3 are authored without Greek accents (*tonoi*): `Συνολικες Ευκαιριες`, `Μεσος Βαθμος Ταυτισης`, `Κυριαρχος Νομος`, `Κυριαρχη Νομικη Μορφη`, `Πελατολογιο σε Αναλυση`, `Συνολικος Τζιρος`, `Κορυφαιος Τζιρος`, and `Δομη Πελατων`.
2. **Intersentence & Intercomponent Language Mixing:** Adjacent elements switch arbitrarily between Greek and English without design hierarchy. In the header bar, `Instant Scoring Active` and `Embeddings: On` are placed next to `Εξαιρέσεις (1)`, `Εξαγωγή`, and `Ρυθμίσεις Αναλυτή`. In table headers, `Rank & ICP` (English) sits alongside `Ταύτιση` (Greek).
3. **Colloquial & Informal Greek Business Slang:**
   * `Τζίρος` is an informal Italian-derived colloquialism (*giro*); professional B2B SaaS analytics standardizes to `Revenue` / `Annual Revenue` in English, and formal Greek `Κύκλος Εργασιών` (or `Έσοδα`) in localized views.
   * `Ομάδες` (teams/groups) is an awkward translation for algorithmic `Clusters` or `ICP Segments`.
   * `field sales days` is clumsily embedded into Greek text: *"ημερήσιες εξορμήσεις πεδίου (field sales days)"*. Standardizing to `Field Sales Routing / Territory Days` provides clarity.
4. **Leaked Internal Code Variables:** Mathematical notation and internal software variables appear directly in user-facing labels: `(w_cos)`, `(w_eb)`, `(w_sp)`, and `Section 5.2`. These must be completely encapsulated and replaced with professional domain terminology (`Cosine Similarity Trait Weight`, `Empirical Bayes Trait Weight`, `Spatial Trait Weight`, `Trait Contribution Breakdown`).

### 5.2 Canonical Standardized Terminology Glossary (43 Domain Terms)
The following canonical glossary standardizes terminology across five primary functional domains, enforcing the project policy of preferring standard professional English for specialized, machine learning, financial, and SaaS concepts:

#### Domain 1: Executive & Sales Intelligence
| # | Current Greek Term / Variation | Canonical Standard English | Context / Placement | Architectural & Ergonomic Rationale |
| :-: | :--- | :--- | :--- | :--- |
| 1 | `Προφίλ Πελατών / ICP` | **`ICP Profiles`** | Global Navigation & Tabs | Universal B2B sales standard for Ideal Customer Profile archetypes. |
| 2 | `Ανακαλυφθέντα ICPs / Ομάδες` | **`Discovered ICPs / ICP Segments`** | KPI Cards & Summary Strip | Eliminates colloquial 'Ομάδες'; standardizes on ML segments/clusters. |
| 3 | `Προτεινόμενοι Νέοι Πελάτες / Leads` | **`Lookalike Prospects / Lookalikes`** | Navigation, Tables, Exports | Aligns with standard B2B lookalike discovery (Apollo, ZoomInfo). |
| 4 | `Υφιστάμενο Πελατολόγιο / Ενεργοί` | **`Active Clientele / Current Accounts`** | Navigation & Cockpit | Standard CRM terminology distinguishing clients from prospects. |
| 5 | `Συνολικες Ευκαιριες (Pipeline)` | **`Total Pipeline Opportunities`** | KPI Card (L516) | Resolves broken accents; standardizes B2B sales pipeline terminology. |
| 6 | `Ταύτιση / Βαθμός Ταύτισης` | **`Match Score / Fit Score`** | Table Headers & Badges | Standard SaaS metric for algorithmic prospect compatibility score. |
| 7 | `Μεσος Βαθμος Ταυτισης` | **`Average Fit Score`** | KPI Card (L521) | Resolves unaccented text; standardizes aggregate match score. |
| 8 | `Θερμότητα Leads` | **`Lead Priority / Lead Heat`** | ICP Card Sub-Tabs | Standard sales velocity tiering (Hot, Warm, Cold). |
| 9 | `Οδηγός Πώλησης / Pitch Hook` | **`Sales Playbook & Pitch Script`** | ICP Card Sub-Tabs | Standard sales enablement nomenclature. |
| 10 | `Αγοραστικό Προφίλ` | **`Buyer Persona`** | Sales Playbook Cards | Standard product marketing customer persona term. |
| 11 | `Σημεία Τριβής` | **`Pain Points`** | Sales Playbook Cards | Universal sales discovery terminology. |
| 12 | `Αντιμετώπιση Αντιρρήσεων` | **`Objection Handling`** | Sales Playbook Cards | Standard sales objection rebuttal terminology. |
| 13 | `Κύκλος Πώλησης` | **`Sales Cycle`** | Sales Playbook Cards | Standard B2B sales velocity metric. |

#### Domain 2: Corporate, Tax & Financial
| # | Current Greek Term / Variation | Canonical Standard English | Context / Placement | Architectural & Ergonomic Rationale |
| :-: | :--- | :--- | :--- | :--- |
| 14 | `ΑΦΜ` | **`VAT ID / Tax ID`** | Data Tables, Drawer, Exports | International B2B standard (Stripe, Salesforce); 'VAT ID (ΑΦΜ)' in headers. |
| 15 | `Τζίρος / Συνολικός Τζίρος` | **`Annual Revenue / Total Revenue`** | KPI Cards, Tables, Popups | Replaces informal colloquial Greek 'Τζίρος' with professional financial term. Formal Greek canonical: `Κύκλος Εργασιών`. |
| 16 | `Μέσος Τζίρος / Διάμεσος Τζίρος` | **`Average Revenue / Median Revenue`** | Summary Bars & Tooltips | Professional statistical revenue distribution terminology. Formal Greek: `Διάμεσος Κύκλος Εργασιών`. |
| 17 | `Κορυφαιος Τζιρος` | **`Peak Account Revenue`** | KPI Card (L670) | Resolves broken accents; standardizes top client turnover metric. |
| 18 | `Μέση Αξία Πελάτη` | **`Average Revenue Per Account (ARPA)`**| KPI Badges & Cards | Industry standard B2B SaaS revenue metric. |
| 19 | `Δημοσιότητα Γ.Ε.ΜΗ.` | **`GEMI Commercial Registry`** | Drawer Actions & Verifications | Retains Greek registry acronym GEMI while standardizing directory context. |

#### Domain 3: Machine Learning & Analytics
| # | Current Greek Term / Variation | Canonical Standard English | Context / Placement | Architectural & Ergonomic Rationale |
| :-: | :--- | :--- | :--- | :--- |
| 20 | `Στατιστικό Lift / Εμπειρικό Bayes` | **`Statistical Lift / Empirical Bayes Lift`** | ICP Cards, Benchmarks | Universal data science metric; never translate literally. |
| 21 | `Ομοιότητα Προφίλ (w_cos)` | **`Cosine Similarity Trait Weight`** | Tuning Drawer Sliders | Purges leaked code variable `(w_cos)`; clarifies cosine trait metric. |
| 22 | `Εμπειρικό Bayes Lift (w_eb)` | **`Empirical Bayes Trait Weight`** | Tuning Drawer Sliders | Purges leaked code variable `(w_eb)`; standardized Bayesian hyperparameter label. |
| 23 | `Εγγύτητα Χάρτη (w_sp)` | **`Spatial Trait Weight`** | Tuning Drawer Sliders | Purges leaked code variable `(w_sp)`; replaces awkward 'Εγγύτητα Χάρτη'. |
| 24 | `Ακτίνα Εγγύτητας (km)` | **`Spatial Bandwidth Radius (km)`** | Tuning Drawer Sliders | Clarifies Gaussian spatial kernel bandwidth parameter. |
| 25 | `Empirical Bayes Prior` | **`Empirical Bayes Prior Strength (α)`**| Tuning Drawer Sliders | Standard Bayesian hyperparameter nomenclature. |
| 26 | `Κατώφλι NO_FIT` | **`NO_FIT Outlier Threshold`** | Tuning Drawer Sliders | Professional outlier detection threshold term. |
| 27 | `Νευρωνικά Embeddings` | **`Neural Semantic Embeddings`** | Header Pill & Tuning Drawer | Natural Language Processing representation terminology. |
| 28 | `Αλγεβρικό SVD Baseline` | **`Algebraic SVD Baseline (TF-IDF)`** | Model Dropdown | Clarifies non-neural linear algebra baseline. |
| 29 | `Παράμετροι Clustering (K-Means)` | **`K-Means Clustering Parameters`** | Tuning Drawer Headers | Standard unsupervised machine learning terminology. |
| 30 | `Bootstrap Σταθερότητα / ARI` | **`Bootstrap Stability (ARI)`** | Diagnostic Cards & Tech Tab | Adjusted Rand Index bootstrap validation terminology. |
| 31 | `Στάθμιση Τζίρου Πελατών` | **`Revenue Weighting Mode`** | Tuning Drawer Select | Clustering sample weight mode in ML pipeline. |
| 32 | `Λογαριθμική (ln τζίρου)` | **`Log-Revenue [ln(Revenue + 1)]`** | Tuning Weight Options | Standard feature engineering mathematical transform. |
| 33 | `Τετραγωνική ρίζα τζίρου` | **`Square Root Revenue [√Revenue]`** | Tuning Weight Options | Standard feature engineering mathematical transform. |
| 34 | `Ισόβαρη (χωρίς τζίρο)` | **`Uniform (Unweighted)`** | Tuning Weight Options | Standard unweighted sample option. |

#### Domain 4: Geospatial & Spatial Intelligence
| # | Current Greek Term / Variation | Canonical Standard English | Context / Placement | Architectural & Ergonomic Rationale |
| :-: | :--- | :--- | :--- | :--- |
| 35 | `Κόμβος Μεταφορών` | **`Transit Hub Catchment`** | Drawer Enrichment Badges | Geospatial accessibility feature label. |
| 36 | `Απόσταση OSM` | **`OpenStreetMap Match & Distance`** | Drawer Enrichment Badges | Spatial POI verification tier label. |
| 37 | `Επέκταση Δικτύου` | **`Network Expansion Velocity`** | Drawer Badges | Temporal corporate growth feature. |
| 38 | `Συντεταγμένες GPS` | **`Coordinates (Lat/Lng)`** | Drawer, Mini-Map, Tooltips | Standard geospatial coordinate notation. |

#### Domain 5: Workflow, Navigation & Actions
| # | Current Greek Term / Variation | Canonical Standard English | Context / Placement | Architectural & Ergonomic Rationale |
| :-: | :--- | :--- | :--- | :--- |
| 39 | `Ρυθμίσεις Αναλυτή` | **`Analyst Tuning Panel`** | Header Button & Drawer | Professional analytical control panel title. |
| 40 | `Καρτέλα Επιχείρησης` | **`Company Inspector / Account Detail`**| Action Buttons & Headers | Replaces colloquial Greek 'Καρτέλα' with standard SaaS 'Company Inspector'. |
| 41 | `Εξαιρέσεις (Blacklist)` | **`Excluded Accounts / Pipeline Blacklist`**| Header Badge & Modal | Clear B2B pipeline exclusion management terminology. |
| 42 | `Εξαγωγή Φιλτραρισμένων (CSV)` | **`Export Filtered Leads (CSV)`** | Table Toolbar Action | Standard enterprise data export action. |
| 43 | `Έλεγχος Γ.Ε.ΜΗ. & Places` | **`Verify via GEMI & Google Places`** | Drawer Action Toolbar | Clear dual-source verification action button. |

---

### 5.3 Prioritized High-Impact String Replacement Catalog (Core Foundation & High-Priority UI)
**Scope Clarification & Architectural Bridge:**
The catalog below details 213 verified concrete string substitutions across six files (`src/static/index.html` [138], `src/static/app.js` [49], `src/service.py` [11], `src/verifier.py` [7], `src/server.py` [5], and `src/llm_explainer.py` [3]). Every row represents an exact, verified code replacement to be executed during post-refactoring rollout.

* **Foundational Catalog Scope:** This catalog establishes immediate terminological consistency across all primary static shell elements, high-visibility metric cards, interaction buttons, API error payloads, and key dynamic templates.
* **Synergy with Section 5.4 JSON Dictionaries:** The remaining ~300+ dynamic strings, contextual tooltips, and backend parameter descriptions in the codebase are systematically transitioned via the structured locale dictionaries (`en.json` and `el.json`) specified in Section 5.4, preventing translation debt and hardcoding regressions.
* **Line-Scoped Execution Rule:** All replacements MUST be executed scoped by `Target File` and `Line` number rather than global unanchored substitutions to avoid collision between identical tokens in different UI contexts.

| # | Target File | Line | Category | Current Text / Code Snippet | Proposed Standardized Replacement | Rationale |
| :---: | :--- | :---: | :--- | :--- | :--- | :--- |
| 1 | `src/static/index.html` | L6 | Header Consistency | `<title>ForceCRM · Ανάλυση Πελατολογίου & Προτάσεις Νέων Πελατών (Archimedes 2026)</title>` | **`<title>ForceCRM · B2B ICP Discovery & Lookalike Scoring (Archimedes 2026)</title>`** | Aligns title with executive B2B platform branding and removes clumsy Greek translation. |
| 2 | `src/static/index.html` | L80 | Header Consistency | `Έξυπνη Ανάλυση Πελατολογίου &amp; Προτάσεις Νέων Πελατών (B2B ICP Discovery)` | **`ForceCRM Executive Intelligence · B2B ICP Discovery & Lookalike Engine`** | Elevates product title to executive standard; removes awkward Greek parenthetical. |
| 3 | `src/static/index.html` | L94 | Tooltip Localization | `title="Ρύθμιση μοντέλων embeddings στην πλευρική γραμμή"` | **`title="Configure Semantic KAD Embeddings in Tuning Drawer"`** | Replaces Greek tooltip on English button 'Embeddings: On'. |
| 4 | `src/static/index.html` | L106 | Tooltip Localization | `title="Προβολή αποκλεισμένων επιχειρήσεων"` | **`title="View Excluded Companies (Blacklist)"`** | Replaces Greek tooltip on English/Greek button. |
| 5 | `src/static/index.html` | L108 | Terminology Standardization | `Εξαιρέσεις (` | **`Excluded (`** | Standardizes navigation badge text to English. |
| 6 | `src/static/index.html` | L115 | Terminology Standardization | `Εξαγωγή` | **`Export`** | Standardizes dropdown button label. |
| 7 | `src/static/index.html` | L124 | Terminology Standardization | `Προτεινόμενοι πελάτες (Top 2.000)` | **`Lookalike Prospects (Top 2,000)`** | Standardizes export option description to B2B lookalike prospecting. |
| 8 | `src/static/index.html` | L131 | Terminology Standardization | `Προφίλ πελατών & AI περιγραφές` | **`ICP Profiles & AI Sales Playbooks`** | Replaces vague Greek description with clear B2B deliverable label. |
| 9 | `src/static/index.html` | L138 | Terminology Standardization | `Κατηγοριοποίηση πελατολογίου` | **`Clientele Segmentation & Cluster Mapping`** | Professional segmentation terminology. |
| 10 | `src/static/index.html` | L147 | Terminology Standardization | `Ρυθμίσεις Αναλυτή` | **`Analyst Tuning`** | Standardizes drawer toggle button. |
| 11 | `src/static/index.html` | L171 | Data Badge Standardization | `\| 24.424 Ελληνικές Επιχειρήσεις Εστίασης` | **`\| 24,424 Universe Enterprises`** | Replaces localized Greek label with standard universe metric. |
| 12 | `src/static/index.html` | L173 | Data Badge Standardization | `300 Ενεργοί Πελάτες CRM` | **`300 Active CRM Clientele`** | Standardizes active customer badge. |
| 13 | `src/static/index.html` | L190 | Action Standardization | `Φόρτωση Νέου Πελατολογίου` | **`Upload Clientele (.xlsx)`** | Clear, concise action button. |
| 14 | `src/static/index.html` | L194 | Technical Standardization | `Παράμετροι Μοντέλου` | **`Model Hyperparameters`** | Technical tuning drawer action button. |
| 15 | `src/static/index.html` | L223 | KPI Metric Card | `Ενεργοί Πελάτες` | **`Active Clientele`** | Executive KPI header. |
| 16 | `src/static/index.html` | L230 | KPI Metric Card | `επιχειρήσεις` | **`accounts`** | Replaces Greek count unit with standard B2B account unit. |
| 17 | `src/static/index.html` | L233 | KPI Metric Card | `286 σε Προφίλ` | **`286 Assigned to ICPs`** | Clarifies customer assignment status. |
| 18 | `src/static/index.html` | L242 | Financial Terminology | `Συνολικός Τζίρος` | **`Total Annual Revenue`** | Replaces colloquial Greek 'Τζίρος' with professional financial term 'Revenue'. |
| 19 | `src/static/index.html` | L251 | Financial Terminology | `€18.144 / πελάτη` | **`€18,144 / account (ARPA)`** | Standard Average Revenue Per Account metric. |
| 20 | `src/static/index.html` | L258 | ML Metric Card | `Ανακαλυφθέντα ICPs` | **`Discovered ICPs`** | Replaces mixed term with standard English. |
| 21 | `src/static/index.html` | L264 | ML Metric Card | `2 Ομάδες` | **`2 ICP Clusters`** | Replaces colloquial 'Ομάδες' with standard clustering term. |
| 22 | `src/static/index.html` | L269 | ML Metric Card | `Bootstrap Σταθερότητα:` | **`Bootstrap Stability (ARI):`** | Standard machine learning validation metric. |
| 23 | `src/static/index.html` | L276 | Sales Intelligence Card | `Προτεινόμενα Leads` | **`Lookalike Prospects`** | Replaces Greek/English hybrid 'Προτεινόμενα Leads'. |
| 24 | `src/static/index.html` | L297 | Tab Navigation | `1. Προφίλ Πελατών (ICPs) &amp; Χάρτης` | **`1. ICP Profiles &amp; Geospatial Map`** | Consistent high-density executive tab navigation. |
| 25 | `src/static/index.html` | L301 | Tab Navigation | `2. Προτεινόμενοι Νέοι Πελάτες (Lookalikes)` | **`2. Lookalike Prospects (B2B Radar)`** | Consistent high-density executive tab navigation. |
| 26 | `src/static/index.html` | L306 | Tab Navigation | `3. Υφιστάμενο Πελατολόγιο` | **`3. Active Clientele (Cockpit)`** | Consistent high-density executive tab navigation. |
| 27 | `src/static/index.html` | L311 | Tab Navigation | `4. Τεχνική Τεκμηρίωση &amp; Αξιολόγηση AI` | **`4. Benchmark Validation &amp; EU AI Act`** | Accurate technical tab title. |
| 28 | `src/static/index.html` | L330 | Summary Bar | `<span class="text-xs text-slate-400 font-medium">Ανά ICP</span>` | **`<span class="text-xs text-slate-400 font-medium">By ICP Profile</span>`** | Fixed exact match (removed trailing colon). |
| 29 | `src/static/index.html` | L340 | Financial Terminology | `Τζίρος` | **`Revenue`** | Replaces colloquial 'Τζίρος'. |
| 30 | `src/static/index.html` | L344 | Financial Terminology | `Μερίδιο Τζίρου €` | **`Revenue Share (%)`** | Clear financial distribution label. |
| 31 | `src/static/index.html` | L352 | Summary Bar | `Πελάτες` | **`Accounts`** | Standard B2B CRM entity label. |
| 32 | `src/static/index.html` | L356 | Summary Bar | `Κατανομή Πελατών` | **`Clientele Distribution`** | Clear distribution label. |
| 33 | `src/static/index.html` | L363 | Financial Terminology | `Μέση Αξία Πελάτη` | **`Avg Account Revenue (ARPA)`** | Standard SaaS metric. |
| 34 | `src/static/index.html` | L367 | Financial Terminology | `Ετήσιος Τζίρος` | **`Annual Contract Value`** | Standard sales intelligence metric. |
| 35 | `src/static/index.html` | L371 | Summary Bar | `Κάλυψη Προφίλ` | **`ICP Coverage`** | Clear clustering coverage ratio. |
| 36 | `src/static/index.html` | L375 | Summary Bar | `300 / 300 Ενεργοί` | **`300 / 300 Active (100%)`** | Concise status indicator. |
| 37 | `src/static/index.html` | L427 | Map Filter | `Όλα τα Σημεία (Πελάτες + Leads)` | **`All Points (Clientele + Lookalikes)`** | Consistent terminology. |
| 38 | `src/static/index.html` | L428 | Map Filter | `⭐ Μόνο Υφιστάμενοι Πελάτες (300)` | **`⭐ Active Clientele Only (300)`** | Consistent terminology. |
| 39 | `src/static/index.html` | L429 | Map Filter | `🎯 Μόνο Προτεινόμενα Leads (2.000)` | **`🎯 Lookalike Prospects Only (2,000)`** | Consistent terminology. |
| 40 | `src/static/index.html` | L436 | Map Preset | `🇬🇷 Όλη η Ελλάδα` | **`🇬🇷 All Greece`** | Bilingual / English standard. |
| 41 | `src/static/index.html` | L437 | Map Preset | `🏛️ Αττική` | **`🏛️ Attica`** | Standard geographic region name. |
| 42 | `src/static/index.html` | L438 | Map Preset | `⚡ Θεσσαλονίκη` | **`⚡ Thessaloniki`** | Standard geographic region name. |
| 43 | `src/static/index.html` | L439 | Map Preset | `🏖️ Χαλκιδική` | **`🏖️ Halkidiki`** | Standard geographic region name. |
| 44 | `src/static/index.html` | L440 | Map Preset | `🏔️ Κρήτη` | **`🏔️ Crete`** | Standard geographic region name. |
| 45 | `src/static/index.html` | L441 | Map Preset | `⛵ Κυκλάδες` | **`⛵ Cyclades`** | Standard geographic region name. |
| 46 | `src/static/index.html` | L442 | Map Preset | `☀️ Δωδεκάνησα` | **`☀️ Dodecanese`** | Standard geographic region name. |
| 47 | `src/static/index.html` | L443 | Map Preset | `🌊 Ιόνια Νησιά` | **`🌊 Ionian Islands`** | Standard geographic region name. |
| 48 | `src/static/index.html` | L444 | Map Preset | `🌿 Πελοπόννησος` | **`🌿 Peloponnese`** | Standard geographic region name. |
| 49 | `src/static/index.html` | L445 | Map Preset | `🌾 Θεσσαλία` | **`🌾 Thessaly`** | Standard geographic region name. |
| 50 | `src/static/index.html` | L458 | Map Legend | `⭐ Υφιστάμενοι Πελάτες (Μεγάλοι κύκλοι)` | **`⭐ Active Clientele (Large Circles)`** | Consistent terminology. |
| 51 | `src/static/index.html` | L459 | Map Legend | `🎯 Lookalike Leads (Μικροί κύκλοι)` | **`🎯 Lookalike Prospects (Small Circles)`** | Consistent terminology. |
| 52 | `src/static/index.html` | L461 | Map Legend | `Κάντε κλικ σε οποιοδήποτε σημείο για πλήρη καρτέλα` | **`Click any marker to open Company Inspector`** | Clear workflow hint. |
| 53 | `src/static/index.html` | L474 | Profile List | `Ταξινομημένα κατά μερίδιο συνολικού τζίρου` | **`Ranked by total revenue share`** | Replaces 'τζίρου' with revenue. |
| 54 | `src/static/index.html` | L500 | Section Description | `Επιχειρήσεις της αγοράς με την υψηλότερη εμπορική συμβατότητα (εξαιρούνται αυστηρά όλοι οι υπάρχοντες πελάτες και τα υποκαταστήματά τους).` | **`Target market enterprises with highest algorithmic compatibility. Active clientele and associated branches are strictly excluded from pipeline.`** | Professional B2B prospecting description. |
| 55 | `src/static/index.html` | L505 | Table Counter | `Εμφανίζονται:` | **`Showing:`** | English standard. |
| 56 | `src/static/index.html` | L508 | Action Button | `Εξαγωγή Φιλτραρισμένων (CSV)` | **`Export Filtered (CSV)`** | Clear export label. |
| 57 | `src/static/index.html` | L516 | Broken Accent / Terminology | `Συνολικες Ευκαιριες (Pipeline)` | **`Total Pipeline Opportunities`** | Fixes unaccented Greek 'Συνολικες Ευκαιριες'; applies standard B2B Pipeline term. |
| 58 | `src/static/index.html` | L518 | KPI Metric | `100% Νέες (Εξαιρούνται πελάτες)` | **`100% Net-New (Clients Excluded)`** | Standard sales pipeline qualifier. |
| 59 | `src/static/index.html` | L521 | Broken Accent / Terminology | `Μεσος Βαθμος Ταυτισης` | **`Average Fit Score`** | Fixes unaccented Greek 'Μεσος Βαθμος Ταυτισης'; standardizes to Fit Score. |
| 60 | `src/static/index.html` | L523 | KPI Metric | `Κορυφαίο σκορ:` | **`Top Match Score:`** | Standard scoring terminology. |
| 61 | `src/static/index.html` | L526 | Broken Accent / Terminology | `Κυριαρχος Νομος` | **`Dominant Region (Prefecture)`** | Fixes unaccented Greek 'Κυριαρχος Νομος'; clarifies prefecture vs region. |
| 62 | `src/static/index.html` | L531 | Broken Accent / Terminology | `Κυριαρχη Νομικη Μορφη` | **`Dominant Legal Entity`** | Fixes unaccented Greek 'Κυριαρχη Νομικη Μορφη'. |
| 63 | `src/static/index.html` | L539 | Filter Preset | `✨ Όλα τα Leads (2.000)` | **`✨ All Prospects (2,000)`** | Consistent prospect terminology. |
| 64 | `src/static/index.html` | L540 | Filter Preset | `⭐ Top 100 Elite Leads` | **`⭐ Top 100 Elite Prospects`** | Consistent prospect terminology. |
| 65 | `src/static/index.html` | L541 | Filter Preset | `🏛️ Μόνο Αττική` | **`🏛️ Attica Only`** | Standard geographic label. |
| 66 | `src/static/index.html` | L542 | Filter Preset | `⚡ Θεσσαλονίκη` | **`⚡ Thessaloniki Only`** | Standard geographic label. |
| 67 | `src/static/index.html` | L543 | Filter Preset | `🏖️ Τουριστικά &amp; Νησιά` | **`🏖️ Tourism &amp; Islands`** | Standard segment label. |
| 68 | `src/static/index.html` | L544 | Filter Preset | `🏢 Εταιρικές (ΙΚΕ/ΑΕ)` | **`🏢 Corporate Entities (IKE/SA)`** | Professional entity format. |
| 69 | `src/static/index.html` | L545 | Filter Preset | `📍 Μόνο Έδρες` | **`📍 Headquarters Only`** | Standard commercial entity type. |
| 70 | `src/static/index.html` | L553 | Search Input | `Αναζήτηση (Επωνυμία, ΑΦΜ, Πόλη)` | **`Search (Company Name, VAT ID, City)`** | Replaces ΑΦΜ with VAT ID. |
| 71 | `src/static/index.html` | L556 | Search Placeholder | `placeholder="π.χ. Εστιατόριο, 094..., Αθήνα"` | **`placeholder="Search company, VAT number, municipality..."`** | B2B search placeholder. |
| 72 | `src/static/index.html` | L562 | Filter Label | `Προφίλ ICP` | **`ICP Profile`** | Standard English. |
| 73 | `src/static/index.html` | L564 | Filter Option | `Όλα τα Προφίλ` | **`All Profiles`** | Standard English. |
| 74 | `src/static/index.html` | L570 | Filter Label | `Νομός` | **`Prefecture`** | Standard English. |
| 75 | `src/static/index.html` | L572 | Filter Option | `Όλοι οι Νομοί` | **`All Prefectures`** | Standard English. |
| 76 | `src/static/index.html` | L578 | Filter Label | `Νομική Μορφή` | **`Legal Form`** | Standard English. |
| 77 | `src/static/index.html` | L580 | Filter Option | `Όλες οι Μορφές` | **`All Entity Types`** | Standard English. |
| 78 | `src/static/index.html` | L590 | Table Header | `Rank &amp; ICP` | **`Rank &amp; ICP Profile`** | Clear header. |
| 79 | `src/static/index.html` | L591 | Table Header | `Ταύτιση` | **`Fit Score`** | Standard SaaS lead qualification metric. |
| 80 | `src/static/index.html` | L592 | Table Header | `Επωνυμία &amp; Δραστηριότητα (ΚΑΔ)` | **`Company &amp; Primary KAD Activity`** | Clear industry classification header. |
| 81 | `src/static/index.html` | L593 | Table Header | `Τοποθεσία &amp; ΑΦΜ` | **`Location &amp; VAT ID`** | Standard corporate identifier. |
| 82 | `src/static/index.html` | L594 | Table Header | `Ενέργειες` | **`Actions`** | Standard table header. |
| 83 | `src/static/index.html` | L643 | Section Description | `Δείτε σε ποιο ICP ανήκει κάθε πελάτης, καθώς και ποιες επιχειρήσεις αποτελούν ειδικές περιπτώσεις (NO_FIT).` | **`Inspect ICP assignment for every active customer, track revenue contribution, and review outlier accounts (NO_FIT).`** | Executive customer audit description. |
| 84 | `src/static/index.html` | L649 | Search Placeholder | `placeholder="Αναζήτηση πελάτη..."` | **`placeholder="Search customer name, VAT, city..."`** | Standard search placeholder. |
| 85 | `src/static/index.html` | L652 | Filter Option | `Όλα τα ICPs &amp; NO_FIT` | **`All ICPs &amp; NO_FIT Outliers`** | Standard outlier terminology. |
| 86 | `src/static/index.html` | L660 | Broken Accent / Terminology | `Πελατολογιο σε Αναλυση` | **`Clientele Under Analysis`** | Fixes unaccented Greek 'Πελατολογιο σε Αναλυση'. |
| 87 | `src/static/index.html` | L661 | KPI Metric | `300 Πελάτες` | **`300 Active Accounts`** | Standard B2B CRM account terminology. |
| 88 | `src/static/index.html` | L662 | KPI Metric | `286 Σε Προφίλ · 14 NO_FIT` | **`286 In ICPs · 14 Outliers (NO_FIT)`** | Clear classification. |
| 89 | `src/static/index.html` | L665 | Broken Accent / Financial Terminology | `Συνολικος Τζιρος` | **`Total Portfolio Revenue`** | Fixes unaccented Greek 'Συνολικος Τζιρος'; replaces 'Τζίρος' with Revenue. |
| 90 | `src/static/index.html` | L667 | KPI Metric | `Μέσος: €18.144 / πελάτη` | **`Avg ARPA: €18,144 / account`** | Standard ARPA metric. |
| 91 | `src/static/index.html` | L670 | Broken Accent / Financial Terminology | `Κορυφαιος Τζιρος` | **`Peak Account Revenue`** | Fixes unaccented Greek 'Κορυφαιος Τζιρος'; replaces 'Τζίρος'. |
| 92 | `src/static/index.html` | L675 | Broken Accent / Terminology | `Δομη Πελατων` | **`Legal Entity Breakdown`** | Fixes unaccented Greek 'Δομη Πελατων'. |
| 93 | `src/static/index.html` | L683 | Filter Preset | `✨ Όλοι οι Πελάτες (300)` | **`✨ All Clients (300)`** | Consistent CRM terminology. |
| 94 | `src/static/index.html` | L684 | Filter Preset | `💎 VIP Τζίρος (>€25.000)` | **`💎 High Revenue (>€25,000)`** | Professional tier label. |
| 95 | `src/static/index.html` | L685 | Filter Preset | `⭐ ICP 1 (Εταιρικές Αττικής)` | **`⭐ ICP 1 (Corporate Attica)`** | Standard segment label. |
| 96 | `src/static/index.html` | L686 | Filter Preset | `⭐ ICP 2 (Τουριστικές Περιφέρειας)` | **`⭐ ICP 2 (Regional Hospitality)`** | Standard segment label. |
| 97 | `src/static/index.html` | L687 | Filter Preset | `⚠️ NO_FIT (Ειδικές Περιπτώσεις)` | **`⚠️ NO_FIT (Outlier Accounts)`** | Professional ML outlier label. |
| 98 | `src/static/index.html` | L688 | Filter Preset | `🏢 Πολλαπλά Υποκαταστήματα` | **`🏢 Multi-Branch Accounts`** | Standard account structure label. |
| 99 | `src/static/index.html` | L696 | Table Header | `ICP &amp; Ταύτιση` | **`ICP Profile &amp; Fit Score`** | Standard header. |
| 100 | `src/static/index.html` | L697 | Table Header | `Επωνυμία &amp; Δραστηριότητα (ΚΑΔ)` | **`Company &amp; Primary KAD Activity`** | Standard header. |
| 101 | `src/static/index.html` | L698 | Table Header | `Τζίρος (€)` | **`Annual Revenue (€)`** | Professional financial header. |
| 102 | `src/static/index.html` | L699 | Table Header | `Τοποθεσία &amp; ΑΦΜ` | **`Location &amp; VAT ID`** | Standard identifier. |
| 103 | `src/static/index.html` | L700 | Table Header | `Ενέργειες` | **`Actions`** | Standard header. |
| 104 | `src/static/index.html` | L758 | Documentation | `Μεθοδολογία Ελέγχου:` | **`Validation Methodology:`** | Technical header. |
| 105 | `src/static/index.html` | L789 | Compliance Header | `Συμμόρφωση με τον Ευρωπαϊκό Κανονισμό Τεχνητής Νοημοσύνης (EU AI Act - Άρθρο 50)` | **`Regulatory Transparency & Grounding Statement (EU AI Act — Article 50)`** | Official EU AI Act regulatory compliance title. |
| 106 | `src/static/index.html` | L842 | Drawer Header | `Ρυθμίσεις Αναλυτή` | **`Analyst Tuning Panel`** | Professional title. |
| 107 | `src/static/index.html` | L856 | Upload Section | `1. Αρχείο Πελατολογίου (.xlsx)` | **`1. Upload Clientele Dataset (.xlsx)`** | Clear workflow header. |
| 108 | `src/static/index.html` | L860 | Upload Box | `Σύρετε ή πατήστε για ανέβασμα Excel` | **`Drag & drop or click to upload Excel file`** | Clean upload instructions. |
| 109 | `src/static/index.html` | L861 | Upload Box | `Στήλες: ΑΦΜ και Τζίρος (€)` | **`Required columns: VAT (ΑΦΜ) and Revenue (€)`** | Clarifies column requirements. |
| 110 | `src/static/index.html` | L868 | Scoring Weights | `2. Βάρη Βαθμολόγησης (Lookalikes)` | **`2. Lookalike Scoring Weights`** | Standard scoring weights header. |
| 111 | `src/static/index.html` | L875 | Model Parameter | `Ομοιότητα Προφίλ (w_cos)` | **`Cosine Similarity Trait Weight`** | Purges leaked code variable `(w_cos)`. |
| 112 | `src/static/index.html` | L884 | Model Parameter | `Στατιστικό EB Lift (w_eb)` | **`Empirical Bayes Trait Weight`** | Purges leaked code variable `(w_eb)`. |
| 113 | `src/static/index.html` | L893 | Model Parameter | `Εγγύτητα Χάρτη (w_sp)` | **`Spatial Trait Weight`** | Purges leaked code variable `(w_sp)`. |
| 114 | `src/static/index.html` | L902 | Model Parameter | `Ακτίνα Εγγύτητας (km)` | **`Spatial Bandwidth Radius (km)`** | Precise mathematical parameter label. |
| 115 | `src/static/index.html` | L911 | Model Parameter | `Empirical Bayes Prior` | **`Empirical Bayes Prior Strength (α)`** | Standard statistical parameter. |
| 116 | `src/static/index.html` | L920 | Model Parameter | `Κατώφλι NO_FIT` | **`NO_FIT Outlier Threshold`** | Professional outlier detection label. |
| 117 | `src/static/index.html` | L930 | NLP Section | `3. Μοντέλα Embeddings ΚΑΔ` | **`3. Semantic KAD Embeddings`** | Replaces awkward phrasing. |
| 118 | `src/static/index.html` | L940 | Toggle Switch | `Χρήση Μοντέλων Embeddings` | **`Enable Neural Embeddings`** | Clear toggle action. |
| 119 | `src/static/index.html` | L953 | Toggle State | `Νευρωνικά Embeddings Ενεργά` | **`Neural Embeddings Active`** | Clear status. |
| 120 | `src/static/index.html` | L961 | Select Box | `Επιλογή Μοντέλου Embeddings` | **`Embedding Model Architecture`** | Standard NLP architecture label. |
| 121 | `src/static/index.html` | L967 | Model Option | `char-ngram-svd (Αλγεβρικό SVD Baseline)` | **`char-ngram-svd (Algebraic SVD Baseline)`** | English standard. |
| 122 | `src/static/index.html` | L990 | Status Alert | `Μοντέλα Embeddings Ανενεργά` | **`Neural Embeddings Disabled`** | Clear status. |
| 123 | `src/static/index.html` | L1001 | Clustering Section | `4. Παράμετροι Clustering (K-Means)` | **`4. K-Means Clustering Parameters`** | Standard unsupervised learning title. |
| 124 | `src/static/index.html` | L1019 | Sample Weighting | `Στάθμιση Τζίρου Πελατών` | **`Revenue Weighting Mode`** | Replaces 'Στάθμιση Τζίρου'. |
| 125 | `src/static/index.html` | L1021 | Option | `Λογαριθμική (ln τζίρου)` | **`Log-Revenue [ln(Revenue + 1)]`** | Standard feature transform. |
| 126 | `src/static/index.html` | L1022 | Option | `Τετραγωνική ρίζα τζίρου` | **`Square Root Revenue [√Revenue]`** | Standard feature transform. |
| 127 | `src/static/index.html` | L1023 | Option | `Γραμμική` | **`Linear Revenue`** | Standard option. |
| 128 | `src/static/index.html` | L1024 | Option | `Ισόβαρη (χωρίς τζίρο)` | **`Uniform (Unweighted)`** | Standard option. |
| 129 | `src/static/index.html` | L1033 | Action Button | `Επαναφορά Προεπιλογών` | **`Reset to Defaults`** | Standard button. |
| 130 | `src/static/index.html` | L1037 | Action Button | `Εφαρμογή &amp; Κλείσιμο` | **`Apply &amp; Recompute`** | Active pipeline execution button. |
| 131 | `src/static/index.html` | L1056 | Drawer Header | `Καρτέλα Επιχείρησης` | **`Company Inspector`** | Replaces colloquial 'Καρτέλα' with standard SaaS 'Company Inspector'. |
| 132 | `src/static/index.html` | L1057 | Drawer Subheader | `Πλήρες Εταιρικό Μητρώο, Γ.Ε.ΜΗ. &amp; Αναλυτικοί ΚΑΔ` | **`Corporate Registry, GEMI Verification &amp; KAD Activities`** | Professional registry verification subheader. |
| 133 | `src/static/index.html` | L1245 | Modal Header | `Αποκλεισμένες Επιχειρήσεις (CRM Blacklist)` | **`Excluded Companies (Pipeline Blacklist)`** | Professional blacklist management header. |
| 134 | `src/static/index.html` | L1246 | Modal Subheader | `Επιχειρήσεις με διακοπή εργασιών που εξαιρούνται από τα Lookalikes` | **`Closed or deprecated enterprises excluded from lookalike prospect scoring`** | Clear administrative subheader. |
| 135 | `src/static/app.js` | L183 | Toast Notification | `📋 Αντιγράφηκε το ΑΦΜ ${cleanVat}. Μετάβαση στη Δημοσιότητα Γ.Ε.ΜΗ.` | **`📋 Copied VAT ID ${cleanVat}. Opening GEMI Public Directory.`** | Replaces Greek toast with professional standard. |
| 136 | `src/static/app.js` | L252 | Toast Notification | `📍 Εστίαση στον Χάρτη: ${companyName \|\| 'Επιχείρηση'}` | **`📍 Focus Map on: ${companyName \|\| 'Enterprise'}`** | Clear map navigation toast. |
| 137 | `src/static/app.js` | L254 | Toast Notification | `⚠️ Δεν υπάρχουν διαθέσιμες συντεταγμένες GPS για αυτή την επιχείρηση.` | **`⚠️ No GPS coordinates available for this enterprise.`** | Precise coordinates warning. |
| 138 | `src/static/app.js` | L380 | Dynamic KPI Badge | `${meta.assigned \|\| 286} σε Προφίλ` | **`${meta.assigned \|\| 286} Assigned to ICPs`** | Consistent assigned status. |
| 139 | `src/static/app.js` | L384 | Dynamic KPI Badge | `€${Math.round(meta.mean_turnover \|\| 0).toLocaleString('el-GR')} / πελάτη` | **`€${Math.round(meta.mean_turnover \|\| 0).toLocaleString('en-US')} / account (ARPA)`** | Standard revenue formatting. |
| 140 | `src/static/app.js` | L386 | Dynamic KPI Badge | `${meta.optimal_k \|\| 2} Ομάδες` | **`${meta.optimal_k \|\| 2} ICP Clusters`** | Replaces 'Ομάδες'. |
| 141 | `src/static/app.js` | L527 | Map Popup | `⭐ Υφιστάμενος Πελάτης · ${escapeHtml(c.profile_id)}` | **`⭐ Active Client · ${escapeHtml(c.profile_id)}`** | Standard customer badge. |
| 142 | `src/static/app.js` | L530 | Map Popup | `Τζίρος: <strong class="text-white">€${(c.turnover \|\| 0).toLocaleString('el-GR')}</strong> \| Ταύτιση: <strong class="text-emerald-400">${(c.score * 100).toFixed(1)}%</strong>` | **`Revenue: <strong class="text-white">€${(c.turnover \|\| 0).toLocaleString('en-US')}</strong> \| Fit Score: <strong class="text-emerald-400">${(c.score * 100).toFixed(1)}%</strong>`** | Standard revenue and score. |
| 143 | `src/static/app.js` | L573 | Map Popup | `🎯 Προτεινόμενο Lead #${l.rank} · ${escapeHtml(l.profile_id)}` | **`🎯 Lookalike Prospect #${l.rank} · ${escapeHtml(l.profile_id)}`** | Consistent prospect badge. |
| 144 | `src/static/app.js` | L576 | Map Popup | `Ταύτιση: <strong class="text-emerald-400">${(l.score * 100).toFixed(1)}%</strong> \| 🏛️ ${escapeHtml(l.legal_type \|\| '')}` | **`Fit Score: <strong class="text-emerald-400">${(l.score * 100).toFixed(1)}%</strong> \| 🏛️ ${escapeHtml(l.legal_type \|\| '')}`** | Standard fit score. |
| 145 | `src/static/app.js` | L602 | Map Status Bar | `📍 ${total.toLocaleString('el-GR')} Σημεία (${clientMarkerCount} Πελάτες · ${leadMarkerCount.toLocaleString('el-GR')} Leads)` | **`📍 ${total.toLocaleString('en-US')} Entities (${clientMarkerCount} Clients · ${leadMarkerCount.toLocaleString('en-US')} Prospects)`** | Clean entity counts. |
| 146 | `src/static/app.js` | L728 | Profile Archetype Badge | `${isHighYield ? 'Υπεραπόδοση Τζίρου' : 'Υψηλός Όγκος'}` | **`${isHighYield ? 'High Revenue Yield' : 'High Volume Accounts'}`** | Replaces literal Greek translations. |
| 147 | `src/static/app.js` | L735 | Profile Card | `<span class="text-slate-400">Μερίδιο Τζίρου:</span>` | **`<span class="text-slate-400">Revenue Share:</span>`** | Standard financial label. |
| 148 | `src/static/app.js` | L745 | Profile Card | `<span class="text-slate-400">Πλήθος Πελατών:</span>` | **`<span class="text-slate-400">Client Count:</span>`** | Standard CRM label. |
| 149 | `src/static/app.js` | L832 | Sales Cadence Action | `action: 'Άμεση Κλήση & Demo (48h)'` | **`action: 'Direct Call & 15-Min Demo (48h)'`** | Standard sales playbook action. |
| 150 | `src/static/app.js` | L837 | Sales Territory Tip | `territory_tip: 'Συγκέντρωση: Προγραμματίστε στοχευμένες ημερήσιες εξορμήσεις πεδίου (field sales days) στους κυριότερους δήμους.'` | **`territory_tip: 'Geographic Density: Schedule targeted field sales days in dominant municipalities.'`** | Eliminates clumsy Greek transliteration 'field sales days'. |
| 151 | `src/static/app.js` | L848 | Persona Card | `€${Math.round(p.turnover_median \|\| 200000).toLocaleString('el-GR')} διάμεσος τζίρος` | **`€${Math.round(p.turnover_median \|\| 200000).toLocaleString('en-US')} Median Revenue`** | Standard financial metric. |
| 152 | `src/static/app.js` | L849 | Persona Card | `sales_cycle: '7 - 14 ημέρες'` | **`sales_cycle: '7 - 14 Days'`** | Standard sales velocity. |
| 153 | `src/static/app.js` | L859 | Broken Accent / Sales Channel | `recommended_channel: 'Inside Sales με τηλεφωνική κλήση & 15-λεπτο Online Demo.'` | **`recommended_channel: 'Inside Sales: Discovery Call & 15-Minute Online Demo'`** | Fixes unaccented '15-λεπτο' and standardizes B2B sales motion. |
| 154 | `src/static/app.js` | L981 | Section Header | `Προτεραιοποίηση &amp; Θερμότητα Leads` | **`Lead Prioritization &amp; Funnel Warmth`** | Standard B2B sales funnel header. |
| 155 | `src/static/app.js` | L985 | Section Header | `Κορυφαίοι Δήμοι Εξόρμησης` | **`Top Municipalities for Field Sales`** | Clear field sales routing header. |
| 156 | `src/static/app.js` | L989 | Section Header | `Pitch Hook &amp; Οδηγός Πώλησης` | **`Sales Playbook &amp; Pitch Script`** | Standard sales enablement header. |
| 157 | `src/static/app.js` | L993 | Section Header | `Στατιστικό Lift vs Αγορά` | **`Statistical Lift vs Universe`** | Standard statistical benchmark. |
| 158 | `src/static/app.js` | L1750 | Toast Notification | `showToast('⚠️ Δεν υπάρχουν εγγραφές για εξαγωγή.', 'alert-circle')` | **`showToast('⚠️ No records available for export.', 'alert-circle')`** | Standard export toast. |
| 159 | `src/static/app.js` | L1786 | Toast Notification | `📥 Εξήχθησαν ${items.length.toLocaleString('el-GR')} leads σε CSV!` | **`📥 Exported ${items.length.toLocaleString('en-US')} prospects to CSV!`** | Standard export confirmation. |
| 160 | `src/static/app.js` | L2463 | Toast Notification | `🚫 Η επιχείρηση "${companyName}" εξαιρέθηκε από το pipeline.` | **`🚫 Company "${companyName}" excluded from pipeline.`** | Clear exclusion notification. |
| 161 | `src/static/app.js` | L2491 | Toast Notification | `✅ Η επιχείρηση επανήλθε στο pipeline στόχευσης.` | **`✅ Company restored to target pipeline.`** | Clear restoration notification. |
| 162 | `src/static/app.js` | L2553 | Toast Notification | `showToast('🔍 Εκτέλεση ζωντανού ελέγχου στο Γ.Ε.ΜΗ. & Google Places...', 'refresh-cw')` | **`showToast('🔍 Running live verification via GEMI & Google Places...', 'refresh-cw')`** | Clear verification action toast. |
| 163 | `src/static/app.js` | L2560 | Toast Notification | `⚠️ Εξακριβώθηκε: Η επιχείρηση έχει κλείσει (${v.details}). Αφαιρέθηκε από το pipeline.` | **`⚠️ Verified Closed: Enterprise is inactive (${v.details}). Removed from pipeline.`** | Clear verification toast. |
| 164 | `src/static/app.js` | L2562 | Toast Notification | `✅ Εξακριβώθηκε: Η επιχείρηση είναι ενεργή (${v.source}).` | **`✅ Verified Active: Enterprise confirmed operational (${v.source}).`** | Clear verification confirmation. |
| 165 | `src/static/app.js` | L2621 | Button State | `execBtn.innerHTML = '<i data-lucide="sparkles" class="w-4 h-4"></i><span>Εκτέλεση με Τοπικό Ollama</span>'` | **`execBtn.innerHTML = '<i data-lucide="sparkles" class="w-4 h-4"></i><span>Execute with Local Ollama</span>'`** | Consistent action button. |
| 166 | `src/static/app.js` | L2632 | Button State | `execBtn.innerHTML = '<i data-lucide="zap" class="w-4 h-4"></i><span>Εφαρμογή Grounded Fallback</span>'` | **`execBtn.innerHTML = '<i data-lucide="zap" class="w-4 h-4"></i><span>Apply Grounded Fallback</span>'`** | Consistent action button. |
| 167 | `src/static/app.js` | L2913 | Toast Notification | `showToast(isChecked ? '🟢 Νευρωνικά Embeddings Ενεργά' : '⚪ Αλγεβρικό SVD Baseline (Neural OFF)', 'cpu')` | **`showToast(isChecked ? '🟢 Neural Embeddings Active' : '⚪ Algebraic SVD Baseline (Neural OFF)', 'cpu')`** | Clean NLP toggle toast. |
| 168 | `src/service.py` | L50 | Service Param Spec | `{"value": "log_turnover", "label": "Λογαριθμική (ln τζίρου)"}` | **`{"value": "log_turnover", "label": "Log-Revenue [ln(Revenue + 1)]"}`** | Standard feature transform. |
| 169 | `src/service.py` | L51 | Service Param Spec | `{"value": "sqrt_turnover", "label": "Τετραγωνική ρίζα τζίρου"}` | **`{"value": "sqrt_turnover", "label": "Square Root Revenue [√Revenue]"}`** | Standard feature transform. |
| 170 | `src/service.py` | L52 | Service Param Spec | `{"value": "linear_turnover", "label": "Γραμμική"}` | **`{"value": "linear_turnover", "label": "Linear Revenue"}`** | Standard option. |
| 171 | `src/service.py` | L53 | Service Param Spec | `{"value": "uniform", "label": "Ισόβαρη (χωρίς τζίρο)"}` | **`{"value": "uniform", "label": "Uniform (Unweighted)"}`** | Standard option. |
| 172 | `src/service.py` | L60 | Service Param Spec | `"label": "Χρήση Μοντέλων Embeddings"` | **`"label": "Enable Neural Semantic Embeddings"`** | Professional NLP feature label. |
| 173 | `src/service.py` | L70 | Service Param Spec | `"label": "Στάθμιση τζίρου"` | **`"label": "Revenue Weighting Mode"`** | Professional parameter label. |
| 174 | `src/service.py` | L75 | Service Param Spec | `"label": "Βάρος ομοιότητας προφίλ"` | **`"label": "Cosine Similarity Trait Weight"`** | Purges leaked code variable `(w_cos)`. |
| 175 | `src/service.py` | L77 | Service Param Spec | `"label": "Βάρος στατιστικών χαρακτηριστικών"` | **`"label": "Empirical Bayes Trait Weight"`** | Purges leaked code variable `(w_eb)`. |
| 176 | `src/service.py` | L79 | Service Param Spec | `"label": "Βάρος γεωγραφικής εγγύτητας"` | **`"label": "Spatial Trait Weight"`** | Purges leaked code variable `(w_sp)`. |
| 177 | `src/service.py` | L85 | Service Param Spec | `"label": "Κατώφλι NO_FIT"` | **`"label": "NO_FIT Outlier Threshold"`** | Clear outlier threshold. |
| 178 | `src/server.py` | L86 | API Error Message | `return JSONResponse({"status": "error", "message": "Δεν επιλέχθηκε αρχείο."}, status_code=400)` | **`return JSONResponse({"status": "error", "message": "No file uploaded."}, status_code=400)`** | Standardized REST API error message in English. |
| 179 | `src/server.py` | L110 | API Error Message | `return JSONResponse({"status": "not_found", "message": "Η επιχείρηση δεν βρέθηκε."}, status_code=404)` | **`return JSONResponse({"status": "not_found", "message": "Company not found."}, status_code=404)`** | Standardized REST API 404 message. |
| 180 | `src/server.py` | L122 | API Error Message | `return JSONResponse({"status": "not_found", "message": "Το προφίλ δεν βρέθηκε."}, status_code=404)` | **`return JSONResponse({"status": "not_found", "message": "ICP Profile not found."}, status_code=404)`** | Standardized REST API 404 message. |
| 181 | `src/verifier.py` | L148 | Verification Message | `excl_map[vat_str].get("reason", "Επιβεβαιωμένη διακοπή εργασιών")` | **`excl_map[vat_str].get("reason", "Confirmed business termination / Inactive")`** | Standardizes status description in drawer and blacklist. |
| 182 | `src/verifier.py` | L200 | Verification Message | `"details": "Ενεργή εγγραφή στο Μητρώο Επιχειρήσεων (χωρίς καταγεγραμμένη διακοπή)."` | **`"details": "Active listing in Commercial Registry (no recorded cessation of business)."`** | Clear corporate registry status message. |
| 183 | `src/verifier.py` | L234 | Verification Message | `"OPERATIONAL": "Ενεργή λειτουργία (Επιβεβαιωμένο στο Google Maps)"` | **`"OPERATIONAL": "Operational (Verified on Google Maps)"`** | Standard Google Places operational status. |
| 184 | `src/verifier.py` | L235 | Verification Message | `"CLOSED_PERMANENTLY": "Μόνιμα κλειστό (Καταγεγραμμένη διακοπή στο Google Maps)"` | **`"CLOSED_PERMANENTLY": "Permanently Closed (Recorded on Google Maps)"`** | Standard Google Places status. |
| 185 | `src/verifier.py` | L236 | Verification Message | `"CLOSED_TEMPORARILY": "Προσωρινά κλειστό (Αναστολή λειτουργίας στο Google Maps)"` | **`"CLOSED_TEMPORARILY": "Temporarily Closed (Operations suspended on Google Maps)"`** | Standard Google Places status. |
| 186 | `src/verifier.py` | L269 | Verification Message | `"details": f"Αποτυχία επαλήθευσης Modulo 11 Checksum ΑΦΜ ({vat}) σύμφωνα με το πρότυπο Γ.Ε.ΜΗ./ΑΑΔΕ."` | **`"details": f"Failed Modulo 11 Checksum for VAT ({vat}) per GEMI / AADE specification."`** | Standard checksum error message. |
| 187 | `src/verifier.py` | L299 | Verification Message | `"details": f"Έγκυρο και πιστοποιημένο ΑΦΜ {vat_clean} στο σύστημα Γενικού Εμπορικού Μητρώου (Γ.Ε.ΜΗ.)."` | **`"details": f"Valid and verified VAT ID {vat_clean} in GEMI Commercial Registry."`** | Standard verification success message. |
| 188 | `src/llm_explainer.py` | L42 | Regulatory Disclosure Constant | `"🤖 «Περιγραφή παραγόμενη από Τοπικό AI (EU AI Act Art. 50)»"` | **`"🤖 «Generated by Local AI (EU AI Act Art. 50)»"`** | Replaces runtime AI disclosure constant `AI_DISCLOSURE_PREFIX_EL` (and docstring L25). |
| 189 | `src/llm_explainer.py` | L409 | Model Identifier | `model_name="Deterministic Grounded Greek Generator (Fallback)"` | **`model_name="Deterministic Grounded Engine (Local Fallback)"`** | Clean fallback engine identifier. |
| 190 | `src/service.py` | L569 | Model Notes | `"notes": ["100% Grounded Deterministic Explainer (Εγγύηση μηδενικού hallucination)"]` | **`"notes": ["100% Grounded Deterministic Explainer (Zero-Hallucination Guarantee)"]`** | Reattributed from `src/llm_explainer.py` to `src/service.py:L569`. |
| 191 | `src/static/index.html` | L328 | Financial Terminology | `Κατανομή Πελατολογίου &amp; Τζίρου` | **`Clientele &amp; Revenue Distribution`** | Purges colloquial 'Τζίρου' from section header. |
| 192 | `src/static/index.html` | L384 | Financial Terminology | `<span class="text-[10px] text-slate-500 font-normal">Τζίρος vs Όγκος</span>` | **`<span class="text-[10px] text-slate-500 font-normal">Revenue vs Account Volume</span>`** | Purges colloquial 'Τζίρος' from efficiency chart header. |
| 193 | `src/static/index.html` | L931 | Variable Encapsulation | `<span id="badge-embedding-status" class="text-[10px] px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/25 font-semibold">Section 5.2</span>` | **`<span id="badge-embedding-status" class="text-[10px] px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/25 font-semibold">Trait Contribution Breakdown</span>`** | Purges internal leaked variable `Section 5.2` from status badge. |
| 194 | `src/static/index.html` | L993 | Variable Encapsulation | `(Section 5.2 Baseline)` | **`(Trait Contribution Baseline)`** | Purges internal leaked variable `Section 5.2` from explanation banner. |
| 195 | `src/static/app.js` | L239 | Drawer Navigation | `<span>Καρτέλα</span>` | **`<span>Company Card</span>`** | Replaces colloquial Greek 'Καρτέλα' on map popup action button. |
| 196 | `src/static/app.js` | L537 | Drawer Navigation | `<span>Καρτέλα</span>` | **`<span>Company Card</span>`** | Replaces colloquial Greek 'Καρτέλα' on client popup action button. |
| 197 | `src/static/app.js` | L584 | Drawer Navigation | `<span>Καρτέλα</span>` | **`<span>Company Card</span>`** | Replaces colloquial Greek 'Καρτέλα' on lookalike popup action button. |
| 198 | `src/static/app.js` | L772 | Financial Terminology | `${revPct}% τζίρου` | **`${revPct}% revenue`** | Purges colloquial 'τζίρου' from profile card metrics. |
| 199 | `src/static/app.js` | L923 | Financial Terminology | `Μερίδιο Τζίρου` | **`Revenue Share`** | Purges colloquial 'Τζίρου' from drawer ICP distribution header. |
| 200 | `src/static/app.js` | L929 | Financial Terminology | `Διάμεσος Τζίρος` | **`Median Revenue`** | Purges colloquial 'Τζίρος' from drawer summary header. |
| 201 | `src/static/app.js` | L2124 | Drawer Action | `<span>Live Έλεγχος</span>` | **`<span>Live Verification</span>`** | Standardizes live external verification button. |
| 202 | `src/static/app.js` | L2133 | Drawer Action | `<span>Στον Χάρτη</span>` | **`<span>Focus on Map</span>`** | Standardizes map focus button in Company Inspector. |
| 203 | `src/static/app.js` | L2149 | Drawer Action | `<span>Γ.Ε.ΜΗ. Πύλη</span>` | **`<span>GEMI Portal</span>`** | Standardizes GEMI public directory launch button. |
| 204 | `src/static/app.js` | L2155 | Drawer Action | `<span>Εύρεση Γ.Ε.ΜΗ.</span>` | **`<span>Search GEMI</span>`** | Standardizes GEMI Google search button. |
| 205 | `src/static/app.js` | L2161 | Drawer Action | `<span>Επαναφορά</span>` | **`<span>Restore</span>`** | Standardizes company un-exclusion action button. |
| 206 | `src/static/app.js` | L2166 | Drawer Action | `<span>Ανενεργή</span>` | **`<span>Mark Inactive</span>`** | Standardizes company exclusion action button. |
| 207 | `src/static/app.js` | L2243 | Drawer Enrichment | `Συγκοινωνιακός Κόμβος` | **`Transit Hub Catchment`** | Standardizes accessibility badge header in Company Inspector. |
| 208 | `src/static/app.js` | L2245 | Drawer Enrichment | `${c.is_transit_hub_catchment ? 'Υψηλή Ροή (<500m)' : 'Τυπική Πρόσβαση'}` | **`${c.is_transit_hub_catchment ? 'High Footfall Catchment (<500m)' : 'Standard Access'}`** | Standardizes footfall catchment badge text. |
| 209 | `src/static/app.js` | L2251 | Drawer Enrichment | `Δυναμική Επέκτασης` | **`Network Expansion Velocity`** | Standardizes commercial growth badge header. |
| 210 | `src/static/app.js` | L2336 | Financial Terminology | `Τζίρος €` | **`Revenue: €`** | Purges colloquial 'Τζίρος' from proximity radar card list. |
| 211 | `src/server.py` | L96 | API Error Message | `return JSONResponse({"status": "error", "message": f"Σφάλμα επεξεργασίας: {exc}"}, status_code=500)` | **`return JSONResponse({"status": "error", "message": f"Processing error: {exc}"}, status_code=500)`** | Standardizes server-side processing error payload. |
| 212 | `src/server.py` | L138 | API Error Message | `return JSONResponse({"status": "not_found", "message": "Το προφίλ δεν βρέθηκε."}, status_code=404)` | **`return JSONResponse({"status": "not_found", "message": "ICP Profile not found."}, status_code=404)`** | Standardizes REST API 404 response for profile endpoint. |
| 213 | `src/llm_explainer.py` | L377 | Regulatory Disclosure Validator | `and "«Περιγραφή παραγόμενη από Τοπικό AI (EU AI Act Art. 50)»" in getattr(p, "ai_description_el", "")` | **`and "«Generated by Local AI (EU AI Act Art. 50)»" in getattr(p, "ai_description_el", "")`** | Updates runtime profile validator assertion to match standardized English AI badge. |

---

### 5.4 Architectural Blueprint for Post-Refactoring Internationalization (i18n)
To prevent hardcoded strings, eliminate orthographic accent errors, and support seamless runtime language toggling between English (canonical executive) and Greek (localized), the frontend architecture will deploy an enhanced zero-dependency i18n engine:

#### 1. Structured JSON Locale Dictionaries (`src/static/locales/`)
UI text is extracted into clean, hierarchical JSON bundles:
* `src/static/locales/en.json` (Canonical B2B SaaS English)
* `src/static/locales/el.json` (Polished Greek Terminology with Formal Accounting Terms)

**Canonical `src/static/locales/en.json` Structure:**
```json
{
  "nav": {
    "title": "ForceCRM Executive Intelligence",
    "subtitle": "B2B ICP Discovery & Lookalike Scoring Engine",
    "scoring_active": "Instant Scoring Active",
    "export": "Export",
    "analyst_tuning": "Analyst Tuning",
    "blacklist": "Excluded Accounts"
  },
  "kpi": {
    "active_clients": "Active Clientele",
    "total_revenue": "Total Portfolio Revenue",
    "discovered_icps": "Discovered ICPs",
    "lookalikes": "Lookalike Prospects",
    "arpa": "Avg Revenue Per Account (ARPA)",
    "stability": "Bootstrap Stability (ARI)"
  },
  "table": {
    "rank": "Rank & ICP Profile",
    "score": "Fit Score",
    "company": "Company & Primary KAD Activity",
    "location": "Location & VAT ID",
    "revenue": "Annual Revenue (€)",
    "actions": "Actions"
  },
  "drawer": {
    "title": "Company Inspector",
    "median_revenue": "Median Revenue",
    "focus_map": "Focus on Map",
    "live_verify": "Live Verification",
    "gemi_portal": "GEMI Portal",
    "restore": "Restore to Pipeline",
    "exclude": "Mark Inactive"
  },
  "toasts": {
    "exported_count": "📥 Exported {count} prospects to CSV!",
    "excluded_success": "🚫 Company "{name}" excluded from pipeline.",
    "restored_success": "✅ Company "{name}" restored to target pipeline."
  }
}
```

**Canonical `src/static/locales/el.json` Structure (Enforcing Formal Greek Accounting):**
```json
{
  "nav": {
    "title": "ForceCRM Executive Intelligence",
    "subtitle": "Μηχανή Ανακάλυψης ICP & Βαθμολόγησης Lookalikes",
    "scoring_active": "Άμεση Βαθμολόγηση Ενεργή",
    "export": "Εξαγωγή",
    "analyst_tuning": "Ρυθμίσεις Αναλυτή",
    "blacklist": "Αποκλεισμένες Επιχειρήσεις"
  },
  "kpi": {
    "active_clients": "Ενεργό Πελατολόγιο",
    "total_revenue": "Συνολικός Κύκλος Εργασιών",
    "discovered_icps": "Ανακαλυφθέντα ICPs",
    "lookalikes": "Προτεινόμενα Lookalikes",
    "arpa": "Μέση Αξία Πελάτη (ARPA)",
    "stability": "Σταθερότητα Bootstrap (ARI)"
  },
  "table": {
    "rank": "Κατάταξη & Προφίλ ICP",
    "score": "Βαθμός Ταύτισης",
    "company": "Επωνυμία & Κύρια Δραστηριότητα (ΚΑΔ)",
    "location": "Τοποθεσία & ΑΦΜ",
    "revenue": "Ετήσιος Κύκλος Εργασιών (€)",
    "actions": "Ενέργειες"
  },
  "drawer": {
    "title": "Καρτέλα Επιχείρησης",
    "median_revenue": "Διάμεσος Κύκλος Εργασιών",
    "focus_map": "Εστίαση στον Χάρτη",
    "live_verify": "Ζωντανός Έλεγχος",
    "gemi_portal": "Πύλη Γ.Ε.ΜΗ.",
    "restore": "Επαναφορά στο Pipeline",
    "exclude": "Επισήμανση ως Ανενεργή"
  },
  "toasts": {
    "exported_count": "📥 Εξήχθησαν {count} προτεινόμενοι πελάτες σε CSV!",
    "excluded_success": "🚫 Η επιχείρηση "{name}" εξαιρέθηκε από το pipeline.",
    "restored_success": "✅ Η επιχείρηση "{name}" επανήλθε στο pipeline."
  }
}
```

#### 2. Enhanced Translation Engine with Parameter Interpolation (`src/static/js/utils/i18n.js`)
The enhanced zero-dependency i18n engine supports parametric template interpolation, pluralization rules, and reactive store integration:

```javascript
let currentLocale = 'en';
const dictionaries = {};

export async function initI18n(store, locale = 'en') {
  currentLocale = locale;
  if (!dictionaries[locale]) {
    const res = await fetch(`/static/locales/${locale}.json`);
    dictionaries[locale] = await res.json();
  }
  document.documentElement.lang = locale;
  applyTranslations();

  // Reactive store hook: automatically re-render dynamic UI when locale toggles
  if (store) {
    store.subscribe('locale', async (newLocale) => {
      if (newLocale && newLocale !== currentLocale) {
        await initI18n(store, newLocale);
        store.dispatch('LOCALE_CHANGED', newLocale);
      }
    });
  }
}

/**
 * Parameter-interpolating translation helper.
 * Example: t('toasts.exported_count', { count: 25 }) -> "Exported 25 prospects to CSV!"
 */
export function t(keyPath, params = {}, fallback = '') {
  const keys = keyPath.split('.');
  let current = dictionaries[currentLocale];
  for (const k of keys) {
    if (!current || !(k in current)) {
      current = fallback || keyPath;
      break;
    }
    current = current[k];
  }

  if (typeof current !== 'string') return current;

  // Token interpolation: replaces {paramKey} with provided values
  return current.replace(/\{(\w+)\}/g, (match, paramName) => {
    return paramName in params ? params[paramName] : match;
  });
}

/**
 * Basic ICU-style pluralization helper.
 */
export function tp(keyPath, count, params = {}) {
  const rules = new Intl.PluralRules(currentLocale);
  const rule = rules.select(count); // 'one', 'other' (en) or 'one', 'few', 'many', 'other'
  const keyWithSuffix = `${keyPath}_${rule}`;
  const translated = t(keyWithSuffix, { count, ...params }, '');
  return translated !== keyWithSuffix ? translated : t(keyPath, { count, ...params });
}

export function applyTranslations() {
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    el.textContent = t(key, {}, el.textContent);
  });
  document.querySelectorAll('[data-i18n-attr]').forEach(el => {
    const [attr, key] = el.getAttribute('data-i18n-attr').split(':');
    el.setAttribute(attr, t(key, {}, el.getAttribute(attr)));
  });
}
```

#### 3. Locale-Aware Formatting Helpers via Native `Intl` API
Formatting for currency, percentages, and large numbers dynamically respects the active locale (`el-GR` vs `en-US`), ensuring proper decimal and thousands separator conventions:

```javascript
export const formatters = {
  currency: (val, locale = currentLocale) => new Intl.NumberFormat(locale === 'el' ? 'el-GR' : 'en-US', {
    style: 'currency',
    currency: 'EUR',
    maximumFractionDigits: 0
  }).format(val),

  percent: (val, locale = currentLocale) => new Intl.NumberFormat(locale === 'el' ? 'el-GR' : 'en-US', {
    style: 'percent',
    minimumFractionDigits: 1
  }).format(val),

  compactNumber: (val, locale = currentLocale) => new Intl.NumberFormat(locale === 'el' ? 'el-GR' : 'en-US', {
    notation: 'compact',
    compactDisplay: 'short'
  }).format(val)
};
```

---

## 6. Post-Refactoring Implementation & Migration Roadmap

### 6.1 Phased Rollout Plan Coordinated with Backend Deliverables
The frontend redesign is structured into five sequential phases to be executed immediately following the completion of ongoing backend service refactoring:

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                           POST-REFACTORING IMPLEMENTATION TIMELINE                                    │
├───────────────┬────────────────────────────────────────────────────────────────────────┬──────────────┤
│ Phase         │ Core Engineering Deliverables                                          │ Duration     │
├───────────────┼────────────────────────────────────────────────────────────────────────┼──────────────┤
│ **Phase 1**   │ • Design system tokens in `style.css` (surface lightness, tabular nums)│ Days 1 – 3   │
│ *Foundation*  │ • Directory scaffolding: `src/static/js/`, `src/static/locales/`       │              │
│               │ • Central reactive store (`store.js`) with equality & recursion guards │              │
│               │ • Native `<script type="importmap">` for unbundled ESM specifiers     │              │
│               │ • Enhanced i18n engine (`i18n.js`) with parametric interpolation       │              │
├───────────────┼────────────────────────────────────────────────────────────────────────┼──────────────┤
│ **Phase 2**   │ • Fixed 100vh cockpit container (`overflow: hidden`)                   │ Days 4 – 7   │
│ *Workspace*   │ • Collapsible executive summary strip (40px) subsuming 140px KPI strip │              │
│               │ • Consolidated 44px command toolbar (filters + search + density toggle)│              │
│               │ • Split-Screen container (`splitLayout.js`) with draggable divider     │              │
│               │ • Leaflet canvas manager (`mapView.js`) with single `L.divIcon` overlay│              │
│               │ • Safe pagination-aware map marker click navigation handler            │              │
├───────────────┼────────────────────────────────────────────────────────────────────────┼──────────────┤
│ **Phase 3**   │ • True slide-over contextual drawer (`drawer.js`) with 3 sub-tabs      │ Days 8 – 10  │
│ *Ergonomics*  │ • MiniSearch in-memory engine with Greek diacritic normalizer          │              │
│               │ • Floating Action Bar (`batchActions.js`) for multi-select operations  │              │
│               │ • Production keyboard manager with Greek input focus shield & Ctrl+K   │              │
├───────────────┼────────────────────────────────────────────────────────────────────────┼──────────────┤
│ **Phase 4**   │ • Execute 213-item prioritized string replacement catalog across files │ Days 11 – 12 │
│ *Localization*│ • Purge internal code variables (`(w_cos)`, `(w_eb)`, `(w_sp)`, `5.2`) │              │
│               │ • Replace colloquial `Τζίρος` with `Revenue` / `Κύκλος Εργασιών`       │              │
│               │ • Replace native `window.prompt()` with in-app exclusion dialog        │              │
│               │ • Deploy bilingual language toggle pill in global header               │              │
├───────────────┼────────────────────────────────────────────────────────────────────────┼──────────────┤
│ **Phase 5**   │ • WCAG 2.2 AA automated contrast audit (axe-core / Lighthouse)         │ Days 13 – 14 │
│ *QA & Launch* │ • Cross-resolution viewport testing (1366x768 up to 4K ultrawide)      │              │
│               │ • Full regression testing of ML hyperparameter re-scoring pipeline     │              │
└───────────────┴────────────────────────────────────────────────────────────────────────┴──────────────┘
```

### 6.2 Acceptance Verification Criteria & Quality Gates

#### R1: UI Ergonomics Verification Gate
- [ ] **Split-Screen Synchronization:** Prospecting data table and geospatial map operate simultaneously in a 55/45 split view. Clicking "Focus on Map" centers the pin on the adjacent map canvas without altering user scroll or pagination state.
- [ ] **Leaflet Canvas & Overlay Safety:** Active/hover pulsing halos use a dedicated single DOM overlay marker (`L.divIcon`). No `marker.getElement().style` mutations on canvas circle markers; zero canvas redraw freezes.
- [ ] **Safe Off-Page Marker Navigation:** Clicking a map marker for a prospect on another page updates `store.currentPage`, awaits DOM re-render, and smoothly scrolls the row into view before opening the drawer.
- [ ] **Contextual Slide-Over Drawer:** Company details render inside a right-anchored 480–550px drawer that leaves the data table visible and interactive. Pressing `ArrowUp`/`ArrowDown` navigates rows and updates the drawer dynamically.
- [ ] **Vertical Viewport Reclamation (450px+):** The Executive Hero Banner and 140px KPI strip collapse into a 40px executive strip, surfacing table rows and map markers immediately above the fold on 1366x768 laptop screens with 464px+ of usable height.
- [ ] **Fixed 100vh Cockpit Container:** The application is encapsulated in a fixed `100vh` flex container with `overflow: hidden` and internal scrollable panes, preventing full-document scrolling.
- [ ] **Keyboard Navigation & Greek Focus Shield:** Production keyboard manager with strict input focus shields preventing shortcut collisions while typing Greek search queries; `Ctrl+K` calls `preventDefault()` to prevent omnibox hijacking.
- [ ] **Sound Reactive Store & Import Maps:** Action-based reactive store with value equality checks preventing cyclic update stack overflow; native `<script type="importmap">` for unbundled ESM imports.
- [ ] **Column Pinning & Sorting:** Company identification column (`Rank + Name + Checkbox`) remains pinned to the left during horizontal scrolling. Clicking column headers sorts leads ascending and descending.
- [ ] **Multi-Select & Batch Actions:** Users can select multiple prospect rows; an elevated Floating Action Bar appears enabling batch CSV export, batch GEMI verification, and batch exclusions.
- [ ] **Sub-3ms Search Latency:** Text filtering across 2,000 lookalikes executes in <3ms using MiniSearch with Greek diacritic normalization without UI jank.
- [ ] **Thread-Blocking Prompt Elimination:** Browser-native `window.prompt()` is completely removed; exclusions execute via an in-app modal with predefined categories and an instant undo toast.
- [ ] **WCAG 2.2 AA Contrast Compliance:** All muted metadata text exceeds a 4.5:1 contrast ratio against the `#090d16` canvas; chart geometry exceeds 3:1 contrast.

#### R2: Localization & Terminology Verification Gate
- [ ] **Resolution of Broken Greek Accents:** All eight identified unaccented KPI cards (`Συνολικες Ευκαιριες`, `Μεσος Βαθμος Ταυτισης`, etc.) are replaced with standardized English executive metrics.
- [ ] **Canonical Terminology Compliance:** All 43 domain terms from the canonical glossary are consistently deployed across navigation, tables, cards, tooltips, and API responses.
- [ ] **Verified String Replacement Execution:** All 213 cataloged string replacements across `index.html` (138), `app.js` (49), `service.py` (11), `verifier.py` (7), `server.py` (5), and `llm_explainer.py` (3) are implemented.
- [ ] **File Bounds & Attribution Accuracy:** Item #190 is correctly attributed to `src/service.py:L569`; Item #28 targets `Ανά ICP` without trailing colon.
- [ ] **Runtime AI Disclosure Alignment:** Item #188 targets the runtime Python constant `AI_DISCLOSURE_PREFIX_EL` on `src/llm_explainer.py:L42` and validation assertion on L377.
- [ ] **Purge of Leaked Code Variables:** Internal variables (`w_cos`, `w_eb`, `w_sp`, `Section 5.2`) are completely purged from user-facing labels in glossary and catalog, standardizing to `Cosine Similarity Trait Weight`, `Empirical Bayes Trait Weight`, `Spatial Trait Weight`, and `Trait Contribution Breakdown`.
- [ ] **Purge of Colloquial Slang:** The informal term `Τζίρος` is completely replaced with `Revenue` / `Annual Revenue` across all financial and analytical views, and mapped to formal `Κύκλος Εργασιών` in the Greek locale bundle.
- [ ] **Enhanced Zero-Dependency i18n Engine:** JSON dictionaries (`en.json`, `el.json`) mounted via `src/static/js/utils/i18n.js` supporting parameter interpolation (`t('key', { param })`), pluralization rules, and locale-aware `Intl.NumberFormat` currency formatting.
- [ ] **Elastic Container Layout:** All buttons and navigation pills dynamically accommodate Greek text expansion (+20–30%) without clipping or layout distortion.
