// Vanilla JS, no build step. Same status vocabulary and confirm/skip
// semantics as the server-side duplicates_core primitives, and keyboard
// bindings on KeyboardEvent.code (physical position) rather than .key -- an
// alternate layout remaps .key to a different character before the browser
// sees it, which would otherwise leave a user on such a layout unable to
// use these shortcuts at all.
//
// The organising idea of this UI is the stage: one candidate visible at a
// time, every candidate laid out at the identical scene rectangle, flipped
// with no transition. See the direction contract at the top of index.html.

const state = {
  status: "idle",
  // The startup scan publishes groups as it finds them, so "scanning" and
  // "reviewable" are no longer opposites. streaming says which kind of scan
  // is running: one appending to the list below (decisions stay live), or a
  // rescan about to replace it wholesale (decisions locked).
  streaming: false,
  progress: null,   // last SSE frame, so the queue rail can show scan progress
  error: null,
  generation: 0,
  params: null,
  groups: [],       // summaries: {index, status, file_count, current_pick, suggested_idx, is_close_call}
  activeIndex: -1,
  detail: null,     // {index, status, current_pick, suggested_idx, is_close_call, paths, dimensions, sizes, size_labels, scores, metrics}
  eventSource: null,
  ledgerOpen: true,
};

// Stage view: which scene point is centred and whether we're inspecting at
// 1:1. Kept in normalized image coordinates so it survives a flip between
// candidates of different pixel dimensions -- inspecting the same corner of
// the photo in every file is the whole point of the stage.
const view = { zoom: false, u: 0.5, v: 0.5 };

const $ = (id) => document.getElementById(id);

async function api(path, opts = {}) {
  const res = await fetch(path, {
    ...opts,
    headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
  });
  if (!res.ok) {
    let message;
    try {
      const detail = (await res.json()).detail;
      // FastAPI/Pydantic validation failures (422) return `detail` as an
      // array of {msg, loc, ...} objects, not a string -- new Error(array)
      // stringifies to "[object Object]", which tells the user nothing.
      message = Array.isArray(detail)
        ? detail.map((d) => (d && d.msg) || JSON.stringify(d)).join("; ")
        : detail;
    } catch { message = res.statusText; }
    // Callers need the code, not just the prose: 409 means this tab's
    // generation is stale and the view has to be reloaded, not toasted.
    const err = new Error(message || `HTTP ${res.status}`);
    err.status = res.status;
    throw err;
  }
  return res.status === 204 ? null : res.json();
}

function showToast(message, isError = false) {
  const el = $("toast");
  el.textContent = message;
  el.className = isError ? "toast is-error" : "toast";
  el.hidden = false;
  clearTimeout(showToast._t);
  showToast._t = setTimeout(() => { el.hidden = true; }, 5000);
}

function fmtInt(n) {
  return String(n).replace(/\B(?=(\d{3})+(?!\d))/g, " ");
}

// ---------------------------------------------------------------------------
// State loading
// ---------------------------------------------------------------------------

async function refreshState() {
  const data = await api("/api/state");
  state.status = data.status;
  state.streaming = !!data.streaming;
  state.error = data.error;
  state.generation = data.generation;
  state.params = data.params;
  state.groups = data.groups;
  renderQueue();
  renderScope();
  renderNotices();
  renderAppState();
  return data;
}

// Monotonic per-call token: if group B is clicked before group A's fetch
// resolves, A's response must not clobber B's once B has already landed.
let loadGroupToken = 0;

async function loadGroup(i) {
  if (i < 0 || i >= state.groups.length) return;
  const token = ++loadGroupToken;
  try {
    const data = await api(`/api/group/${i}`);
    if (token !== loadGroupToken) return; // superseded by a newer loadGroup
    state.activeIndex = i;
    state.detail = data;
    view.zoom = false;
    view.u = 0.5;
    view.v = 0.5;
    renderQueue();
    renderGroup();
    prefetchNextGroup();
  } catch (e) {
    if (token === loadGroupToken) showToast(`Couldn't load group ${i + 1}: ${e.message}`, true);
  }
}

// The hot path is confirm -> advance -> next group, so the next pending
// group's first stage render is warmed while this one is being decided.
function prefetchNextGroup() {
  const n = state.groups.length;
  for (let off = 1; off <= n; off++) {
    const j = (state.activeIndex + off) % n;
    if (j === state.activeIndex) break;
    if (state.groups[j].status === "pending") {
      new Image().src = `/api/stage/${j}/${state.groups[j].current_pick}?g=${state.generation}`;
      return;
    }
  }
}

function reviewCounts() {
  const confirmed = state.groups.filter((g) => g.status === "confirmed").length;
  const skipped = state.groups.filter((g) => g.status === "skipped").length;
  return { confirmed, skipped, pending: state.groups.length - confirmed - skipped, total: state.groups.length };
}

// True when a scan is running but its groups are already reviewable: the
// startup scan appends as it goes, so once the first group has landed the
// full-screen progress panel would be hiding work the user can already do.
function reviewableDuringScan() {
  return state.streaming && state.groups.length > 0;
}

function appState() {
  if (state.status === "scanning" && !reviewableDuringScan()) return "scanning";
  if (state.status === "error" && !state.groups.length) return "error";
  if (!state.groups.length) return "empty";
  return state.detail ? "group" : "empty";
}

function renderAppState() {
  const mode = appState();
  $("app").dataset.state = mode;
  $("scanning").hidden = mode !== "scanning";

  // Nothing from the last group may survive into a scan, an error or an
  // empty result: a photo still sitting on the stage under a progress
  // readout reads as "this is what I'm working on", which is false.
  if (mode !== "group") {
    $("stage-frame").innerHTML = "";
    stageImgs.length = 0;
    view.zoom = false;
    $("stage").dataset.zoom = "off";
    $("hud-group").textContent = "";
    $("hud-zoom").textContent = "";
    renderDecision();
  }

  const msg = $("stage-message");
  if (mode === "group" || mode === "scanning") {
    msg.hidden = true;
  } else {
    msg.hidden = false;
    msg.innerHTML = "";
    const h = document.createElement("h2");
    const p = document.createElement("p");
    if (mode === "error") {
      h.textContent = "That scan didn't run";
      p.textContent = `${state.error} — open the directory field above and try another path.`;
    } else if (state.status === "ready" && !state.groups.length) {
      h.textContent = "No duplicates here";
      p.textContent = `Nothing in ${state.params ? state.params.directory : "this directory"} hashed close enough to anything else to be a duplicate. Raise the threshold or scan somewhere else from the directory field above.`;
    } else {
      h.textContent = "Nothing selected";
      p.textContent = "Pick a group from the list to review it.";
    }
    msg.appendChild(h);
    msg.appendChild(p);
  }
  updateActionButtons();
  updateDocumentTitle();
}

function updateDocumentTitle() {
  if (!state.groups.length) {
    document.title = "Duplicate review";
    return;
  }
  const { pending, total } = reviewCounts();
  document.title = pending === 0 ? "Done — duplicate review" : `${pending}/${total} left — duplicate review`;
}

// ---------------------------------------------------------------------------
// Command bar + notices
// ---------------------------------------------------------------------------

function renderScope() {
  const p = state.params;
  if (!p) return;
  $("scope-dir").textContent = p.directory;
  $("scope-dir").title = p.directory;
  const bits = [`threshold ${p.threshold}/64`];
  bits.push(p.recursive ? "recursive" : "top level only");
  if (p.dry_run) bits.push("dry run");
  $("scope-meta").textContent = bits.join(" · ");
}

function notice(kind, tag, text) {
  const el = document.createElement("div");
  el.className = `notice notice-${kind}`;
  const strong = document.createElement("strong");
  strong.textContent = tag;
  const span = document.createElement("span");
  span.textContent = text;
  el.appendChild(strong);
  el.appendChild(span);
  return el;
}

// Persistent, not toasts: scan failure, dry-run mode and "everything is
// reviewed" all have to stay on screen rather than time out.
function renderNotices() {
  const host = $("notices");
  host.innerHTML = "";
  if (state.status === "error" && state.error) {
    host.appendChild(notice("error", "Scan failed", state.error));
  }
  if (state.params && state.params.dry_run) {
    host.appendChild(notice("dry", "Dry run", "Confirm and skip update this review only. No file will be moved."));
  }
  const { confirmed, skipped, pending, total } = reviewCounts();
  if (total && pending === 0) {
    host.appendChild(notice(
      "done",
      "All reviewed",
      `${total} group${total === 1 ? "" : "s"} — ${confirmed} kept, ${skipped} skipped. Nothing is half-done; you can close this tab.`,
    ));
  }
}

// ---------------------------------------------------------------------------
// Queue rail
// ---------------------------------------------------------------------------

const STATUS_WORD = { pending: "Pending", confirmed: "Kept", skipped: "Skipped" };

function renderQueue() {
  const { confirmed, skipped, pending, total } = reviewCounts();
  $("queue-left").textContent = total ? pending : "";
  $("queue-total").textContent = total ? `left of ${total} group${total === 1 ? "" : "s"}` : "No groups yet";
  $("meter-kept").style.width = total ? `${(100 * confirmed) / total}%` : "0";
  $("meter-skipped").style.width = total ? `${(100 * skipped) / total}%` : "0";
  $("queue-meter").setAttribute("aria-label", `${confirmed} kept, ${skipped} skipped, ${pending} left`);
  // While a streaming scan is still running the queue is incomplete, and the
  // full-screen progress panel is gone as soon as the first group lands --
  // so the tally line carries the phase readout instead of vanishing.
  const p = state.progress;
  $("queue-tally").textContent = state.status === "scanning" && state.streaming
    ? `Still scanning — ${p && p.total > 0 ? `${fmtInt(p.done)} / ${fmtInt(p.total)}` : "…"}`
    : (total ? `${confirmed} kept · ${skipped} skipped` : "");

  const list = $("queue-list");
  list.innerHTML = "";
  if (!total) {
    const li = document.createElement("li");
    li.className = "queue-empty";
    li.textContent = state.status === "scanning" ? "Scanning…" : "Nothing to review.";
    list.appendChild(li);
    return;
  }

  state.groups.forEach((g, i) => {
    const li = document.createElement("li");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "q-item"
      + (i === state.activeIndex ? " is-active" : "")
      + (g.status !== "pending" ? " is-done" : "");
    if (i === state.activeIndex) btn.setAttribute("aria-current", "true");

    const idx = document.createElement("span");
    idx.className = "q-idx";
    idx.textContent = String(i + 1).padStart(2, "0");

    // Shape carries the status as well as colour: an open square, a filled
    // square and a bar are three different silhouettes at 10px.
    const dot = document.createElement("span");
    dot.className = `q-dot q-dot-${g.status}`;

    const label = document.createElement("span");
    label.className = "q-label";
    label.textContent = g.status === "confirmed"
      ? `${g.file_count} files · kept [${g.current_pick + 1}]`
      : `${g.file_count} files`;

    btn.append(idx, dot, label);

    // No close-call flag in the queue on purpose: most groups in a real scan
    // are close calls, so flagging them here marks nearly every row and the
    // mark stops meaning anything. It stays where it can be acted on -- the
    // ledger note on the group being looked at -- and in the row's tooltip.
    btn.title = `Group ${i + 1} · ${STATUS_WORD[g.status]} · ${g.file_count} files`
      + (g.is_close_call ? " · close call: the top two scored nearly the same" : "");
    btn.addEventListener("click", () => loadGroup(i));
    li.appendChild(btn);
    list.appendChild(li);
  });

  const active = list.querySelector(".is-active");
  if (active) active.scrollIntoView({ block: "nearest", inline: "nearest" });
}

// ---------------------------------------------------------------------------
// Stage
// ---------------------------------------------------------------------------

const stageImgs = [];   // one <img> per candidate, all laid out identically

function buildStage() {
  const frame = $("stage-frame");
  frame.innerHTML = "";
  stageImgs.length = 0;
  const d = state.detail;
  if (!d) return;
  d.paths.forEach((path, j) => {
    const img = document.createElement("img");
    img.className = "stage-img";
    img.id = `stage-img-${j}`;
    // Without this, a mousedown-drag on the image starts the browser's own
    // native image-drag gesture instead of ours -- it wins the pointer
    // stream, so pan stalls the instant it begins.
    img.draggable = false;
    // Only the visible layer is exposed: the others are the same photo at
    // other sizes, stacked underneath, and announcing all six as images is
    // noise no one can act on.
    img.alt = j === d.current_pick ? `${path} — the file you're keeping` : "";
    img.setAttribute("aria-hidden", j === d.current_pick ? "false" : "true");
    img.decoding = "async";
    img.src = `/api/stage/${d.index}/${j}?g=${state.generation}`;
    img.dataset.full = "0";
    frame.appendChild(img);
    stageImgs.push(img);
  });
}

function stageBox() {
  const frame = $("stage-frame");
  return { w: frame.clientWidth, h: frame.clientHeight };
}

function dimsOf(j) {
  const d = state.detail;
  const dim = (d.dimensions && d.dimensions[j]) || [0, 0];
  return { w: dim[0] > 0 ? dim[0] : 1, h: dim[1] > 0 ? dim[1] : 1 };
}

// The inspect scale shows the group's largest file at its true 1:1 pixels;
// every other candidate is scaled to the same scene rectangle, so a smaller
// export renders visibly upscaled. That comparison -- same framing, same
// scene, real pixels against interpolated ones -- is the verdict this whole
// screen exists to deliver.
function inspectMaxWidth() {
  const d = state.detail;
  if (!d) return 0;
  return Math.max(...d.paths.map((_, j) => dimsOf(j).w));
}

function scaleFor(j, box) {
  const { w, h } = dimsOf(j);
  const contain = Math.min(box.w / w, box.h / h);
  if (!view.zoom) return contain;
  return Math.max(contain, inspectMaxWidth() / w);
}

function layoutStage() {
  const d = state.detail;
  if (!d || !stageImgs.length) return;
  const box = stageBox();
  if (!box.w || !box.h) return;

  stageImgs.forEach((img, j) => {
    const { w, h } = dimsOf(j);
    const s = scaleFor(j, box);
    const dw = w * s;
    const dh = h * s;
    const x = dw <= box.w ? (box.w - dw) / 2 : clamp(box.w / 2 - view.u * dw, box.w - dw, 0);
    const y = dh <= box.h ? (box.h - dh) / 2 : clamp(box.h / 2 - view.v * dh, box.h - dh, 0);
    img.style.width = `${dw}px`;
    img.style.height = `${dh}px`;
    img.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
    img.classList.toggle("is-active", j === d.current_pick);
    // Only fetch the original once the stage render (STAGE_MAX_SIDE) is
    // actually being magnified past its own pixels -- /api/full re-encodes
    // HEIC on every request and can be tens of megabytes.
    if (view.zoom && img.dataset.full === "0" && dw > 1600 * 1.05) upgradeToFullRes(img, d.index, j);
  });

  const zoomable = inspectMaxWidth() > box.w;
  $("stage").dataset.zoomable = zoomable ? "yes" : "no";
  $("stage").dataset.zoom = view.zoom ? "on" : "off";
  renderHud();
}

function clamp(v, lo, hi) { return Math.min(Math.max(v, lo), hi); }

function upgradeToFullRes(img, i, j) {
  img.dataset.full = "1";
  const pre = new Image();
  // Geometry comes from the API's pixel dimensions, not from the loaded
  // bitmap, so swapping in a bigger source can't shift the framing.
  pre.onload = () => { img.src = pre.src; };
  pre.onerror = () => { img.dataset.full = "0"; };
  pre.src = `/api/full/${i}/${j}?g=${state.generation}`;
}

function renderHud() {
  const d = state.detail;
  if (!d) { $("hud-group").textContent = ""; $("hud-zoom").textContent = ""; return; }
  $("hud-group").textContent = `Group ${d.index + 1} of ${state.groups.length} · file ${d.current_pick + 1} of ${d.paths.length}`;

  if (!view.zoom) {
    $("hud-zoom").textContent = $("stage").dataset.zoomable === "no"
      ? "Fits at full pixels"
      : "Click to inspect 1:1 · Z";
    return;
  }
  const factor = inspectMaxWidth() / dimsOf(d.current_pick).w;
  const pan = "drag or shift+arrows to pan";
  $("hud-zoom").textContent = factor > 1.02
    ? `Inspecting 1:1 · this file upscaled ${factor.toFixed(1)}× · ${pan}`
    : `Inspecting 1:1 · true pixels · ${pan}`;
}

// Keyboard pan, one tenth of the visible frame per press. Full keyboard
// operation is a durable product constraint, and "drag to pan" was the one
// affordance in this UI a pointer alone could reach.
function panBy(dirU, dirV) {
  const d = state.detail;
  if (!d || !view.zoom) return;
  const box = stageBox();
  const { w, h } = dimsOf(d.current_pick);
  const s = scaleFor(d.current_pick, box);
  view.u = clamp(view.u + dirU * 0.1 * (box.w / (w * s)), 0, 1);
  view.v = clamp(view.v + dirV * 0.1 * (box.h / (h * s)), 0, 1);
  layoutStage();
}

function setZoom(on, u, v) {
  if (on && $("stage").dataset.zoomable === "no") return;
  view.zoom = on;
  if (u !== undefined) { view.u = clamp(u, 0, 1); view.v = clamp(v, 0, 1); }
  layoutStage();
}

function pointToScene(ev) {
  const d = state.detail;
  const box = stageBox();
  const rect = $("stage-frame").getBoundingClientRect();
  const { w, h } = dimsOf(d.current_pick);
  const s = scaleFor(d.current_pick, box);
  const dw = w * s;
  const dh = h * s;
  const x = dw <= box.w ? (box.w - dw) / 2 : clamp(box.w / 2 - view.u * dw, box.w - dw, 0);
  const y = dh <= box.h ? (box.h - dh) / 2 : clamp(box.h / 2 - view.v * dh, box.h - dh, 0);
  return {
    u: clamp((ev.clientX - rect.left - x) / dw, 0, 1),
    v: clamp((ev.clientY - rect.top - y) / dh, 0, 1),
  };
}

function attachStageHandlers() {
  const stage = $("stage");
  let drag = null;

  stage.addEventListener("pointerdown", (ev) => {
    if (!state.detail || ev.button !== 0) return;
    if (view.zoom) ev.preventDefault(); // stop the native image-drag from stealing the gesture
    drag = { x: ev.clientX, y: ev.clientY, moved: false, u: view.u, v: view.v };
    // Captured unconditionally, matching `drag` above: if capture were only
    // taken while already zoomed, a press-and-release outside #stage while
    // unzoomed never reaches `endDrag` (it's bound to #stage), stranding
    // `drag` non-null. Zoom later turning on some other way (the Z key, e.g.)
    // then reads that stale `drag` and pans on bare mouse movement, no button
    // held.
    stage.setPointerCapture(ev.pointerId);
    if (view.zoom) stage.classList.add("is-panning");
  });
  stage.addEventListener("dragstart", (ev) => ev.preventDefault());

  stage.addEventListener("pointermove", (ev) => {
    if (!drag || !view.zoom) return;
    // Chrome can drop the pointerup entirely when the button is released
    // outside the browser window while the pointer is captured -- confirmed
    // by reproducing this in Chrome (including a fresh incognito window, so
    // not stale JS) while the same drag is fine in Safari. `buttons` is
    // re-read from the OS on every event rather than tracked from past
    // events, so it still reflects a real release even when the discrete
    // pointerup never fired; trust it over the stale `drag`.
    if ((ev.buttons & 1) === 0) {
      drag = null;
      stage.classList.remove("is-panning");
      if (stage.hasPointerCapture && stage.hasPointerCapture(ev.pointerId)) stage.releasePointerCapture(ev.pointerId);
      return;
    }
    const box = stageBox();
    const { w, h } = dimsOf(state.detail.current_pick);
    const s = scaleFor(state.detail.current_pick, box);
    const dx = ev.clientX - drag.x;
    const dy = ev.clientY - drag.y;
    if (Math.abs(dx) > 3 || Math.abs(dy) > 3) drag.moved = true;
    view.u = clamp(drag.u - dx / (w * s), 0, 1);
    view.v = clamp(drag.v - dy / (h * s), 0, 1);
    layoutStage();
  });

  const endDrag = (ev) => {
    if (!drag) return;
    stage.classList.remove("is-panning");
    if (stage.hasPointerCapture && stage.hasPointerCapture(ev.pointerId)) stage.releasePointerCapture(ev.pointerId);
    const wasDrag = drag.moved;
    drag = null;
    if (wasDrag || !state.detail) return;
    if (view.zoom) setZoom(false);
    else { const p = pointToScene(ev); setZoom(true, p.u, p.v); }
  };
  stage.addEventListener("pointerup", endDrag);
  stage.addEventListener("pointercancel", () => { drag = null; stage.classList.remove("is-panning"); });

  new ResizeObserver(() => layoutStage()).observe($("stage-frame"));
}

// ---------------------------------------------------------------------------
// Candidate switcher
// ---------------------------------------------------------------------------

// Truncates in the middle, never at the tail. Exports of one photo differ at
// the end of the name ("… copy 2.jpg" against "… copy.jpg"), so an ellipsis
// that eats the tail can render two different files as identical text.
function shortName(name, max) {
  if (name.length <= max) return name;
  const head = Math.ceil((max - 1) * 0.5);
  return `${name.slice(0, head)}…${name.slice(head + 1 - max)}`;
}

// How much name a tab can hold before the two-line clamp starts eating the
// tail anyway: tabs share the strip, so more candidates means less room.
function nameBudget(count) {
  if (count >= 5) return 34;
  if (count >= 3) return 52;
  return 90;
}

function renderSwitcher() {
  const host = $("switcher");
  // Activating a tab by keyboard rebuilds the strip under the focused
  // element; remember where focus was so it lands on the same tab again
  // instead of dropping to <body>.
  const focusedTab = host.contains(document.activeElement) ? document.activeElement.id : null;
  host.innerHTML = "";
  const d = state.detail;
  if (!d) return;
  const best = Math.max(...d.scores);
  const budget = nameBudget(d.paths.length);

  d.paths.forEach((path, j) => {
    const active = j === d.current_pick;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "cand" + (active ? " is-active" : "");
    btn.id = `cand-${j}`;
    btn.setAttribute("role", "tab");
    btn.setAttribute("aria-selected", String(active));
    // The stage is the one panel every tab controls -- it's the same box
    // showing a different file, which is exactly the tab/tabpanel relation.
    btn.setAttribute("aria-controls", "stage");
    btn.tabIndex = active ? 0 : -1;

    const top = document.createElement("span");
    top.className = "cand-top";
    const idx = document.createElement("span");
    idx.className = "cand-idx";
    idx.textContent = String(j + 1);
    const name = document.createElement("span");
    name.className = "cand-name";
    name.textContent = shortName(path, budget);
    top.append(idx, name);

    // Facts and the keeping/suggested tag share a line so the filename above
    // them gets the tab's full width -- with six candidates it needs it.
    const meta = document.createElement("span");
    meta.className = "cand-meta";
    const facts = document.createElement("span");
    facts.className = "cand-facts";
    const dim = d.dimensions[j];
    facts.textContent = `${dim[0]}×${dim[1]} · ${d.size_labels[j]}`;
    meta.appendChild(facts);
    if (active) {
      const tag = document.createElement("span");
      tag.className = "cand-tag";
      tag.textContent = "Keeping";
      meta.appendChild(tag);
    } else if (j === d.suggested_idx) {
      const mark = document.createElement("span");
      mark.className = "cand-mark";
      mark.textContent = "Suggested";
      meta.appendChild(mark);
    }

    const score = document.createElement("span");
    score.className = "cand-score";
    const track = document.createElement("span");
    track.className = "cand-score-track";
    const fill = document.createElement("span");
    fill.className = "cand-score-fill";
    fill.style.width = `${clamp(d.scores[j], 0, 1) * 100}%`;
    track.appendChild(fill);
    const num = document.createElement("span");
    num.textContent = d.scores[j].toFixed(2);
    score.append(track, num);

    btn.append(top, meta, score);
    btn.title = `${path} — quality score ${d.scores[j].toFixed(3)}`
      + (d.scores[j] === best ? " (top scored)" : "")
      + `. Press ${j + 1} to keep this one.`;
    btn.addEventListener("click", () => pick(j));
    host.appendChild(btn);
  });

  if (focusedTab && $(focusedTab)) $(focusedTab).focus({ preventScroll: true });
}

// ---------------------------------------------------------------------------
// Measurement ledger
// ---------------------------------------------------------------------------

// Winning index/indices for one row. Values are the exact strings the server
// formatted (METRIC_ROWS' lambdas) -- always a plain decimal or "n/a", so
// parseFloat is enough and "n/a" drops out as NaN. Needs at least 2 real
// values: BRISQUE/NIQE can be "n/a" for one file and a number for another,
// and a lone value has nothing to have won against.
function bestIndices(direction, values) {
  if (!direction) return [];
  const nums = values.map((v) => parseFloat(v));
  const finite = nums.filter((n) => !Number.isNaN(n));
  if (finite.length < 2) return [];
  const best = direction > 0 ? Math.max(...finite) : Math.min(...finite);
  return nums.flatMap((n, i) => (n === best ? [i] : []));
}

function renderLedger() {
  const d = state.detail;
  const thead = document.querySelector("#metrics-table thead");
  const tbody = document.querySelector("#metrics-table tbody");
  thead.innerHTML = "";
  tbody.innerHTML = "";
  if (!d) return;

  const headRow = document.createElement("tr");
  const corner = document.createElement("th");
  corner.scope = "col";
  // Not "Metric": dimensions and file size are reference only, and quality
  // score is the composite they don't feed.
  corner.textContent = "Measurement";
  headRow.appendChild(corner);
  d.paths.forEach((path, j) => {
    const th = document.createElement("th");
    th.scope = "col";
    if (j === d.current_pick) th.classList.add("col-pick");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "col-head-btn";
    btn.textContent = j === d.current_pick ? `${j + 1} · keeping` : String(j + 1);
    btn.title = `${path} — click to keep this one instead`;
    btn.addEventListener("click", () => pick(j));
    th.appendChild(btn);
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);

  let prevWasReference = null;
  d.metrics.forEach(({ label, values, direction, kind }) => {
    const isScore = kind === "score";
    const isReference = kind === "reference";
    const winners = isScore ? [] : bestIndices(direction, values);

    const tr = document.createElement("tr");
    if (isReference) tr.classList.add("row-reference");
    else if (prevWasReference) tr.classList.add("row-first-scored");
    if (isScore) tr.classList.add("row-score");
    prevWasReference = isReference;

    const th = document.createElement("th");
    th.scope = "row";
    th.textContent = label;
    tr.appendChild(th);

    values.forEach((v, j) => {
      const td = document.createElement("td");
      if (j === d.current_pick) td.classList.add("col-pick");
      if (isScore) {
        const cell = document.createElement("span");
        cell.className = "score-cell";
        const track = document.createElement("span");
        track.className = "score-track";
        track.setAttribute("aria-hidden", "true");
        const fill = document.createElement("span");
        fill.className = "score-fill";
        fill.style.width = `${clamp(parseFloat(v) || 0, 0, 1) * 100}%`;
        track.appendChild(fill);
        const num = document.createElement("span");
        num.className = "score-num";
        num.textContent = v;
        cell.append(track, num);
        td.appendChild(cell);
      } else {
        td.textContent = v;
        if (winners.includes(j)) {
          td.classList.add("is-best");
          td.title = "Best value in this row";
        } else if (v === "n/a") {
          td.classList.add("is-na");
          // Names the state, not a cause: "n/a" means either the optional
          // package isn't installed or the measurement failed on this one
          // file, and the UI can't tell which.
          td.title = "Not measured for this file — either the optional metric isn't installed, or it failed on this image. See Help.";
        }
      }
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });

  $("ledger-note").textContent = d.is_close_call
    ? "Close call — the top two scored nearly the same"
    : "";
  $("ledger-note").className = d.is_close_call ? "ledger-note is-close" : "ledger-note";
}

function setLedgerOpen(open) {
  state.ledgerOpen = open;
  $("ledger-scroll").hidden = !open;
  $("ledger-toggle").setAttribute("aria-expanded", String(open));
  layoutStage();
}

// ---------------------------------------------------------------------------
// Decision bar
// ---------------------------------------------------------------------------

function renderDecision() {
  const el = $("consequence");
  el.innerHTML = "";
  const d = state.detail;
  if (!d) {
    if (state.status === "scanning" && !state.streaming) el.textContent = "Scanning — confirm and skip resume when it finishes.";
    else if (state.status === "scanning") el.textContent = "Still scanning — groups appear as they're found, and you can decide them now.";
    else el.textContent = state.groups.length ? "No group selected." : "";
    return;
  }
  const nMoved = d.paths.length - 1;
  const dest = state.params ? state.params.dest : "the destination folder";
  const keptName = d.paths[d.current_pick];

  const add = (text, cls) => {
    const s = document.createElement(cls ? "span" : "span");
    if (cls) s.className = cls;
    s.textContent = text;
    el.appendChild(s);
  };
  const strong = (text) => {
    const s = document.createElement("strong");
    s.textContent = text;
    el.appendChild(s);
  };

  // Under --dry-run nothing is ever moved, so the sentence the user reads
  // before acting has to be conditional too -- stating "moves 5 files" under
  // a banner that says no file will be moved is the wrong voice in the one
  // place that describes a destructive action.
  const dry = !!(state.params && state.params.dry_run);
  const files = `${nMoved} file${nMoved === 1 ? "" : "s"}`;

  if (d.status === "confirmed") {
    add(dry ? "Marked kept (dry run) " : "Kept ");
    strong(`[${d.current_pick + 1}] ${keptName}`);
    add(dry ? `. ${files} would have moved to ${dest}.` : `. ${files} moved to ${dest} — undo with Skip group.`);
    if (d.needs_reapply) add("  Pick changed since confirming — confirm again to apply it.", "warn");
  } else if (d.status === "skipped") {
    add("Skipped — every file left where it is. Confirming would keep ");
    strong(`[${d.current_pick + 1}] ${keptName}`);
    add(` and move ${nMoved} other${nMoved === 1 ? "" : "s"}.`);
  } else {
    add(dry ? "Confirm would keep " : "Confirm keeps ");
    strong(`[${d.current_pick + 1}] ${keptName}`);
    if (!nMoved) add(dry ? " and move nothing else." : " and moves nothing else.");
    else add(dry ? ` and would move ${files} to ${dest} — dry run, so nothing actually moves.` : ` and moves ${files} to ${dest}.`);
    if (d.current_pick !== d.suggested_idx) {
      add(`  Suggested was [${d.suggested_idx + 1}] ${d.paths[d.suggested_idx]}.`);
    }
  }
}

// A dimmed control that doesn't say why reads as broken rather than as "not
// yet". Help is never disabled: the one control that explains the app has to
// work when there's nothing to review.
const NO_GROUP_REASON = "Nothing to act on yet — pick a group in the list first.";

function updateActionButtons() {
  const hasGroup = !!state.detail && (state.status !== "scanning" || state.streaming);
  for (const id of ["btn-confirm", "btn-skip", "btn-open"]) {
    const btn = $(id);
    if (!btn.dataset.enabledTitle) btn.dataset.enabledTitle = btn.title;
    btn.disabled = !hasGroup;
    btn.title = hasGroup
      ? btn.dataset.enabledTitle
      : (state.status === "scanning" ? "Locked while a scan is running." : NO_GROUP_REASON);
  }
  // Skip does three different things depending on where the group stands,
  // and one of them moves files back out of the destination folder. The
  // label has to say which -- "Skip group" on a confirmed group described
  // the opposite of what pressing it does.
  const skip = $("btn-skip");
  const status = state.detail ? state.detail.status : "pending";
  if (hasGroup && status === "confirmed") {
    skip.textContent = "Undo keep";
    skip.dataset.enabledTitle = "Move the files this group already moved back where they came from, and mark it skipped. Shortcut: Delete, Backspace or S";
  } else if (hasGroup && status === "skipped") {
    skip.textContent = "Un-skip group";
    skip.dataset.enabledTitle = "Put this group back in the queue as pending. Shortcut: Delete, Backspace or S";
  } else {
    skip.textContent = "Skip group";
    skip.dataset.enabledTitle = "Leave every file in this group where it is and move on. Shortcut: Delete, Backspace or S";
  }
  if (hasGroup) skip.title = skip.dataset.enabledTitle;
}

// ---------------------------------------------------------------------------
// Rendering entry points
// ---------------------------------------------------------------------------

function renderGroup() {
  buildStage();
  renderSwitcher();
  renderLedger();
  renderDecision();
  renderAppState();
  layoutStage();
}

// Re-render everything that depends on the active group without rebuilding
// the stage's <img> layers -- rebuilding them on every pick would re-decode
// the images and destroy the instant flip.
function renderPickChange() {
  const d = state.detail;
  if (d) {
    stageImgs.forEach((img, j) => {
      const active = j === d.current_pick;
      img.classList.toggle("is-active", active);
      img.alt = active ? `${d.paths[j]} — the file you're keeping` : "";
      img.setAttribute("aria-hidden", active ? "false" : "true");
    });
    $("stage").setAttribute("aria-labelledby", `cand-${d.current_pick}`);
  }
  renderSwitcher();
  renderLedger();
  renderDecision();
  layoutStage();
}

// ---------------------------------------------------------------------------
// Actions -- pick/confirm/skip, via the server-side pick/confirm/skip
// endpoints (duplicates_web.py), themselves built on the same
// duplicates_core apply_pick/unapply/pick_needs_reapply primitives.
// ---------------------------------------------------------------------------

function applyGroupPatch(i, data, keepLocalPick = false) {
  const localPick = state.detail && i === state.activeIndex ? state.detail.current_pick : data.current_pick;
  const pick = keepLocalPick ? localPick : data.current_pick;
  state.groups[i] = {
    index: i,
    status: data.status,
    file_count: data.paths.length,
    current_pick: pick,
    suggested_idx: data.suggested_idx,
    is_close_call: data.is_close_call,
  };
  if (i === state.activeIndex) state.detail = { ...data, current_pick: pick };
  renderQueue();
  renderNotices();
  if (i === state.activeIndex) renderPickChange();
  renderAppState();
}

// Every /pick runs through one promise chain, so the last key pressed is
// always the last write the server sees. Two picks in flight at once could
// otherwise land out of order and leave the server holding an earlier
// choice -- and /confirm moves files against the server's pick, not this
// tab's. confirmGroup() awaits this chain before acting for the same reason.
let pickChain = Promise.resolve();
let pendingPicks = 0;
let confirming = false;

// The mutating POSTs carry ?g=<generation>; the server answers 409 once a
// rescan has replaced the groups this tab is looking at. Reload rather than
// toasting a raw error -- index i now means a different group entirely.
let staleReload = null;

async function handleStale() {
  if (staleReload) return staleReload;   // queued POSTs all 409 together
  staleReload = staleReloadNow();
  try { await staleReload; } finally { staleReload = null; }
}

async function staleReloadNow() {
  showToast("A rescan replaced these groups — reloading this view.", true);
  try {
    await refreshState();
    await loadFirstPending();
  } catch (e) {
    showToast(`Couldn't reload after the rescan: ${e.message}`, true);
  }
}

// Reset to the first group still awaiting a decision.
async function loadFirstPending() {
  state.detail = null;
  state.activeIndex = -1;
  if (!state.groups.length) { renderGroup(); return; }
  const first = state.groups.findIndex((g) => g.status === "pending");
  await loadGroup(first >= 0 ? first : 0);
}

function pick(j) {
  const i = state.activeIndex;
  const d = state.detail;
  if (i < 0 || !d || j < 0 || j >= d.paths.length) return pickChain;
  if (state.status === "scanning" && !state.streaming) { showToast("A scan is running — decisions are locked until it finishes."); return pickChain; }
  // A pick queued behind an in-flight confirm would race it on the wire and
  // could move files for a candidate the user never saw confirmed.
  if (confirming) { showToast("Confirming — the pick is locked until it lands."); return pickChain; }

  // Optimistic: the flip must be instant, that's the point of the stage.
  d.current_pick = j;
  state.groups[i].current_pick = j;
  renderPickChange();
  renderQueue();

  pendingPicks += 1;
  pickChain = pickChain
    .then(() => api(`/api/group/${i}/pick?g=${state.generation}`, { method: "POST", body: JSON.stringify({ idx: j }) }))
    .then((data) => {
      pendingPicks -= 1;
      // While more picks are queued, this response is already stale for the
      // pick field -- keep the local one and take the rest.
      applyGroupPatch(i, data, pendingPicks > 0);
    })
    .catch(async (e) => {
      pendingPicks -= 1;
      if (e.status === 409) return handleStale();
      showToast(`Couldn't change the pick: ${e.message}`, true);
      // The optimistic flip never reached the server, so confirm would keep
      // a different file than the stage is showing: re-read what it holds.
      // A later queued pick will land the truth itself, so only the last
      // failure has to repair.
      if (pendingPicks > 0) return;
      try {
        applyGroupPatch(i, await api(`/api/group/${i}`));
      } catch (err) {
        showToast(`Lost track of which file the server is keeping (${err.message}) — reload before confirming.`, true);
      }
    });
  return pickChain;
}

function pickRelative(delta) {
  const d = state.detail;
  if (!d) return;
  const n = d.paths.length;
  pick(((d.current_pick + delta) % n + n) % n);
}

async function confirmGroup() {
  const i = state.activeIndex;
  if (i < 0 || !state.detail || confirming) return;
  confirming = true;
  try {
    await pickChain;  // the server must be holding the pick that's on screen
    const data = await api(`/api/group/${i}/confirm?g=${state.generation}`, { method: "POST" });
    applyGroupPatch(i, data);
    const moved = data.paths.length - 1;
    const files = `${moved} file${moved === 1 ? "" : "s"}`;
    // Named off the confirm response, never off the pick at key-press: the
    // toast has to say the file the server actually kept.
    const name = data.paths[data.current_pick];
    // A real move gets said out loud, with the way back: the group advances
    // immediately, so the decision bar is already describing the next group
    // by the time the user looks down at it.
    showToast(state.params && state.params.dry_run
      ? `Dry run: group ${i + 1} would keep ${name} and move ${files}. Nothing was moved.`
      : `Group ${i + 1}: kept ${name}, ${files} moved. Reopen group ${i + 1} to undo.`);
    advance();
  } catch (e) {
    if (e.status === 409) await handleStale();
    else showToast(`Confirm failed: ${e.message}`, true);
  } finally {
    confirming = false;
  }
}

async function skipGroup() {
  const i = state.activeIndex;
  if (i < 0 || !state.detail) return;
  const wasConfirmed = state.detail.status === "confirmed";
  try {
    await pickChain;
    const data = await api(`/api/group/${i}/skip?g=${state.generation}`, { method: "POST" });
    applyGroupPatch(i, data);
    // Undoing a confirmed group is a second real move -- files come back out
    // of the destination folder -- so it gets said as plainly as the first.
    if (wasConfirmed && !(state.params && state.params.dry_run)) {
      const moved = data.paths.length - 1;
      showToast(`Group ${i + 1}: ${moved} file${moved === 1 ? "" : "s"} moved back, group marked skipped.`);
    }
    if (data.status === "skipped") advance();
  } catch (e) {
    if (e.status === 409) await handleStale();
    else showToast(`Skip failed: ${e.message}`, true);
  }
}

function advance() {
  const n = state.groups.length;
  if (!n) return;
  for (let off = 1; off <= n; off++) {
    const j = (state.activeIndex + off) % n;
    if (state.groups[j].status === "pending") {
      loadGroup(j);
      // Focus follows the review, not the button that was clicked: leaving it
      // on Confirm means the next Enter or Space fires that button again --
      // against the group that just slid in underneath it.
      $("stage").focus({ preventScroll: true });
      return;
    }
  }
  showToast("That was the last one — every group is reviewed.");
}

function stepGroup(delta) {
  const n = state.groups.length;
  if (!n) return;
  const next = state.activeIndex < 0 ? 0 : (state.activeIndex + delta + n) % n;
  loadGroup(next);
}

function openFullRes() {
  const d = state.detail;
  if (!d) return;
  window.open(`/api/full/${d.index}/${d.current_pick}?g=${state.generation}`, "_blank", "noopener");
}

// ---------------------------------------------------------------------------
// Scan panel + SSE progress
// ---------------------------------------------------------------------------

function setScanPanelOpen(open) {
  $("scan-panel").hidden = !open;
  $("scope-toggle").setAttribute("aria-expanded", String(open));
  if (open) $("f-directory").focus();
  layoutStage();
}

function populateForm() {
  const p = state.params;
  if (!p) return;
  $("f-directory").value = p.directory;
  $("f-threshold").value = p.threshold;
  $("f-recursive").checked = p.recursive;
  $("f-dest").value = p.dest || "";
  $("f-dry-run").checked = p.dry_run;
}

function renderProgress(data) {
  const pct = data.total > 0 ? (100 * data.done) / data.total : 0;
  $("scanning-fill").style.width = `${pct}%`;
  // Just the phase: the directory being scanned is already the biggest
  // thing in the command bar, and uppercased at this size it reads as
  // shouting a path at someone who chose it ten seconds ago.
  $("scanning-phase").textContent = data.label || "Reading the directory";
  $("scanning-count").textContent = data.total > 0
    ? `${fmtInt(data.done)} / ${fmtInt(data.total)}`
    : "…";
}

// How long the stream may stay down before the user is told. EventSource
// reconnects on its own, so a blip must not raise an alarm -- only a
// connection that never comes back.
const PROGRESS_GRACE_MS = 15000;
let progressGraceTimer = null;

// The stream is gone for good: unlock the UI rather than leaving every
// control disabled behind a "scanning" state that will never end.
async function progressLost(es) {
  clearTimeout(progressGraceTimer);
  progressGraceTimer = null;
  es.close();
  if (state.eventSource === es) state.eventSource = null;
  showToast("Lost contact with the scan — the server may have stopped. Reload once it's back.", true);
  try { await refreshState(); } catch { /* the server is unreachable too */ }
  if (state.status === "scanning") {
    state.status = "error";
    state.error = "Lost contact with the running scan.";
    renderNotices();
    renderAppState();
    renderQueue();
  }
}

// Highest group count this stream has reported, so a re-read of /api/state is
// triggered once per new group rather than on every progress frame.
let lastSeenGroups = 0;

function connectProgress() {
  lastSeenGroups = 0;
  if (state.eventSource) state.eventSource.close();
  clearTimeout(progressGraceTimer);
  progressGraceTimer = null;
  const es = new EventSource("/api/progress");
  state.eventSource = es;
  state.status = "scanning";
  renderAppState();
  renderQueue();

  es.onerror = () => {
    if (es !== state.eventSource) return;
    // CLOSED means the browser gave up; otherwise it is retrying, and only a
    // retry that never succeeds is worth surfacing.
    if (es.readyState === EventSource.CLOSED) { progressLost(es); return; }
    if (progressGraceTimer) return;
    progressGraceTimer = setTimeout(() => {
      progressGraceTimer = null;
      if (es === state.eventSource && es.readyState !== EventSource.OPEN) progressLost(es);
    }, PROGRESS_GRACE_MS);
  };

  es.onmessage = (ev) => {
    clearTimeout(progressGraceTimer);
    progressGraceTimer = null;
    let data;
    // A malformed frame is not a dead server -- drop it and keep listening.
    try { data = JSON.parse(ev.data); } catch { return; }
    state.progress = data;
    renderProgress(data);
    if (data.status === "scanning") {
      // A streaming scan publishes groups as it finds them; the count growing
      // is the only signal that there is more to show. Re-read state, and
      // open the first group as soon as one exists so the review can start
      // without waiting for the scan to end.
      // Not while a pick or confirm is in flight: refreshState replaces
      // state.groups wholesale and would briefly undo the optimistic flip.
      // lastSeenGroups isn't advanced, so the next frame retries in 0.2s.
      if (data.groups > lastSeenGroups && !confirming && pendingPicks === 0) {
        lastSeenGroups = data.groups;
        refreshState()
          .then(() => (state.detail ? null : loadFirstPending()))
          .catch(() => { /* the terminal frame will refresh state anyway */ });
      } else {
        renderQueue();  // keep the tally's progress readout ticking
      }
      return;
    }

    // Must close explicitly: a browser EventSource treats a closed stream as
    // an error and reconnects a few seconds later, which would re-open this
    // endpoint forever once the scan is done.
    es.close();
    state.eventSource = null;
    refreshState()
      .then(loadFirstPending)
      .catch((e) => showToast(`Scan finished, but the result didn't load: ${e.message}`, true));
  };
}

// ---------------------------------------------------------------------------
// Help sheet, rendered from /api/metrics-info so it can't drift from what's
// actually scored -- same principle as find_duplicates.py's _help_body.
// ---------------------------------------------------------------------------

let lastFocused = null;

async function showHelp() {
  const body = $("help-body");
  try {
    const info = await api("/api/metrics-info");
    body.innerHTML = "";
    body.appendChild(helpContent(info));
  } catch (e) {
    body.textContent = `Couldn't load the metric list: ${e.message}`;
  }
  lastFocused = document.activeElement;
  $("help-sheet").hidden = false;
  document.querySelector(".sheet-panel").focus();
}

function helpContent(info) {
  const frag = document.createDocumentFragment();
  const h = (tag, text, cls) => {
    const el = document.createElement(tag);
    if (text) el.textContent = text;
    if (cls) el.className = cls;
    return el;
  };

  frag.appendChild(h("h3", "The quality score"));
  frag.appendChild(h("p", "A weighted composite of the measurements below, normalized 0–1 within this group only — min-max against the other files here, so it never compares across different photos. It's a hand-tuned heuristic, not a lab measurement: treat it as a strong hint, and look harder when the group is flagged as a close call."));
  frag.appendChild(h("p", "Dimensions and file size are shown for reference and do not feed the score."));

  frag.appendChild(h("h3", "Weighted measurements, by influence"));
  const ul = h("ul", null, "metric-list");
  Object.entries(info.weights)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .forEach(([name, weight]) => {
      const li = document.createElement("li");
      li.appendChild(h("span", Math.abs(weight).toFixed(2), "weight"));
      const right = document.createElement("span");
      right.appendChild(h("span", name, "metric-name"));
      right.appendChild(h("span", ` ${weight > 0 ? "higher is better" : "lower is better"}`, "metric-dir"));
      right.appendChild(document.createElement("br"));
      right.appendChild(h("span", info.descriptions[name], "metric-desc"));
      li.appendChild(right);
      ul.appendChild(li);
    });
  frag.appendChild(ul);

  frag.appendChild(h("h3", "Reading the stage"));
  frag.appendChild(h("p", "One file fills the stage at a time and every file in the group is laid out in exactly the same frame, so moving between them changes the pixels and nothing else — the sharper file is the one that stops looking soft. The file on the stage is the file you're keeping."));
  frag.appendChild(h("p", "Click the stage (or press Z) to inspect at 1:1. At that zoom the largest file in the group is shown at its true pixels and the others are scaled to match the same part of the scene, so an export upscaled from a smaller original gives itself away. Drag or hold shift with the arrow keys to pan; the spot you're inspecting stays put as you move between files."));
  frag.appendChild(h("p", "n/a in the table means that measurement has no value for that file — either its optional package isn't installed, or it failed on that one image. A measurement missing for any file is dropped from the whole group's score and the remaining weights are rescaled, so the group is still scored, just on fewer inputs."));

  frag.appendChild(h("h3", "Keyboard"));
  const keys = document.createElement("dl");
  keys.className = "keys";
  [
    ["← →", "Move between the files in this group"],
    ["1 – 9", "Keep a specific file"],
    ["↑ ↓", "Move between groups"],
    ["Enter / C", "Confirm keep"],
    ["Delete / S", "Skip group — on a confirmed group, move its files back"],
    ["Z", "Inspect at 1:1"],
    ["Shift + arrows", "Pan while inspecting"],
    ["O", "Open the kept file full-res in a new tab"],
    ["M", "Show or hide the measurements"],
    ["? / F1", "This panel"],
  ].forEach(([k, v]) => {
    keys.appendChild(h("dt", k));
    keys.appendChild(h("dd", v));
  });
  frag.appendChild(keys);

  frag.appendChild(h("h3", "Stopping and finishing"));
  frag.appendChild(h("p", "Confirm and skip apply immediately — there's no save step and nothing is left half-done, so it's safe to close this tab at any point. Non-kept files are moved, never deleted. The server keeps running until you stop it from the terminal, so reopening this URL picks up where you left off."));
  return frag;
}

function closeHelp() {
  $("help-sheet").hidden = true;
  if (lastFocused && lastFocused.focus) lastFocused.focus();
}

function helpOpen() { return !$("help-sheet").hidden; }

// aria-modal is a claim the browser doesn't enforce: without this, Tab walks
// straight out of the sheet and into the page behind it.
function trapTab(e) {
  const panel = document.querySelector(".sheet-panel");
  const items = Array.from(panel.querySelectorAll(
    'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
  )).filter((el) => !el.disabled && el.offsetParent !== null);
  if (!items.length) { e.preventDefault(); return; }
  const active = document.activeElement;
  const first = items[0];
  const last = items[items.length - 1];
  // The panel itself holds focus when the sheet opens: treat that like being
  // outside, or Shift+Tab from it walks backwards into the page behind.
  const outside = !panel.contains(active) || active === panel;
  if (e.shiftKey && (active === first || outside)) { last.focus(); e.preventDefault(); }
  else if (!e.shiftKey && (active === last || outside)) { first.focus(); e.preventDefault(); }
}

// ---------------------------------------------------------------------------
// Keyboard shortcuts. Bindings read KeyboardEvent.code (physical key
// position) rather than .key: an alternate layout remaps .key to a
// different character before the browser sees it.
// ---------------------------------------------------------------------------

function attachKeyboardHandler() {
  document.addEventListener("keydown", (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;

    if (helpOpen()) {
      if (e.code === "Escape" || e.key === "?" || e.code === "F1") { closeHelp(); e.preventDefault(); }
      else if (e.code === "Tab") trapTab(e);
      return;
    }

    const el = document.activeElement;
    if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA")) {
      if (e.code === "Escape") { setScanPanelOpen(false); $("scope-toggle").focus(); e.preventDefault(); }
      return;
    }

    if (e.code === "F1" || e.key === "?") { showHelp(); e.preventDefault(); return; }
    if (e.code === "Escape") {
      if (view.zoom) { setZoom(false); e.preventDefault(); }
      else if (!$("scan-panel").hidden) { setScanPanelOpen(false); e.preventDefault(); }
      return;
    }

    // Held keys must never chain destructive actions: confirm advances to the
    // next group, so one leaned-on Enter would walk the queue moving files
    // group after group. Arrows repeat on purpose (flipping candidates and
    // stepping the queue are both safe).
    const destructive = e.code === "Enter" || e.code === "KeyC"
      || e.code === "Delete" || e.code === "Backspace" || e.code === "KeyS";
    if (destructive && e.repeat) { e.preventDefault(); return; }

    // Enter belongs to whatever control has focus. Without this, Enter on
    // Help, Skip, a queue row or a candidate tab would confirm the group and
    // preventDefault would swallow the button's own activation -- a stray
    // Enter or click must never silently confirm/skip a group.
    const focused = document.activeElement;
    const onControl = !!(focused && focused.matches && focused.matches("button, a[href], summary"));

    if (view.zoom && e.shiftKey && e.code.startsWith("Arrow")) {
      const step = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.code];
      panBy(step[0], step[1]);
      e.preventDefault();
    }
    else if (e.code === "ArrowLeft") { pickRelative(-1); e.preventDefault(); }
    else if (e.code === "ArrowRight") { pickRelative(1); e.preventDefault(); }
    else if (e.code === "ArrowUp") { stepGroup(-1); e.preventDefault(); }
    else if (e.code === "Enter") { if (!onControl) { confirmGroup(); e.preventDefault(); } }
    else if (e.code === "ArrowDown") { stepGroup(1); e.preventDefault(); }
    else if (e.code === "KeyC") { confirmGroup(); e.preventDefault(); }
    else if (e.code === "Delete" || e.code === "Backspace" || e.code === "KeyS") { skipGroup(); e.preventDefault(); }
    else if (e.code === "KeyZ") { setZoom(!view.zoom); e.preventDefault(); }
    else if (e.code === "KeyO") { openFullRes(); e.preventDefault(); }
    else if (e.code === "KeyM") { setLedgerOpen(!state.ledgerOpen); e.preventDefault(); }
    else if (e.code.startsWith("Digit")) {
      const n = parseInt(e.code.slice(5), 10);
      if (n >= 1 && n <= 9 && state.detail && n <= state.detail.paths.length) { pick(n - 1); e.preventDefault(); }
    }
  });
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

function attachHandlers() {
  $("scope-toggle").addEventListener("click", () => setScanPanelOpen($("scan-panel").hidden));
  $("btn-confirm").addEventListener("click", confirmGroup);
  $("btn-skip").addEventListener("click", skipGroup);
  $("btn-open").addEventListener("click", openFullRes);
  $("btn-help").addEventListener("click", showHelp);
  $("help-close").addEventListener("click", closeHelp);
  $("help-scrim").addEventListener("click", closeHelp);
  $("ledger-toggle").addEventListener("click", () => setLedgerOpen(!state.ledgerOpen));

  $("scan-panel").addEventListener("submit", async (e) => {
    e.preventDefault();
    const body = {
      directory: $("f-directory").value,
      threshold: parseInt($("f-threshold").value, 10),
      recursive: $("f-recursive").checked,
      dest: $("f-dest").value || null,
      dry_run: $("f-dry-run").checked,
    };
    $("scan-btn").disabled = true;
    try {
      await api("/api/scan", { method: "POST", body: JSON.stringify(body) });
      setScanPanelOpen(false);
      state.detail = null;
      state.activeIndex = -1;
      renderProgress({ label: "", done: 0, total: 0 });
      connectProgress();
    } catch (err) {
      showToast(`Scan didn't start: ${err.message}`, true);
    } finally {
      $("scan-btn").disabled = false;
    }
  });

  attachStageHandlers();
  attachKeyboardHandler();
  window.addEventListener("resize", layoutStage);
}

async function init() {
  attachHandlers();
  // On a short screen the ledger would take the height the stage needs, and
  // the stage is where the decision is actually made -- the candidate strip
  // still carries each file's dimensions, size and score. One press of M (or
  // the header) brings the full table back.
  setLedgerOpen(window.innerHeight >= 940);
  await refreshState();
  populateForm();
  if (state.status === "scanning") connectProgress();
  else await loadFirstPending();
}

init();
