// H3 widget persistence + a repair for the one historical layout shift.
//
// ComfyUI serializes widgets_values as a POSITIONAL array. Insert a widget in
// the middle of INPUT_TYPES and every saved value after it lands in the wrong
// slot - silently, with no warning, until something happens to be out of range
// and the queue dies on a message that names the wrong dial. That is exactly
// what "Value 4 bigger than max of 3: memory_frames" is: a workflow saved
// before v1.2 reading its old anchor_frames into memory_frames.
//
// Two parts:
//
// 1. PERSISTENCE (forward-looking). On serialize, also store {name: value} in
//    node.properties - a dict, so it is immune to ordering. On configure,
//    re-apply by name after the stock positional restore. From the first save
//    under this extension onward, no future layout change can shift anything.
//    NOTE for anyone editing workflow JSON by hand: values now live in TWO
//    places. Patch h3_advance_widget_values as well as widgets_values, or the shadow
//    copy restores over your edit on load.
//
// 2. REPAIR (backward-looking). Persistence cannot help a workflow saved
//    before it existed, so the one shift that already shipped is repaired on
//    load. v1.2 (2026-08-05) inserted seed_per_shot at widget index 8, ahead
//    of memory_frames and anchor_frames. Discriminator: index 8 is a BOOLEAN
//    in every version from v1.2 on, and an INT (memory_frames, 0-3) in v1.0 /
//    v1.1. If it is not a boolean, splice the default back in and everything
//    downstream falls back into place.

import { app } from "../../scripts/app.js";

const PROP = "h3_advance_widget_values";
const PERSISTED_NODES = new Set([
    "H3MultishotMemorySamplerAdvance",
    "H3ControlsAdvance",
]);
const SAMPLER_NODES = new Set(["H3MultishotMemorySamplerAdvance"]);
const H3_CONTROL_NODES = new Set(["H3ControlsAdvance"]);

const STUDIO_ASPECT_RATIOS = {
    "1:1 (Square)": [1, 1],
    "2:3 (Portrait Photo)": [2, 3],
    "3:2 (Photo)": [3, 2],
    "3:4 (Portrait Standard)": [3, 4],
    "4:3 (Standard)": [4, 3],
    "9:16 (Portrait Widescreen)": [9, 16],
    "16:9 (Widescreen)": [16, 9],
    "21:9 (Ultrawide)": [21, 9],
};

// v1.1 and earlier: [script, shot_count, width, height, frames_per_shot,
//                    seed, (seed control), steps, memory_frames, anchor_frames]
// v1.2 and later:   [..., steps, seed_per_shot, memory_frames, anchor_frames]
const SEED_PER_SHOT_INDEX = 8;

function repairLegacyLayout(node) {
    const wv = node?.widgets_values;
    if (!Array.isArray(wv) || wv.length <= SEED_PER_SHOT_INDEX) return false;
    if (typeof wv[SEED_PER_SHOT_INDEX] === "boolean") return false;  // current layout
    // A pre-v1.2 save: an INT (the old memory_frames) sits where the boolean
    // belongs. seed_per_shot defaults ON, and that is the measured recipe.
    wv.splice(SEED_PER_SHOT_INDEX, 0, true);
    return true;
}

function closestStudioAspect(width, height) {
    const target = Number(width) / Math.max(Number(height), 1);
    let best = "3:4 (Portrait Standard)";
    let bestDelta = Infinity;
    for (const [name, [w, h]] of Object.entries(STUDIO_ASPECT_RATIOS)) {
        const delta = Math.abs(Math.log(target / (w / h)));
        if (delta < bestDelta) {
            best = name;
            bestDelta = delta;
        }
    }
    return best;
}

function repairH3ControlsLayout(node) {
    const wv = node?.widgets_values;
    if (!Array.isArray(wv)) return false;
    if (typeof wv[0] === "string") {
        if (typeof wv[5] !== "string") return false;
        // KursatAs 2026-08-18 12:46: insert the new H3Controls seed
        // widget after frames_per_shot for saves made after aspect/MP/snap
        // was integrated but before H3Controls owned the sampler seed.
        wv.splice(4, 0, 0);
        return true;
    }
    const named = node?.widgets_values_named ?? {};
    const width = Number(named.width ?? wv[0] ?? 896);
    const height = Number(named.height ?? wv[1] ?? 1184);
    const aspect = closestStudioAspect(width, height);
    const megapixels = Math.max(0.1, Math.min(16,
        Math.round((width * height / 1024 / 1024) * 10) / 10));
    const multiple = 32;
    const frames = named.frames_per_shot ?? wv[2] ?? 243;
    const seed = named.seed ?? 0;
    const steps = named.steps ?? wv[3] ?? 12;
    const sampler = named.sampler_name ?? wv[4] ?? "euler";
    const scheduler = named.scheduler ?? wv[5] ?? "beta";
    const shots = named.shot_count ?? wv[6] ?? 0;
    const filePrompts = named.use_file_prompts ?? wv[7] ?? false;
    const takeSeconds = named.take_seconds ?? wv[8] ?? 0;
    const window = named.window ?? wv[9] ?? "auto";
    const pinFrames = named.pin_frames ?? wv[10] ?? 22;
    // KursatAs 2026-08-18 12:28: convert the old width/height H3 controls
    // widget layout to the integrated aspect/MP/snap layout before restore.
    node.widgets_values = [aspect, megapixels, multiple, frames, seed, steps,
        sampler, scheduler, shots, filePrompts, takeSeconds, window, pinFrames];
    return true;
}

app.registerExtension({
    name: "h3.widgetPersistence",

    beforeConfigureGraph(graphData) {
        // Runs before the nodes are built, so the splice lands before any
        // widget reads its value - and before ComfyUI range-checks it.
        let n = 0, c = 0;
        for (const node of graphData?.nodes ?? []) {
            if (H3_CONTROL_NODES.has(node?.type)) { if (repairH3ControlsLayout(node)) c++; continue; }
            if (SAMPLER_NODES.has(node?.type) && repairLegacyLayout(node)) n++;
        }
        if (c) {
            console.warn(`[H3-Multishot-Advance] mapped ${c} H3 Controls panel(s) to integrated aspect/MP/snap controls with the new fixed seed slot.`);
        }
        if (n) {
            console.warn(
                `[H3-Multishot-Advance] repaired ${n} sampler node(s) saved before ` +
                `v1.2 - seed_per_shot was inserted ahead of memory_frames in ` +
                `that release, shifting every dial after it. Save the ` +
                `workflow to make the repair permanent.`);
        }
    },

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!PERSISTED_NODES.has(nodeData?.name)) return;

        const origSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (o) {
            origSerialize?.apply(this, arguments);
            if (!this.widgets?.length) return;
            const map = {};
            for (const w of this.widgets) {
                if (w.name !== undefined && w.value !== undefined) {
                    map[w.name] = w.value;
                }
            }
            o.properties = o.properties || {};
            o.properties[PROP] = map;
        };

        const origConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (o) {
            origConfigure?.apply(this, arguments);
            const saved = o?.properties?.[PROP];
            if (!saved || !this.widgets?.length) return;
            for (const w of this.widgets) {
                if (!(w.name in saved)) continue;
                const v = saved[w.name];
                const opts = w.options?.values;
                if (Array.isArray(opts) && !opts.includes(v)) continue;  // renamed combo option
                w.value = v;
                w.callback?.(v);
            }
        };
    },
});
