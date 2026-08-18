# -*- coding: utf-8 -*-
"""INPUT_TYPES schema for the H3 Multishot memory sampler."""

try:
    from .h3_upscale import _up_model_list
except Exception:
    from h3_upscale import _up_model_list


_H3_DEFAULT_SEED = 0


def _sampler_names():
    """From core, so the list cannot rot out of step with ComfyUI."""
    try:
        import comfy.samplers
        return list(comfy.samplers.KSampler.SAMPLERS)
    except Exception:
        return ["res_multistep", "euler", "dpmpp_2m"]


def _scheduler_names():
    try:
        import comfy.samplers
        return list(comfy.samplers.KSampler.SCHEDULERS)
    except Exception:
        return ["simple", "normal", "beta"]


def h3_memory_sampler_input_types():
    return {"required": {
        "model": ("MODEL",),
        "clip": ("CLIP",),
        "video_vae": ("VAE",),
        "audio_vae": ("VAE",),
        "script": ("STRING", {"multiline": True, "default": "",
                              "tooltip": "One prompt per shot: JSON "
                                         "{\"prompts\": [...]} or plain "
                                         "blocks separated by --- lines."}),
        "shot_count": ("INT", {"default": 0, "min": 0, "max": 64,
                               "tooltip": "0 = one shot per script prompt."}),
        "width": ("INT", {"default": 768, "min": 32, "max": 4096, "step": 32}),
        "height": ("INT", {"default": 1344, "min": 32, "max": 4096, "step": 32}),
        "frames_per_shot": ("INT", {"default": 243, "min": 5, "max": 1450,
                                    "step": 17,
                                    "tooltip": "Trained range is ~124-362;"
                                    " longer single shots are ladder "
                                    "territory (RoPE extrapolation) - "
                                    "the audio-spine pass runs 719f "
                                    "low-res through exactly this."}),
        "seed": ("INT", {"default": _H3_DEFAULT_SEED, "min": 0,
                         "max": 0xffffffffffffffff,
                         "control_after_generate": "fixed"}),
        "steps": ("INT", {"default": 20, "min": 1, "max": 50}),
        "seed_per_shot": ("BOOLEAN", {"default": True,
                                      "label_on": "vary per shot",
                                      "label_off": "same seed every shot"}),
        "memory_frames": ("INT", {
            "default": 0, "min": 0, "max": 3,
            "tooltip": "RECENCY slots: how many of the most RECENT shots "
                       "stay in the bank, on top of the pinned one. Total "
                       "bank = pinned + recent, capped at 3 by H3's "
                       "reference limit.\n"
                       "DEFAULT 0, deliberately. The recent slots hand "
                       "each shot's ACCRETED output forward as reference "
                       "images, on top of the latent pin, so invented "
                       "detail compounds. Measured over ten shots at "
                       "960x544, moving 2 -> 0: texture growth 1.055 -> "
                       "1.022 per hop, chroma 1.086 -> 1.039, framing "
                       "correlation at shot 10 0.976 -> 0.995, and the "
                       "drift stops ACCELERATING - 4.2%->6.7% per hop "
                       "became 2.3%->2.0%. At 0 the only reference is "
                       "shot 1, which nothing has been added to yet.\n"
                       "Identity and framing held on a static scene. If a "
                       "busy scene loses motion continuity between shots, "
                       "raise it to 1."}),
        "anchor_frames": ("INT", {
            "default": 1, "min": 0, "max": 9,
            "tooltip": "Identity reference images taken from start_image, "
                       "used on EVERY shot (seed identity on shot 1 and "
                       "let the bank carry it after that; keep this "
                       "at 0 or 1)."}),
    }, "optional": {
        "start_image": ("IMAGE", {
            "tooltip": "Optional identity reference image. NOT a first "
                       "frame - this node has no keyframe."}),
        "keyframe_images": ("IMAGE", {
            "tooltip": "flf_chain only: N+1 boundary stills for N shots, "
                       "in order. Shot i is generated between image i "
                       "and image i+1. Best source is a single long "
                       "low-res take of the whole scene - its frames at "
                       "the boundary times are already colour-matched, "
                       "identity-matched and correctly posed, so every "
                       "join inherits one consistent look."}),
        "guide_audio": ("AUDIO", {
            "tooltip": "AUDIO SPINE: a continuous audio track for the "
                       "WHOLE take - a voice recording, a song, or a "
                       "low-res long pass. Each shot's audio stream is "
                       "locked to its time-slice of this track at every "
                       "sampling step; the video follows the locked "
                       "audio (lips included) via the model's own "
                       "audio-video attention. Works with EVERY "
                       "continuity mode - the per-shot stride accounts "
                       "for each mode's seam trim (render-verified on "
                       "context_pin). Any sample rate: the track is "
                       "resampled to the audio VAE's rate and mono is "
                       "upmixed. This is the locked-audio music-video "
                       "path."}),
        "sampler_name": (_sampler_names(), {"default": "er_sde"}),
        "scheduler": (_scheduler_names(), {"default": "beta"}),
        "bank_pinned": ("INT", {
            "default": 1, "min": 0, "max": 3,
            "tooltip": "How many of the EARLIEST shots stay in the bank "
                       "permanently. This is the anti-drift lever: with "
                       "shot 1 pinned, later shots always see where the "
                       "episode started. 0 = pure recency."}),
        "chain_gain_control": (["off", "flatten", "match_output",
                                "flatten_pin"], {
            "default": "off",
            "tooltip": "Texture levelling across the chain. flatten = "
                       "level the DECODED frames and the bank to shot 1 "
                       "(the pin still carries accreted texture; measured "
                       "x1.13-1.15 per hop at 736x1280 with flatten on). "
                       "flatten_pin = EXPERIMENTAL: flatten PLUS level "
                       "the pinned latents' fine-detail energy to shot "
                       "1's before they are pinned. First A/Bs "
                       "(2026-08-17) moved the ratchet only slightly; "
                       "kept for testing, not a default. It also performs "
                       "a temporary full-latent high-pass pass before "
                       "sampling, so it can raise the context_pin VRAM "
                       "peak. Long context_pin chains and extend takes "
                       "past ~4 windows still sharpen visibly - keep "
                       "takes short until the fix lands."}),
        "continuity": (["cut", "seamless", "seamless_tail",
                        "latent_handoff", "first_frame", "flf_chain",
                        "context_pin"], {
            "default": "context_pin",
            "tooltip": "cut = memory-bank only: no keyframe, every shot is a "
                       "fresh take held together by the bank. Framing and "
                       "exposure step between shots - correct for "
                       "multishot storytelling with cuts.\n"
                       "seamless = LEGACY, kept for comparison: hands "
                       "the next shot its predecessor's last frame as a "
                       "latent-only keyframe. That is a SOFT hint - no "
                       "vision tokens - and the model often satisfies it "
                       "loosely, so the join can still read as a cut. "
                       "For a real join use context_pin or "
                       "first_frame.\n"
                       "seamless_tail = LEGACY, kept for comparison: "
                       "pins the previous shot's frames -9/-5/-1 at "
                       "keyframe indices 0/4/8 so velocity carries too. "
                       "Needs interior keyframe anchors, which CONFLICT "
                       "with the Motion-Context pack - with that pack "
                       "installed this mode stops with an error naming "
                       "the alternatives instead of crashing mid-chain.\n"
                       "latent_handoff = one denoise trajectory: the next "
                       "shot's first latent block (video AND audio) is "
                       "hard-locked to the previous shot's actual tail "
                       "latents at every sampling step, released only for "
                       "the final detail steps. Speech continues mid-word "
                       "because the model wakes up inside its own "
                       "previous state - no keyframes involved.\n"
                       "first_frame = the model's OWN continuation "
                       "mechanism (fl2va task): the previous shot's "
                       "last frame is handed over the way the stock "
                       "Image-to-Video node does it - as VISION TOKENS "
                       "through the text encoder AND as the frame-0 "
                       "keyframe latent. The new shot literally starts "
                       "on that frame; only the duplicate is trimmed. "
                       "USE AN fl2va CHECKPOINT (ref2va is trained for "
                       "reference rows, not first-frame hand-off) - and "
                       "note the bank is disabled here, because fl2va "
                       "has no reference rows.\n"
                       "flf_chain = TRUE FFLF. Supply N+1 boundary "
                       "keyframes for N shots; shot i renders BETWEEN "
                       "keyframe i and keyframe i+1. Shot i ends on "
                       "exactly the image shot i+1 begins on, so the "
                       "join is one shared picture rather than two "
                       "independent guesses - colour, framing and pose "
                       "match by construction.\n"
                       "context_pin = Motion-Context chaining (needs the "
                       "ComfyUI-H3-Motion-Context pack): the previous "
                       "shot's last 22 frames are pinned into the next "
                       "shot's head AS RAW LATENTS at interior keyframe "
                       "coordinates - bit-identical content, no VAE "
                       "round trip, so velocity AND colour carry - plus "
                       "a timeline-placed audio ref. The regenerated "
                       "head is trimmed on decode. Composes with the "
                       "bank, colour levels and join fx."}),
        "bank_clip_frames": ("INT", {
            "default": 22, "min": 5, "max": 124, "step": 17,
            "tooltip": "Frames per bank slot, taken from the middle of "
                       "each shot (the bank stores a clip, not a single "
                       "frame). Reference rows cost time on every sampling "
                       "step, so keep this small: 22 is ~0.9s."}),
        "color_level": (["off", "scene", "mvgd"], {
            "default": "off",
            "tooltip": "Colour drift across a chain. 'scene' is the one to "
                       "use: ONE reference for the whole piece, applied "
                       "per frame at the very end, so every shot is pulled "
                       "to the same target and there is no step at any "
                       "join. 'mvgd' is DEPRECATED - it matches each shot "
                       "to a rolling house and leaves a hard step at every "
                       "seam (measured 29% warmth step; a render with it "
                       "on drifted +18% brighter over three shots). It is "
                       "kept only so saved graphs still load."}),
        "join_anchor_noise": ("FLOAT", {
            "default": 0.0, "min": 0.0, "max": 0.05, "step": 0.005,
            "tooltip": "Mix this much seeded noise into every join "
                       "keyframe latent (SkyReels noised-clean-condition). "
                       "The texture ratchet exists because the model "
                       "treats its own output as pristine and adds ~1.2x "
                       "detail on top; a little noise closes that gap at "
                       "the source. 0.02 is the researched setting. The "
                       "noised frames never reach the final cut."}),
        "join_blend": ("BOOLEAN", {
            "default": True, "label_on": "crossfade overlap",
            "label_off": "hard drop",
            "tooltip": "seamless_tail only: instead of hard-dropping the "
                       "9 regenerated overlap frames, crossfade them "
                       "against the previous tail (with a grain guard so "
                       "the blend band does not read as a grain dip) and "
                       "fade audio over the same 375ms. Any residual step "
                       "is spread across 9 frames instead of landing on "
                       "one boundary."}),
        "handoff_release": ("FLOAT", {
            "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
            "tooltip": "latent_handoff only: sigma below which the locked "
                       "overlap is released so the detail steps can "
                       "reconcile it with the new content. Higher = freer "
                       "(released earlier), 0 = locked to the very end."}),
        "bank_ref_noise": ("FLOAT", {
            "default": 0.0, "min": 0.0, "max": 0.05, "step": 0.005,
            "tooltip": "Mix this much seeded noise into every bank clip "
                       "before it is stored (SkyReels noised-clean-"
                       "condition, same idea as join_anchor_noise but for "
                       "the references). The texture ratchet rides bank "
                       "clips: the model copies its own output and adds "
                       "~1.2x detail, worst on faces/skin. 0.02 is the "
                       "keyframe-researched setting."}),
        "end_anchor": ("BOOLEAN", {
            "default": False, "label_on": "return to house framing",
            "label_off": "off",
            "tooltip": "Pin shot 1's FIRST frame as a keyframe at the "
                       "LAST frame of every later shot. H3 push-in creep "
                       "compounds across chained shots (each shot "
                       "inherits the previous crept tail and pushes "
                       "further), so tails drift ever further from the "
                       "prompted framing and every join mechanism "
                       "inherits an off-spec tail. The end pin closes "
                       "the loop: a shot may breathe inward mid-take but "
                       "must settle back to house framing by its tail, "
                       "so the next join starts from framing the text "
                       "agrees with."}),
        "join_fx": (["off", "vhs_glitch"], {
            "default": "off",
            "tooltip": "Diegetic join masking: dress every join in a "
                       "short VHS tracking hiccup (displacement bands, "
                       "chroma shift, dropout flecks, audio head-switch "
                       "duck+hiss) peaking on the boundary. For analog-"
                       "horror content the cut stops being an artifact "
                       "to hide and becomes part of the tape. Works with "
                       "every continuity mode."}),
        "audio_lock": ("BOOLEAN", {
            "default": False, "label_on": "locked (replay/spine)",
            "label_off": "free (silent-join)",
            "tooltip": "latent_handoff only. ON: the audio head is a "
                       "locked replay of the previous tail (or the "
                       "spine). OFF = the SILENT-JOIN policy: script "
                       "each shot to land its line and hold still for "
                       "the last beat; audio generates freely, the new "
                       "head's audio is kept in full and the previous "
                       "tail's audio is trimmed instead. Nothing the "
                       "model generates is discarded, so no word can "
                       "be lost BY CONSTRUCTION. A connected "
                       "guide_audio spine overrides this to locked."}),
        "handoff_taper": ("INT", {
            "default": 0, "min": 0, "max": 10,
            "tooltip": "latent_handoff only. Rows AFTER the hard lock "
                       "that are softly biased toward a continuation of "
                       "the previous motion, at linearly decaying "
                       "strength. Without it the lock ends at a cliff: "
                       "the replayed frames are faithful, then the next "
                       "frame follows the shot's OWN pose plan - "
                       "measured as a 3x larger discontinuity on the "
                       "person than on the static room. The taper gives "
                       "the pose a ramp to follow. 3-5 is a good start "
                       "(each row = 4 frames)."}),
        "handoff_depth": (["block", "bootstrap"], {
            "default": "block",
            "tooltip": "latent_handoff overlap depth. block = lock the "
                       "first full 17-frame block (strong video anchor, "
                       "22 frames trimmed, ~0.92s join gap). bootstrap "
                       "= lock only the 2 bootstrap rows to the "
                       "previous last 5 frames (5 frames trimmed, "
                       "~0.21s gap - an ordinary breath), with a "
                       "frame-0 keyframe pin of the previous last "
                       "frame; end_anchor + the bank carry the rest."}),
        # NEW WIDGETS GO LAST, ALWAYS: saved canvases map widgets_values
        # by index, and a widget inserted mid-list silently shifts every
        # value after it on the next load (the v1.4 lesson). Sockets
        # (IMAGE/AUDIO/forceInput STRING) take no widget slot, so their
        # position here is free.
        "reference_images": ("IMAGE", {
            "tooltip": "Optional SUBJECT/CHARACTER reference images (batch "
                       "= multiple refs, e.g. via Batch Images), carried "
                       "into EVERY shot as <Picture 1>, <Picture 2>, ... "
                       "ahead of the bank slots, so their numbering never "
                       "shifts as the bank fills. Distinct from "
                       "start_image, which seeds identity for shot 1 only. "
                       "Bind them in each shot's prompt: 'She looks like "
                       "the woman in <Picture 1>.'"}),
        "voice_ref": ("AUDIO", {
            "tooltip": "Optional VOICE ANCHOR carried into EVERY shot as a "
                       "reference audio (<Audio 1>). Feed a clean solo "
                       "line of the character and the voice is PINNED "
                       "across the chain instead of re-performed from "
                       "text. The bank carries voice too, but only from "
                       "shot 2 on - this covers shot 1 as well. Trimmed to "
                       "15s; reference rows cost speed on every step."}),
        "sampler_override": ("STRING", {
            "forceInput": True,
            "tooltip": "Link a sampler NAME here (e.g. from H3 Studio "
                       "Controls) to drive this widget from one master "
                       "source. Overrides sampler_name when connected."}),
        "scheduler_override": ("STRING", {
            "forceInput": True,
            "tooltip": "Link a scheduler NAME here to single-source it. "
                       "Overrides scheduler when connected."}),
        "self_anchor_voice": ("BOOLEAN", {
            "default": True, "label_on": "anchor to shot 1's voice",
            "label_off": "off",
            "tooltip": "AUTOMATIC voice identity: after shot 1 renders, "
                       "its own audio becomes the reference (<Audio 1>) "
                       "for every later shot - the voice the model "
                       "actually performed is pinned, no file needed. "
                       "Write shot 1 so the character speaks a clean "
                       "solo line. An external voice_ref, if connected, "
                       "takes priority."}),
        "reference_image_size": (["match", "max"], {
            "default": "max",
            "tooltip": "Reference image sizing. 'match' scales each ref "
                       "(down only, keeping aspect) to the generation's "
                       "pixel area; 'max' uses the reference pipeline's "
                       "2048px short edge for best identity fidelity. "
                       "Reference tokens ride through every sampling "
                       "step, so 'max' can be several times slower."}),
        "preview_first_shot": ("BOOLEAN", {
            "default": True, "label_on": "save shot 1 early",
            "label_off": "off",
            "tooltip": "Write shot 1 to output/video/H3_FIRSTSHOT/ the "
                       "MOMENT it finishes decoding - minutes before the "
                       "full chain completes - so a bad take can be "
                       "cancelled early. The full path is printed to the "
                       "console."}),
        "output_scale": ("FLOAT", {
            "default": 1.0, "min": 1.0, "max": 4.0, "step": 0.05,
            "tooltip": "FINAL size multiplier, applied after decode. "
                       "No upscale model: a lanczos resize, 1.0 is off. "
                       "WITH a model: the model runs at its OWN fixed "
                       "factor (usually 4x) and this brings the result to "
                       "source x this value, so 2.0 on a 4x model gives "
                       "2x, not 8x. "
                       "CAREFUL - 1.0 does NOT mean off once a model is "
                       "wired; it means do-not-correct, so you get the "
                       "full 4x. At 1344x768 that is 5376x3072: 94 MB a "
                       "frame, 22 GB a shot, and every shot stays in "
                       "system RAM until the master is joined. The console "
                       "prints the projected size when the model loads - "
                       "read it. "
                       "Adds resolution, not detail. Works with every "
                       "continuity mode; the bank still stores "
                       "base-resolution clips."}),
        "sigmas": ("SIGMAS", {
            "tooltip": "Optional custom sigma schedule, replacing "
                       "sampler/scheduler + steps entirely. Some turbo "
                       "LoRAs ship a schedule they need in order to work "
                       "at all. When this is connected the 'steps' and "
                       "'scheduler' widgets are IGNORED - the step count "
                       "becomes len(sigmas)-1 - and the console says so. "
                       "The two-pass upscale split is taken as a fraction "
                       "of the supplied schedule."}),
        "save_every_shot": ("BOOLEAN", {
            "default": True, "label_on": "write each shot as it decodes",
            "label_off": "off",
            "tooltip": "Write EVERY shot to output/video/H3_SHOTS/ the "
                       "moment it decodes, in addition to the master. "
                       "Insurance for long chains: everything that fails "
                       "after the last shot - a mux OOM, a full disk, a "
                       "cancelled tab - otherwise destroys the whole "
                       "render at once. Shots are written BEFORE the seam "
                       "trim, so consecutive files overlap by ~1s; the "
                       "master is still the clean join. Costs one file "
                       "write per shot."}),
        "upscale_model_name": (["(none)"] + _up_model_list(), {
            "default": "(none)",
            "tooltip": "Pick an upscale model by name instead of wiring a "
                       "loader node. Synthesises detail rather than "
                       "resizing, per shot, at the model's OWN fixed "
                       "factor - usually 4x. "
                       "Set output_scale to the size you actually want; "
                       "leaving it at 1.0 lets the raw 4x through, which "
                       "is rarely what you meant. The console prints the "
                       "projected frame size and RAM cost when the model "
                       "loads, before any sampling. "
                       "Reads models/upscale_models/, the same folder "
                       "ComfyUI's own Load Upscale Model reads."}),
        "master_normalize": (["off", "luma", "luma+contrast"], {
            "default": "luma+contrast",
            "tooltip": "Deflicker the FINISHED chain: every frame driven "
                       "to ONE global luma target, after the master "
                       "exists. Per-shot correction cannot work here - it "
                       "never reaches the raw-latent pin that carries the "
                       "drift, and correcting shots against a rolling "
                       "target leaves a step at every join (measured: all "
                       "per-shot dials ON still gave +142% texture and "
                       "+18% luma over three shots). This runs outside the "
                       "feedback loop and lands every frame on the same "
                       "number, so it cannot create a seam. Brightness "
                       "only: texture drift is NOT fixable after the fact, "
                       "because the only lever is blur and blur destroys "
                       "real detail along with the invented kind."}),
        # APPEND-ONLY from here. Inserting a widget above this line shifts
        # every saved workflow's values by one (v1.2 did exactly that with
        # seed_per_shot, and users got "Value 4 bigger than max of 3:
        # memory_frames" on graphs they had never edited).
        "pin_frames": (["22", "5", "39", "56"], {
            "default": "22",
            "tooltip": "context_pin only: how many frames of the previous "
                       "shot are pinned as raw latents at the head of the "
                       "next one. This is the whole join. 22 is the shipped "
                       "default; the longer settings hold the previous "
                       "shot's composition further into the new one, which "
                       "matters most at the FIRST join - shot 1 has nothing "
                       "pinned behind it, so it is the only shot whose "
                       "framing can disagree with the text. All four values "
                       "are latent-aligned; arbitrary numbers are not."}),
        "pin_noise": ("FLOAT", {
            "default": 0.05, "min": 0.0, "max": 0.10, "step": 0.005,
            "tooltip": "context_pin only: mix this much seeded noise into "
                       "the PINNED LATENT before it conditions the next "
                       "shot. Same noised-clean-condition idea as "
                       "join_anchor_noise, aimed at the thing that "
                       "actually carries the drift here - measured, the "
                       "texture ratchet under context_pin rides the raw "
                       "latent pin, and neither join_anchor_noise "
                       "(keyframes only) nor bank_ref_noise (bank images) "
                       "touches it. "
                       "Small, and scene-dependent: measured -1.8% per hop at "
                       "640x352 and -0.9% at 960x544 on a "
                       "detail-heavy scene, against much larger "
                       "gains on a scene that barely ratcheted at "
                       "all. It cannot touch the dominant drift in a "
                       "busy frame - master_normalize=luma+contrast "
                       "is what does that. Above 0.10 it gets WORSE "
                       "(0.20 measured 1.228 against a 1.211 "
                       "control), which is why the range stops "
                       "there. Set 0 to disable. "
                       "The noised latent conditions the next shot but "
                       "never reaches the final cut."}),
        "pin_renorm": ("BOOLEAN", {
            "default": True,
            "label_on": "hold shot 1's level",
            "label_off": "off",
            "tooltip": "context_pin only: rescale each pinned latent so "
                       "its standard deviation matches the FIRST "
                       "pin's. The pin's own sigma climbs every hop "
                       "(1.0325, 1.0368 against a 1.0220 shot-1 "
                       "anchor), and that inflated pin is what "
                       "conditions the next shot - so it compounds "
                       "upstream of anything a master pass can reach. "
                       "Measured against a matched control, same seed, "
                       "960x544 124f x4: texture growth over the chain "
                       "+15.1% -> +11.5%, and framing correlation to "
                       "shot 1 held 0.985 -> 0.996 by the last shot. "
                       "The framing gain is the bigger one and cannot "
                       "be a metric artifact - post passes do not move "
                       "composition. One seed, one canvas. A scalar "
                       "rescale moves no structure, so unlike a pixel "
                       "correction it cannot blur detail."}),
        # NEW IN 2.2.4 - appended at the very END of the optional block on
        # purpose. widgets_values is a POSITIONAL array, so inserting a
        # widget anywhere but the end silently shifts every value after it
        # in workflows people have already saved. That is what produced the
        # per-shot sharpening incident; never insert mid-list.
        "reference_subjects": ("STRING", {
            "default": "",
            "tooltip": "How your reference pictures group into PEOPLE. "
                       "Empty (default) = every reference picture is "
                       "declared a photograph of the same one person. That "
                       "is correct for a single character and WRONG for "
                       "several - the model is told they are all the same "
                       "individual and renders the average. Comma counts in "
                       "picture order: '3,3' means pictures 1-3 are person "
                       "A and 4-6 are person B; '2,2,2' for three people. "
                       "Only <Subject 1> is described as speaking, because "
                       "H3's voice conditioning is single-speaker."}),
        "reference_video": ("IMAGE", {
            "tooltip": "NEW IN 2.2.4. Frames of an EXISTING clip, handed to "
                       "the model as a video reference alongside the "
                       "picture references. Read what this does before "
                       "wiring it: H3 is told a video reference is 'an "
                       "earlier moment of this same continuous scene' and "
                       "to keep its framing, camera distance, room contents "
                       "and colour temperature. It is SCENE and APPEARANCE "
                       "conditioning. It is NOT motion transfer - there is "
                       "no pose, depth or optical-flow path in H3, so it "
                       "will not make your subject copy the movement in the "
                       "clip. Trim it before wiring; a raw 25-second clip "
                       "is ~50 reference frames riding every sampling "
                       "step after the model's 2 fps subsample."}),
        "reference_video_audio": ("AUDIO", {
            "tooltip": "Optional soundtrack for reference_video. H3 pairs "
                       "video references with an audio reference, so if you "
                       "leave this empty silence is generated to match. "
                       "Supply the clip's real audio when you want its "
                       "voice timbre referenced too."}),
        # APPENDED LAST (saved-graph widget order).
        "low_ram_master": ("BOOLEAN", {
            "default": False,
            "tooltip": "Stream the master to disk instead of holding every "
                       "decoded shot in host RAM until the join - peak RAM "
                       "becomes ONE shot. Each shot is staged lossless the "
                       "moment it decodes, levelled with the same "
                       "master_normalize math (from stored statistics), "
                       "and the finished file's path comes out of the new "
                       "master_path output; master_frames carries a single "
                       "placeholder frame. v1 streams the default config "
                       "only: join_blend, join_fx and color_level=scene "
                       "fall back to the RAM path with a printed reason. "
                       "Needs ffmpeg on PATH."}),
        "audio_pin_frames": ("INT", {
            "default": 0, "min": 0, "max": 240, "step": 1,
            "tooltip": "context_pin only: frames of the previous shot's "
                       "AUDIO to pin as reference, independent of the "
                       "picture pin. 0 = same as pin_frames (22 = 0.9 s). "
                       "Longer audio context costs conditioning rows but "
                       "NO delivered frames - the head trim stays at "
                       "pin_frames. 96 (4 s) is the audio-memory window "
                       "between chunks in earlier memory-bank builds; try "
                       "it for continuous speech "
                       "across joins. Experimental."}),
        "shot_cache": (["off", "use_cache", "rebuild_cache"], {
            "default": "use_cache",
            "tooltip": "off: render all clips. use_cache: reuse unchanged "
                       "clips and continue from the first changed --- "
                       "prompt. rebuild_cache: ignore old cache, render "
                       "all clips, and write a fresh cache."}),
    },
        # hidden inputs are not widgets, so saved workflows are unaffected
        "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}}
