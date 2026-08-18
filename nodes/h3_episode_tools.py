# -*- coding: utf-8 -*-
"""Reference, prompt, and studio-control helpers for H3 Advance."""
import math


_H3_STUDIO_ASPECT_RATIOS = {
    "1:1 (Square)": (1, 1),
    "2:3 (Portrait Photo)": (2, 3),
    "3:2 (Photo)": (3, 2),
    "3:4 (Portrait Standard)": (3, 4),
    "4:3 (Standard)": (4, 3),
    "9:16 (Portrait Widescreen)": (9, 16),
    "16:9 (Widescreen)": (16, 9),
    "21:9 (Ultrawide)": (21, 9),
}


def _h3_studio_resolution(aspect_ratio, megapixels, multiple):
    # KursatAs 2026-08-18 12:28: fold ComfyUI's Resolution Selector math into
    # H3Controls so size, frames and sampler settings live in one panel.
    w_ratio, h_ratio = _H3_STUDIO_ASPECT_RATIOS.get(
        aspect_ratio, _H3_STUDIO_ASPECT_RATIOS["3:4 (Portrait Standard)"])
    multiple = min(256, max(32, int(multiple or 32)))
    megapixels = min(16.0, max(0.1, float(megapixels or 1.0)))
    total_pixels = megapixels * 1024 * 1024
    scale = math.sqrt(total_pixels / (w_ratio * h_ratio))
    width = max(multiple, round(w_ratio * scale / multiple) * multiple)
    height = max(multiple, round(h_ratio * scale / multiple) * multiple)
    return int(width), int(height)


class H3Controls:
    """ONE set of widgets that drives BOTH stages of the studio graph.

    The two-stage chain has duplicated settings (stage A's conditioning node
    and stage B's multishot sampler each carry width/height/frames/steps/
    sampler/scheduler). Editing one and forgetting the other produces
    mismatched renders that fail at the concat - or worse, succeed at two
    different qualities. This node is the single source: wire its outputs to
    both stages and there is exactly one place to change anything.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "aspect_ratio": (list(_H3_STUDIO_ASPECT_RATIOS.keys()), {
                "default": "3:4 (Portrait Standard)",
                "tooltip": "Canvas aspect ratio. Replaces the separate "
                           "Resolution Selector node."}),
            "megapixels": ("FLOAT", {
                "default": 1.0, "min": 0.1, "max": 16.0, "step": 0.1,
                "tooltip": "Target total megapixels. 1.0 MP with 3:4 and "
                           "multiple 32 gives 896x1184."}),
            "multiple": ("INT", {
                "default": 32, "min": 32, "max": 256, "step": 32,
                "tooltip": "Snap width and height to this multiple. 32 is "
                           "the H3-safe default."}),
            "frames_per_shot": ("INT", {
                "default": 243, "min": 5, "max": 481, "step": 17,
                "tooltip": "H3's 17k+5 grid at 24fps. 243 = ~10.1s default; "
                           "362 = ~15.1s trained max. Drives stage A's "
                           "length AND stage B's per-shot length."}),
            # KursatAs 2026-08-18 12:46: H3Controls owns the sampler seed
            # so the root workflow can drive render size, duration and
            # determinism from one panel.
            "seed": ("INT", {
                "default": 0, "min": 0, "max": 0xffffffffffffffff,
                "control_after_generate": "fixed",
                "tooltip": "Seed sent to the H3 sampler. Default is 0 and "
                           "the after-generate mode defaults to fixed."}),
            "steps": ("INT", {"default": 12, "min": 1, "max": 50}),
            "sampler_name": (_sampler_names_sc(), {"default": "er_sde"}),
            "scheduler": (_scheduler_names_sc(), {"default": "beta"}),
        }, "optional": {
            # KursatAs 2026-08-18 12:46: optional controls stay below the
            # core render controls so the panel reads top-to-bottom as setup
            # -> duration -> helpers.
            "shot_count": ("INT", {
                "default": 0, "min": 0, "max": 30,
                "tooltip": "How many shots the scene is. 0 = one shot per "
                           "prompt in the script (or lets prompt planning "
                           "decide). Wire this to BOTH the sampler's "
                           "shot_count and the prompt generator's num_shots so they "
                           "cannot disagree."}),
            "use_file_prompts": ("BOOLEAN", {
                "default": False,
                "label_on": "file / prompt set",
                "label_off": "manual entry",
                "tooltip": "Legacy prompt-source flag. OFF = manual prompt "
                           "entry; ON = external prompt-set source when the "
                           "workflow provides its own switch."}),
            # EXTEND TAKE (2026-08-17): give it a length instead of shots x
            # frames. 0 = off (everything above behaves exactly as before).
            "take_seconds": ("FLOAT", {
                "default": 0.0, "min": 0.0, "max": 600.0, "step": 0.5,
                "tooltip": "EXTEND TAKE: 0 = off. Any other value = the length "
                           "of the finished take; frames_per_shot and "
                           "shot_count above are then OVERRIDDEN by a window "
                           "sized for this card (see 'window') and the count "
                           "that fills the time. Set prompt planning to "
                           "'extend take' so it writes ONE speech across "
                           "the windows."}),
            "window": (["auto", "243", "226", "209", "192", "175", "158",
                        "141", "124", "107", "90"], {
                "default": "auto",
                "tooltip": "Frames per window when take_seconds is set. auto "
                           "= the largest window whose estimated activation "
                           "pool fits with most of the weights resident on "
                           "THIS card (wire model for a real weight size; 15 "
                           "GB assumed otherwise). Bigger window = fewer "
                           "joins; smaller = less VRAM."}),
            "model": ("MODEL", {
                "tooltip": "Optional, EXTEND TAKE only: the loaded H3 model, "
                           "so 'auto' can size its weights."}),
            # KursatAs 2026-08-17 20:48: extend-take planning must use the
            # same replay trim as the sampler; otherwise custom pin lengths
            # make take_seconds over- or under-plan the delivered duration.
            "pin_frames": ("INT", {
                "default": 22, "min": 0, "max": 56, "step": 1,
                "tooltip": "EXTEND TAKE only: set this to the same value as "
                           "the memory sampler's pin_frames. 22 is the "
                           "tested default. If you change the sampler pin but "
                           "leave this at 22, take_seconds will calculate the "
                           "wrong delivered length."}),
        }}

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "STRING", "STRING",
                    "INT", "BOOLEAN")
    RETURN_NAMES = ("width", "height", "frames_per_shot", "seed", "steps",
                    "sampler_name", "scheduler",
                    "shot_count", "use_file_prompts")
    FUNCTION = "emit"
    CATEGORY = "video/minimax"

    def emit(self, aspect_ratio, megapixels, multiple, frames_per_shot, seed,
             steps, sampler_name, scheduler, shot_count=0,
             use_file_prompts=False, take_seconds=0.0, window="auto",
             model=None, pin_frames=22):
        width, height = _h3_studio_resolution(
            aspect_ratio, megapixels, multiple)
        seed = int(seed or 0)
        if take_seconds and take_seconds > 0:
            from .h3_extend import plan_take
            pin_frames = max(0, int(pin_frames or 0))
            n, f, total, summary = plan_take(take_seconds, window, width,
                                             height, 24, pin_frames, model)
            frames_per_shot, shot_count = f, n
            print("[H3Controls] " + summary, flush=True)
        print(f"[H3Controls] {width}x{height}, {frames_per_shot}f/shot, "
              f"seed {seed}, {steps} steps, {sampler_name}/{scheduler}, "
              f"{aspect_ratio}, {megapixels:g} MP, snap {multiple}, "
              f"shots={'auto' if not shot_count else shot_count}, "
              f"prompts={'file' if use_file_prompts else 'manual'}",
              flush=True)
        return (width, height, frames_per_shot, seed, steps,
                sampler_name, scheduler, shot_count, use_file_prompts)


def _sampler_names_sc():
    try:
        import comfy.samplers
        return comfy.samplers.KSampler.SAMPLERS
    except Exception:
        return ["euler", "res_multistep", "res_2s"]


def _scheduler_names_sc():
    try:
        import comfy.samplers
        return comfy.samplers.KSampler.SCHEDULERS
    except Exception:
        return ["beta", "normal", "simple", "beta57"]




NODE_CLASS_MAPPINGS = {
    "H3Controls": H3Controls,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3Controls": "H3 Controls Advance",
}
