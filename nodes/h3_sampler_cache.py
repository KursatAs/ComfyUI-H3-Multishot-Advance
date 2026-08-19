# -*- coding: utf-8 -*-
"""Sampler-specific shot-cache fingerprint construction."""

try:
    from .h3_cache import _H3_SHOT_CACHE_VERSION, _h3_cache_fingerprint
except Exception:
    from h3_cache import _H3_SHOT_CACHE_VERSION, _h3_cache_fingerprint


def h3_build_sampler_cache_base(
        *, width, height, frames_per_shot, seed, steps, seed_per_shot,
        sampler_name, scheduler, sigmas, memory_frames, anchor_frames,
        bank_pinned, chain_gain_control, bank_clip_frames, continuity,
        color_level, join_anchor_noise, join_blend, handoff_release,
        bank_ref_noise, end_anchor, join_fx, audio_lock, handoff_taper,
        handoff_depth, self_anchor_voice, reference_image_size,
        preview_first_shot, save_every_shot, output_scale,
        upscale_model_name, master_normalize, pin_frames, pin_noise,
        pin_renorm, reference_subjects, audio_pin_frames, model, clip,
        video_vae, audio_vae, start_image, keyframe_images, reference_images,
        guide_audio, voice_ref, reference_video, reference_video_audio):
    return {
        "version": _H3_SHOT_CACHE_VERSION,
        "node": "H3MultishotMemorySampler",
        # KursatAs 2026-08-19 04:45: speech guard changes conditioning refs;
        # old prefix caches were built with voice/audio always riding along.
        "speech_guard": 1,
        # KursatAs 2026-08-18 07:49: Keep total shot_count out of the base key
        # so a 2-shot cache can resume when the user extends the same project
        # to 3+ shots. Prefix keys still include the exact prior prompts.
        "width": int(width),
        "height": int(height),
        "frames_per_shot": int(frames_per_shot),
        "seed": int(seed),
        "steps": int(steps),
        "seed_per_shot": bool(seed_per_shot),
        "sampler_name": str(sampler_name),
        "scheduler": str(scheduler),
        "sigmas": _h3_cache_fingerprint(sigmas),
        "memory_frames": int(memory_frames),
        "anchor_frames": int(anchor_frames),
        "bank_pinned": int(bank_pinned),
        "chain_gain_control": str(chain_gain_control),
        "bank_clip_frames": int(bank_clip_frames),
        "continuity": str(continuity),
        "color_level": str(color_level),
        "join_anchor_noise": float(join_anchor_noise),
        "join_blend": bool(join_blend),
        "handoff_release": float(handoff_release),
        "bank_ref_noise": float(bank_ref_noise),
        "end_anchor": bool(end_anchor),
        "join_fx": str(join_fx),
        "audio_lock": bool(audio_lock),
        "handoff_taper": int(handoff_taper),
        "handoff_depth": str(handoff_depth),
        "self_anchor_voice": bool(self_anchor_voice),
        "reference_image_size": str(reference_image_size),
        "preview_first_shot": bool(preview_first_shot),
        "save_every_shot": bool(save_every_shot),
        "output_scale": float(output_scale),
        "upscale_model_name": str(upscale_model_name),
        "master_normalize": str(master_normalize),
        "pin_frames": str(pin_frames),
        "pin_noise": float(pin_noise),
        "pin_renorm": bool(pin_renorm),
        "reference_subjects": str(reference_subjects),
        "audio_pin_frames": int(audio_pin_frames),
        "model": {
            "class": type(model).__name__,
            "id": id(model),
            "checkpoint": str(getattr(
                getattr(model, "model", None), "h3_checkpoint_name", "") or ""),
        },
        "clip": {"class": type(clip).__name__, "id": id(clip)},
        "video_vae": {"class": type(video_vae).__name__, "id": id(video_vae)},
        "audio_vae": {"class": type(audio_vae).__name__, "id": id(audio_vae)},
        "inputs": {
            "start_image": _h3_cache_fingerprint(start_image),
            "keyframe_images": _h3_cache_fingerprint(keyframe_images),
            "reference_images": _h3_cache_fingerprint(reference_images),
            "guide_audio": _h3_cache_fingerprint(guide_audio),
            "voice_ref": _h3_cache_fingerprint(voice_ref),
            "reference_video": _h3_cache_fingerprint(reference_video),
            "reference_video_audio": _h3_cache_fingerprint(
                reference_video_audio),
        },
    }
