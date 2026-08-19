# -*- coding: utf-8 -*-
"""Conditioning payload construction for the H3 Multishot sampler."""

try:
    from .h3_prompting import (
        _parse_ref_groups,
        _subject_defs,
    )
except Exception:
    from h3_prompting import (
        _parse_ref_groups,
        _subject_defs,
    )


def _h3_encode_keyframes(mmh3, video_vae, keyframes, target_width,
                         target_height, join_anchor_noise, shot_seed):
    """Encode a keyframe list onto a target grid, in place."""
    for keyframe_index, keyframe in enumerate(keyframes):
        z = video_vae.encode(
            mmh3._resize(
                keyframe.pop("image"), target_width, target_height,
                "disabled"))
        if join_anchor_noise > 0:
            # noised clean condition: the model must not treat its own output
            # as pristine (that is the 1.2x ratchet). Seeded so same-seed A/B
            # arms stay clean.
            import torch
            generator = torch.Generator(device=z.device).manual_seed(
                (shot_seed ^ 0x5EED) + keyframe_index)
            t_add = float(join_anchor_noise)
            z = ((1.0 - t_add) * z
                 + t_add * torch.randn(
                     z.shape, generator=generator, device=z.device,
                     dtype=z.dtype))
        keyframe["latent"] = z
    return keyframes


def h3_build_shot_conditioning(clip, node_helpers, mmh3, video_vae, prompt,
                               si, ref_items, ref_blocks, kf_vision,
                               keyframes, reference_subjects,
                               two_pass_upscale, width, height, frame_count,
                               join_anchor_noise, shot_seed, pass1_width=None,
                               pass1_height=None, speech_active=True):
    if kf_vision:
        tokens = clip.tokenize(prompt, images=kf_vision)
    elif ref_items:
        n_img = sum(1 for item in ref_items if item["type"] == "image")
        groups = _parse_ref_groups(reference_subjects, n_img)
        # KursatAs 2026-08-19 04:45: pass per-shot speech state into automatic
        # subject_definitions so silent shots are not described as speaking.
        subject_defs = _subject_defs(
            n_img,
            sum(1 for item in ref_items if item["type"] == "audio"),
            sum(1 for item in ref_items if item["type"] == "video"),
            image_subjects=groups, speaking=speech_active)
        if subject_defs:
            if si == 1:
                print("[H3Memory] subject_definitions added for %d reference "
                      "item(s) - the model is now told what the refs ARE and "
                      "to preserve identity, room, colour and voice timbre"
                      % len(ref_items), flush=True)
                if groups:
                    print("[H3Memory] reference_subjects %r -> %d distinct "
                          "subject(s) across %d picture(s); each is declared "
                          "its own person instead of all being <Subject 1>"
                          % (reference_subjects, max(groups), n_img),
                          flush=True)
                elif n_img > 1:
                    print("[H3Memory] %d reference pictures are all declared "
                          "<Subject 1> (one person). If they show DIFFERENT "
                          "people, set reference_subjects (e.g. '3,3') or "
                          "they will blend." % n_img, flush=True)
            prompt = prompt.rstrip() + "\n" + subject_defs
        tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
    else:
        tokens = clip.tokenize(prompt)

    cond = clip.encode_from_tokens_scheduled(tokens)
    cond_hi = cond if two_pass_upscale else None

    if keyframes:
        # two-pass pins the SAME pixels on both grids - encoding twice is
        # cheap and exact, where resampling a latent is neither.
        keyframes_hi = (
            [{"resolved_frame_index": item["resolved_frame_index"],
              "image": item["image"]} for item in keyframes]
            if two_pass_upscale else None)
        cond = node_helpers.conditioning_set_values(cond, {
            "minimax_keyframes": _h3_encode_keyframes(
                mmh3, video_vae, keyframes,
                pass1_width if two_pass_upscale else width,
                pass1_height if two_pass_upscale else height,
                join_anchor_noise, shot_seed),
            "minimax_frame_count": frame_count,
        })
        if keyframes_hi is not None:
            cond_hi = node_helpers.conditioning_set_values(cond_hi, {
                "minimax_keyframes": _h3_encode_keyframes(
                    mmh3, video_vae, keyframes_hi, width, height,
                    join_anchor_noise, shot_seed),
                "minimax_frame_count": frame_count,
            })

    if ref_blocks:
        cond = node_helpers.conditioning_set_values(
            cond, {"minimax_refs": ref_blocks})
        if cond_hi is not None:
            cond_hi = node_helpers.conditioning_set_values(
                cond_hi, {"minimax_refs": ref_blocks})

    return prompt, cond, cond_hi
