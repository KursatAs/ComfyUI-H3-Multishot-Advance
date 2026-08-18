# -*- coding: utf-8 -*-
"""Per-shot join/trim handling for assembled H3 timelines."""

try:
    from .h3_audio import _smart_head_trim
except Exception:
    from h3_audio import _smart_head_trim

try:
    from .h3_color import _cg_gauss
except Exception:
    from h3_color import _cg_gauss

try:
    from .h3_join import _vhs_glitch_audio, _vhs_glitch_frames
except Exception:
    from h3_join import _vhs_glitch_audio, _vhs_glitch_frames


def h3_apply_join_handling(imgs, wav, *, si, continuity, cp_trim, ov, ov_ho,
                           frames_parts, audio_parts, stream_assembler,
                           join_blend, audio_lock, spine_present, shot_seed,
                           seed, join_fx, sr):
    import torch

    # join handling. seamless: 1 duplicated frame. seamless_tail: the next
    # shot REGENERATES ov frames that duplicate this tail - drop them hard,
    # or crossfade them (join_blend) so any residual step is spread across the
    # band instead of landing on one boundary.
    if si > 0 and continuity == "context_pin" and cp_trim > 0:
        # the head is a regeneration of the previous shot's tail, there purely
        # to carry motion/colour/sound - drop it whole (video frames + the
        # exact matching audio span)
        imgs = imgs[min(cp_trim, imgs.shape[0] - 1):]
        wav = wav[..., int(round(sr * cp_trim / 24.0)):]
    elif si > 0 and continuity in ("seamless", "first_frame", "flf_chain"):
        # frame 0 IS the previous last frame - drop the duplicate (audio via
        # the quietest-gap cut so a head word survives)
        imgs = imgs[1:]
        wav = _smart_head_trim(wav, sr, int(round(sr / 24.0)))
    elif si > 0 and continuity in ("seamless_tail", "latent_handoff"):
        overlap = min(ov_ho if continuity == "latent_handoff" else ov,
                      imgs.shape[0] - 1)
        blend_prev = (frames_parts[-1] if frames_parts else
                      (stream_assembler.pending
                       if stream_assembler is not None else None))
        if join_blend and blend_prev is not None:
            prev = blend_prev
            band = min(overlap, prev.shape[0])
            weight = torch.linspace(1.0, 0.0, band).view(-1, 1, 1, 1)
            new_band = imgs[:band].cpu()
            blend = weight * prev[-band:] + (1.0 - weight) * new_band
            # grain guard: uncorrelated grain averages down in a blend;
            # re-inject it so the band has no grain dip
            high_pass = prev[-band:] - _cg_gauss(prev[-band:], 1.0)
            grain_sigma = float(high_pass.std())
            rng = torch.Generator().manual_seed(shot_seed ^ 0xB1E0D)
            blend = blend + grain_sigma * torch.sqrt(
                2.0 * weight * (1.0 - weight)) * torch.randn(
                blend.shape, generator=rng, dtype=blend.dtype)
            prev[-band:] = blend.clamp(0, 1)
        imgs = imgs[overlap:]
        xfade = max(1, int(sr * 40 / 1000.0))
        if (continuity == "latent_handoff"
                and (audio_lock or spine_present)):
            # Symmetric trim: the overlap audio is a locked REPLAY of the
            # previous tail - the real words already live in the previous
            # part's kept audio. Drop it with the replayed frames
            # (weld-compensated) so any residual step lands simultaneously
            # with the video join.
            keep_from = int(round(sr * overlap / 24.0)) - xfade
            if keep_from > 0:
                wav = wav[..., keep_from:]
        else:
            # silent-join (audio free) and seamless_tail: the new head's audio
            # is genuine content - keep it in full and trim the PREVIOUS tail
            # instead, weld-compensated so A/V stay sample-locked.
            cut = int(round(sr * overlap / 24.0)) - xfade
            if (audio_parts and cut > 0
                    and audio_parts[-1].shape[-1] > cut):
                audio_parts[-1] = audio_parts[-1][..., :-cut]
                # micro fade-out on the trimmed tail: the cut can land
                # mid-phoneme and the 40ms weld lets a clipped fragment tick
                # through; 100ms to silence reads as natural decay.
                fade_count = min(int(sr * 0.10), audio_parts[-1].shape[-1])
                if fade_count > 8:
                    fade = (torch.linspace(1.0, 0.0, fade_count) ** 0.5) \
                        .to(audio_parts[-1].dtype)
                    audio_parts[-1][..., -fade_count:] = \
                        audio_parts[-1][..., -fade_count:] * fade

    if join_fx == "vhs_glitch" and si > 0 and frames_parts:
        # dress the boundary: 2 tail frames of the previous part + 3 head
        # frames of this one, plus the audio hiccup on both sides of the weld.
        # Seeded per join for reproducibility.
        fx_seed = (seed ^ 0x7A9E) + si
        prev_tail = frames_parts[-1]
        prev_count = min(2, prev_tail.shape[0])
        if prev_count:
            prev_tail[-prev_count:] = _vhs_glitch_frames(
                prev_tail[-prev_count:], fx_seed)
        head_count = min(3, imgs.shape[0])
        if head_count:
            imgs[:head_count] = _vhs_glitch_frames(
                imgs[:head_count], fx_seed + 1).to(imgs.device, imgs.dtype)
        audio_parts[-1] = _vhs_glitch_audio(
            audio_parts[-1], sr, at_start=False, seed=fx_seed)
        wav = _vhs_glitch_audio(wav, sr, at_start=True, seed=fx_seed + 1)
        print("[H3Memory] join %d dressed as VHS glitch" % si, flush=True)

    return imgs, wav
