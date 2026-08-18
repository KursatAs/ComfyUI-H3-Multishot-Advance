# -*- coding: utf-8 -*-
"""Mutable per-shot sampler state and cache payload helpers."""

from dataclasses import dataclass, field


@dataclass
class H3SamplerState:
    sr: object = None
    cg_ref: object = None
    last_tail: object = None
    dbg_pins: list = field(default_factory=list)
    ho_v: object = None
    ho_a: object = None
    ho_taper_src: object = None
    ho_guard: object = None
    ho_wav_tail: object = None
    house_frame: object = None
    cc_mu: object = None
    cc_cov: object = None
    cp_prev: object = None
    pin_sig0: object = None
    pin_hf0: object = None
    cg_last_raw: object = None
    cp_trim: int = 0


def h3_build_sampler_cache_state(bank, frames_parts, audio_parts, lat_v_parts,
                                 lat_a_parts, voice_block, state):
    return {
        "bank_entries": bank._entries,
        "frames_parts": frames_parts,
        "audio_parts": audio_parts,
        "lat_v_parts": lat_v_parts,
        "lat_a_parts": lat_a_parts,
        "sr": state.sr,
        "voice_block": voice_block,
        "cg_ref": state.cg_ref,
        "last_tail": state.last_tail,
        "ho_v": state.ho_v,
        "ho_a": state.ho_a,
        "ho_taper_src": state.ho_taper_src,
        "ho_guard": state.ho_guard,
        "ho_wav_tail": state.ho_wav_tail,
        "house_frame": state.house_frame,
        "cc_mu": state.cc_mu,
        "cc_cov": state.cc_cov,
        "cp_prev": state.cp_prev,
        "pin_sig0": state.pin_sig0,
        "pin_hf0": state.pin_hf0,
        "cg_last_raw": state.cg_last_raw,
        "cp_trim": state.cp_trim,
    }


def h3_restore_sampler_cache_state(payload, bank):
    bank._entries = list(payload["bank_entries"])
    state = H3SamplerState(
        sr=payload["sr"],
        cg_ref=payload["cg_ref"],
        last_tail=payload["last_tail"],
        ho_v=payload["ho_v"],
        ho_a=payload["ho_a"],
        ho_taper_src=payload["ho_taper_src"],
        ho_guard=payload["ho_guard"],
        ho_wav_tail=payload["ho_wav_tail"],
        house_frame=payload["house_frame"],
        cc_mu=payload["cc_mu"],
        cc_cov=payload["cc_cov"],
        cp_prev=payload["cp_prev"],
        pin_sig0=payload["pin_sig0"],
        pin_hf0=payload["pin_hf0"],
        cg_last_raw=payload["cg_last_raw"],
        cp_trim=payload["cp_trim"],
    )
    return (
        list(payload["frames_parts"]),
        list(payload["audio_parts"]),
        list(payload["lat_v_parts"]),
        list(payload["lat_a_parts"]),
        payload["voice_block"],
        state,
    )


def h3_update_post_decode_state(state, imgs, wav, sr, continuity, ov, ov_ho,
                                audio_lock, audio_vae, aud_env):
    state.sr = sr
    tail_n = ov_ho if continuity == "latent_handoff" else ov
    state.last_tail = imgs[-max(tail_n, 1):].clone()
    if continuity != "latent_handoff":
        return state

    state.ho_wav_tail = wav[..., -int(round(sr * 0.925)):] \
        .detach().cpu().clone()
    # onset guard: encode this shot's quietest 0.4s as room tone. Locked into
    # the next shot's audio JUST PAST the trim point, it stops the model
    # planning speech under the replay - a lock alone merely masks the plan,
    # and the free region then resumes MID-WORD at the boundary.
    try:
        if not audio_lock:
            state.ho_guard = None
            raise StopIteration
        import torch
        guard_window = max(1, int(sr * 0.02))
        env = aud_env(wav.detach().cpu(), guard_window)
        guard_samples = int(sr * 0.4)
        guard_bins = max(1, guard_samples // guard_window)
        if wav.shape[-1] > guard_samples and env.shape[0] > guard_bins + 1:
            csum = torch.cumsum(torch.cat([torch.zeros(1), env]), 0)
            idx = int((csum[guard_bins:] - csum[:-guard_bins]).argmin()) \
                * guard_window
            segment = wav[..., idx:idx + guard_samples]
            wav3 = segment if segment.ndim == 3 else segment.unsqueeze(0)
            state.ho_guard = audio_vae.encode(
                wav3.movedim(1, -1)).detach().clone()
        else:
            state.ho_guard = None
    except StopIteration:
        pass
    except Exception as exc:
        state.ho_guard = None
        print("[H3Memory] onset guard skipped: %r" % (exc,), flush=True)
    return state
