# -*- coding: utf-8 -*-
"""Sampling-output decode and continuity-state capture helpers."""


def h3_decode_shot_outputs(out, continuity, handoff_taper, ho_rows, ho_acols,
                           audio_lock, lat_v_parts, lat_a_parts, video_vae,
                           audio_vae, vae_decode_audio, cp_prev=None,
                           ho_taper_src=None, ho_v=None, ho_a=None):
    lat = out["samples"]
    if continuity == "context_pin":
        # the WHOLE AV latent, exactly as sampled - the next shot pins its tail
        # bit-identically, no decode in the path.
        cp_prev = {"samples": out["samples"]}

    audio_lat = None
    if getattr(lat, "is_nested", False):
        comps = lat.unbind()
        audio_lat = comps[1] if len(comps) > 1 else None
        lat = comps[0]

    # issue #12: keep each shot's latent exactly as sampled. Trimming here
    # would be throwing the pin material away on the user's behalf, and the pin
    # is the part that cannot be recovered later.
    lat_v_parts.append(lat.detach().cpu())
    if audio_lat is not None:
        lat_a_parts.append(audio_lat.detach().cpu())

    if continuity == "latent_handoff":
        ho_taper_src = (lat[:, :, -1:].detach().clone()
                        if handoff_taper > 0 else None)
        ho_v = lat[:, :, -ho_rows:].detach().clone()
        ho_a = (audio_lat[..., -ho_acols:].detach().clone()
                if audio_lat is not None and audio_lock else None)

    imgs = video_vae.decode(lat)
    if imgs.ndim == 5:
        imgs = imgs.reshape(-1, imgs.shape[-3], imgs.shape[-2],
                            imgs.shape[-1])
    aud = vae_decode_audio(audio_vae, out)
    sr = aud["sample_rate"]
    wav = aud["waveform"]

    return imgs, wav, sr, cp_prev, ho_taper_src, ho_v, ho_a
