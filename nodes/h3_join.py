# -*- coding: utf-8 -*-
"""Join masking and memory-bank clip helpers for H3 Multishot Advance."""


def _vhs_glitch_frames(frames, seed, strength=1.0):
    """Diegetic VHS tracking glitch over a short frame run (join masking).

    Horizontal displacement bands, a slight chroma shift, dropout flecks and
    a noise veil, peaking mid-run and fading at both ends so the artifact
    reads as one tape hiccup rather than a processed boundary. Deterministic
    per seed. frames [N,H,W,C] in 0..1; returns a new tensor.
    """
    import torch
    g = torch.Generator().manual_seed(seed)
    out = frames.clone()
    N, H, W, _C = out.shape
    for i in range(N):
        amp = strength * (1.0 - abs(i - (N - 1) / 2.0) / ((N + 1) / 2.0))
        if amp <= 0:
            continue
        for _b in range(2 + int(torch.randint(0, 3, (1,), generator=g))):
            y0 = int(torch.randint(0, max(1, H - 24), (1,), generator=g))
            bh = int(torch.randint(4, 24, (1,), generator=g))
            dx = int(int(torch.randint(-40, 41, (1,), generator=g)) * amp)
            if dx:
                out[i, y0:y0 + bh] = torch.roll(out[i, y0:y0 + bh],
                                                shifts=dx, dims=1)
        dxc = int(6 * amp)
        if dxc:
            out[i, ..., 0] = torch.roll(out[i, ..., 0], dxc, dims=1)
        for _l in range(int(6 * amp)):
            y = int(torch.randint(0, H - 2, (1,), generator=g))
            x0 = int(torch.randint(0, W // 2, (1,), generator=g))
            ln = int(torch.randint(20, W // 2, (1,), generator=g))
            hot = float(torch.rand(1, generator=g)) > 0.5
            out[i, y:y + 2, x0:x0 + ln] = 0.9 if hot else 0.05
        out[i] = (out[i] + amp * 0.06 * torch.randn(
            out[i].shape, generator=g).to(out.device, out.dtype)).clamp(0, 1)
    return out


def _vhs_glitch_audio(wav, sr, at_start, seed, ms=90):
    """Tape head-switch audio hiccup: duck the signal and lay hiss over ~ms
    at the head (at_start=True) or tail of the waveform. Deterministic."""
    import torch
    g = torch.Generator().manual_seed(seed)
    n = min(int(sr * ms / 1000.0), wav.shape[-1])
    if n < 8:
        return wav
    out = wav.clone()
    seg = out[..., :n] if at_start else out[..., -n:]
    t = torch.linspace(0, 1, n)
    env = torch.sin(t * 3.14159265)          # fade the hiccup in and out
    hiss = 0.05 * torch.randn(seg.shape, generator=g).to(seg.device,
                                                         seg.dtype)
    seg = seg * (1.0 - 0.6 * env) + hiss * env
    if at_start:
        out[..., :n] = seg
    else:
        out[..., -n:] = seg
    return out


def _jb_grid(n):
    """Largest valid H3 clip length <= n: frames must satisfy n % 17 == 5."""
    n = int(n)
    if n < 5:
        return 5
    while n % 17 != 5 and n > 5:
        n -= 1
    return max(5, n)


def _jb_centre_clip(imgs, want):
    """Centre clip of `want` frames (snapped to the 17k+5 grid).

    The memory bank selects its slot around the CENTRE of the shot
    (_select_video_clip_around_frame, default mode "center"), not the tail.
    Returns (clip, start_index) so the audio window can be cut to match.
    """
    total = int(imgs.shape[0])
    n = _jb_grid(min(int(want), total))
    start = max(0, (total - n) // 2)
    return imgs[start:start + n], start
