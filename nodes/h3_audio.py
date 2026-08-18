# -*- coding: utf-8 -*-
"""Audio helper functions for H3 Multishot Advance."""

_AT_NBANDS = 8


def _at_ltas(wav, sr=32000, nfft=2048):
    """Long-term average spectrum of a [C, L] / [1, C, L] waveform."""
    import torch
    x = wav.reshape(-1, wav.shape[-1]).float().mean(0)
    n = (x.shape[-1] // nfft) * nfft
    if n < nfft:
        return None
    S = torch.stft(x[:n], nfft, hop_length=nfft // 2,
                   window=torch.hann_window(nfft), return_complex=True)
    P = (S.abs() ** 2).mean(-1)
    f = torch.linspace(0, sr / 2, P.shape[0])
    edges = torch.logspace(2, 4.08, _AT_NBANDS + 1)
    out = []
    for i in range(_AT_NBANDS):
        m = (f >= edges[i]) & (f < edges[i + 1])
        out.append(P[m].mean() if m.any() else P.new_tensor(0.0))
    return torch.stack(out).clamp_min(1e-12)


def _at_flatten(wav, house, sr=32000, nfft=2048, max_db=9.0):
    """EQ-match a shot's long-term spectral envelope to the house envelope."""
    import torch
    cur = _at_ltas(wav, sr, nfft)
    if cur is None or house is None:
        return wav, 0.0
    gain_db = (10.0 * torch.log10(house / cur)).clamp(-max_db, max_db)
    gain_db[-1] = gain_db[-1] * 0.5
    if float(gain_db.abs().max()) < 0.75:
        return wav, 0.0
    f = torch.linspace(0, sr / 2, nfft // 2 + 1)
    edges = torch.logspace(2, 4.08, _AT_NBANDS + 1)
    centres = (edges[:-1] * edges[1:]).sqrt()
    logf = torch.log10(f.clamp_min(1.0))
    logc = torch.log10(centres)
    idx = torch.bucketize(logf, logc).clamp(1, _AT_NBANDS - 1)
    x0, x1 = logc[idx - 1], logc[idx]
    w = ((logf - x0) / (x1 - x0)).clamp(0, 1)
    curve = 10.0 ** ((gain_db[idx - 1] * (1 - w) + gain_db[idx] * w) / 20.0)
    shape = wav.shape
    x = wav.reshape(-1, shape[-1]).float()
    win = torch.hann_window(nfft)
    S = torch.stft(x, nfft, hop_length=nfft // 4, window=win,
                   return_complex=True)
    S = S * curve.unsqueeze(0).unsqueeze(-1)
    y = torch.istft(S, nfft, hop_length=nfft // 4, window=win,
                    length=shape[-1])
    return y.reshape(shape).to(wav.dtype), float(gain_db.abs().max())


def _wav_for_vae(audio_vae, audio, what):
    """AUDIO dict -> [1, C, L] waveform at the VAE's own sample rate, stereo."""
    w = audio["waveform"]
    sr = int(audio["sample_rate"])
    w3 = w if w.ndim == 3 else w.unsqueeze(0)
    vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
    if sr != vae_sr:
        import torchaudio
        w3 = torchaudio.functional.resample(w3, sr, vae_sr)
        print(f"[H3] {what}: resampled {sr} -> {vae_sr} Hz", flush=True)
    if w3.shape[1] == 1:
        w3 = w3.repeat(1, 2, 1)
        print(f"[H3] {what}: mono upmixed to stereo", flush=True)
    return w3[:1], vae_sr


def _encode_ref_audio_compat(mmh3, audio_vae, audio):
    """Encode MiniMax H3 reference audio across ComfyUI API versions."""
    fn = getattr(mmh3, "_encode_ref_audio", None)
    if fn is None:
        fn = getattr(getattr(mmh3, "MiniMaxH3ReferenceToVideo", None),
                     "_encode_ref_audio", None)
    if fn is not None:
        return fn(audio_vae, audio)

    waveform, _sr = _wav_for_vae(audio_vae, audio, "ref_audio")
    z = audio_vae.encode(waveform.movedim(1, -1))
    return z, z.shape[-1]


def _smart_head_trim(wav, sr, trim, search_s=0.75):
    """Remove `trim` samples from a chained shot's audio head."""
    import torch
    n = wav.shape[-1]
    if n <= trim:
        return wav[..., :0]
    limit = min(n - trim, int(sr * search_s))
    if limit <= 1:
        return wav[..., trim:]
    mono = wav.float().abs()
    while mono.ndim > 1:
        mono = mono.mean(0)
    sq = mono[:limit + trim] ** 2
    cs = torch.cumsum(torch.cat([torch.zeros(1, device=sq.device), sq]), 0)
    win_energy = cs[trim:limit + trim] - cs[:limit]
    i = int(win_energy.argmin())
    if i > 0:
        print(f"[H3Multishot] smart weld: seam cut moved {i / sr * 1000:.0f}ms "
              f"into the head (quietest gap), word onsets preserved",
              flush=True)
    return torch.cat([wav[..., :i], wav[..., i + trim:]], dim=-1)


def _xfade_audio(parts, sr, ms=40):
    """Concatenate shot audio with a short equal-power crossfade at each seam."""
    import torch
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    n = max(1, int(sr * ms / 1000.0))
    out = parts[0]
    for nxt in parts[1:]:
        k = min(n, out.shape[-1], nxt.shape[-1])
        if k < 8:
            out = torch.cat([out, nxt], dim=-1)
            continue
        t = torch.linspace(0, 1, k, dtype=out.dtype, device=out.device)
        fade_out = torch.cos(t * 3.14159265 / 2)
        fade_in = torch.sin(t * 3.14159265 / 2)
        head, tail = out[..., :-k], out[..., -k:]
        seam = tail * fade_out + nxt[..., :k] * fade_in
        out = torch.cat([head, seam, nxt[..., k:]], dim=-1)
    return out


def _aud_env(x, win):
    """Mono windowed RMS envelope of a waveform tensor [..., T]."""
    import torch
    x = x.reshape(-1, x.shape[-1]).float().mean(dim=0)
    m = (x.shape[-1] // win) * win
    if m < win:
        return torch.zeros(1)
    return x[:m].reshape(-1, win).pow(2).mean(dim=-1).sqrt()


def _jb_audio_window(wav, sr, start_frame, num_frames, fps=24.0):
    """The audio under a clip's frame range, as an AUDIO dict."""
    a = wav if wav.ndim == 3 else wav.unsqueeze(0)
    s = int(round(start_frame / fps * sr))
    e = int(round((start_frame + num_frames) / fps * sr))
    s = max(0, min(s, a.shape[-1]))
    e = max(s + 1, min(e, a.shape[-1]))
    return {"waveform": a[..., s:e].clone(), "sample_rate": int(sr)}
