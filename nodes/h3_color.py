# -*- coding: utf-8 -*-
"""Colour, luma, and texture leveling helpers for H3 Multishot Advance."""


def _mn_normalize(parts, mode, med=9):
    """Level luma across the WHOLE assembled chain, to ONE global target."""
    import torch
    if mode == "off" or not parts:
        return parts, ""
    n = sum(int(p.shape[0]) for p in parts)
    if n < 8:
        return parts, ""
    # Per-frame statistics only - never the whole timeline as one tensor.
    luma = torch.cat([p.mean(dim=(1, 2, 3)) for p in parts])
    k = max(3, min(int(med) | 1, (n // 2) * 2 - 1))
    pad = k // 2

    def _med(v):
        return torch.nn.functional.pad(
            v[None, None], (pad, pad), mode="replicate"
        )[0, 0].unfold(0, k, 1).median(-1).values

    med_l = _med(luma)
    target = luma.median()
    gain = (target / med_l.clamp_min(1e-4)).clamp(0.70, 1.43)

    if mode == "luma+contrast":
        sd = torch.cat([p.std(dim=(1, 2, 3)) for p in parts])
        med_s = _med(sd)
        s_target = sd[:parts[0].shape[0]].median()
        cgain = (s_target / med_s.clamp_min(1e-4)).clamp(0.70, 1.43)
    else:
        cgain = None

    res, i = [], 0
    for idx in range(len(parts)):
        p = parts[idx]
        j = i + int(p.shape[0])
        g = gain[i:j, None, None, None]
        _dt = p.dtype
        _pf = p.float() if _dt != torch.float32 else p
        if cgain is not None:
            m = luma[i:j, None, None, None]
            _out = ((_pf - m) * cgain[i:j, None, None, None] + m * g).clamp(0, 1)
        else:
            _out = (_pf * g).clamp(0, 1)
        res.append(_out.to(_dt) if _dt != torch.float32 else _out)
        if _pf is not p:
            del _pf
        del _out
        parts[idx] = None
        del p
        i = j

    after = torch.cat([q.mean(dim=(1, 2, 3)) for q in res])
    msg = ("luma %.3f-%.3f -> %.3f-%.3f (target %.3f, gain %.3f-%.3f)"
           % (float(luma.min()), float(luma.max()), float(after.min()),
              float(after.max()), float(target), float(gain.min()),
              float(gain.max())))
    if mode == "luma+contrast":
        a_sd = torch.cat([q.std(dim=(1, 2, 3)) for q in res])
        sd0 = sd
        msg += ("; contrast %.4f-%.4f -> %.4f-%.4f"
                % (float(sd0.min()), float(sd0.max()),
                   float(a_sd.min()), float(a_sd.max())))
    return res, msg


def _cg_lap_var(img):
    """CONTRAST-NORMALISED texture energy of an IMAGE batch [B,H,W,C] in 0..1."""
    import torch
    import torch.nn.functional as F
    x = img if img.ndim == 4 else img.unsqueeze(0)
    g = (x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114).unsqueeze(1)
    if g.shape[-1] > 8 and g.shape[-2] > 8:
        mx_r = g.amax(dim=(0, 1, 3)) > 0.02
        mx_c = g.amax(dim=(0, 1, 2)) > 0.02
        if bool(mx_r.any()) and bool(mx_c.any()):
            g = g[:, :, mx_r][:, :, :, mx_c]
    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                     dtype=g.dtype, device=g.device).view(1, 1, 3, 3)
    lap = float(F.conv2d(g, k, padding=1).var())
    con = float(g.var())
    return lap / max(con, 1e-9)


def _cg_gauss(img, sigma):
    """Separable gaussian blur on an IMAGE batch [B,H,W,C]."""
    import torch
    import torch.nn.functional as F
    if sigma <= 0:
        return img
    r = max(1, int(round(sigma * 3)))
    xs = torch.arange(-r, r + 1, dtype=img.dtype, device=img.device)
    k = torch.exp(-(xs ** 2) / (2 * sigma * sigma))
    k = k / k.sum()
    v = img.permute(0, 3, 1, 2)
    c = v.shape[1]
    kh = k.view(1, 1, 1, -1).expand(c, 1, 1, k.numel())
    v = F.conv2d(F.pad(v, (r, r, 0, 0), mode="reflect"), kh, groups=c)
    kv = k.view(1, 1, -1, 1).expand(c, 1, k.numel(), 1)
    v = F.conv2d(F.pad(v, (0, 0, r, r), mode="reflect"), kv, groups=c)
    return v.permute(0, 2, 3, 1).clamp(0, 1)


def _cg_sigma_for(img, target_lap, max_sigma=1.6):
    """Smallest gaussian sigma bringing img texture energy down to target."""
    if target_lap <= 0:
        return 0.0
    best_s, best_d = 0.0, abs(_cg_lap_var(img) - target_lap)
    s = 0.05
    while s <= max_sigma + 1e-9:
        d = abs(_cg_lap_var(_cg_gauss(img, s)) - target_lap)
        if d < best_d:
            best_d, best_s = d, s
        s += 0.05
    return best_s


def _cg_flatten(imgs, target, block=8, max_sigma=1.6):
    """Level a shot to a constant texture energy: blur only, per block."""
    n = imgs.shape[0]
    if n == 0 or target <= 0:
        return imgs, 0.0
    idx = list(range(0, n, block))
    sig = []
    for i in idx:
        f = imgs[i:i + 1]
        sig.append(_cg_sigma_for(f, target, max_sigma)
                   if _cg_lap_var(f) > target * 1.02 else 0.0)
    sm = []
    for j in range(len(sig)):
        lo, hi = max(0, j - 1), min(len(sig), j + 2)
        sm.append(sum(sig[lo:hi]) / (hi - lo))
    if max(sm) <= 0.0:
        return imgs, 0.0
    out = imgs.clone()
    for j, i in enumerate(idx):
        s = sm[j]
        if s > 0.02:
            out[i:i + block] = _cg_gauss(imgs[i:i + block], s)
    return out, (sum(sm) / len(sm))


def _cc_stats(imgs):
    """(mu[3], cov[3,3]) of an IMAGE batch [B,H,W,C] in linear RGB."""
    x = imgs.reshape(-1, imgs.shape[-1]).clamp(0, 1) ** 2.2
    mu = x.median(dim=0).values if x.shape[0] > 1 else x.mean(dim=0)
    d = x - mu
    cov = (d.T @ d) / max(x.shape[0] - 1, 1)
    return mu, cov


def _cc_sqrtm(m):
    """Symmetric PSD matrix square root via eigendecomposition."""
    import torch
    vals, vecs = torch.linalg.eigh(m.double())
    vals = vals.clamp_min(1e-12)
    return (vecs @ torch.diag(vals.sqrt()) @ vecs.T)


def _cc_mvgd_T(cov_src, cov_dst):
    """MVGD transfer matrix: src distribution -> dst distribution."""
    import torch
    s_half = _cc_sqrtm(cov_src.double())
    s_ihalf = torch.linalg.inv(s_half)
    inner = _cc_sqrtm(s_half @ cov_dst.double() @ s_half)
    return (s_ihalf @ inner @ s_ihalf)


def _cc_apply_perframe(imgs, target_mu, strength=1.0, smooth=13):
    """Level EVERY FRAME to one fixed colour target."""
    import torch
    if strength <= 0:
        return imgs
    n = imgs.shape[0]
    lin = imgs.clamp(0, 1).double() ** 2.2
    per = lin.reshape(n, -1, imgs.shape[-1]).median(dim=1).values
    gain = (target_mu.double().view(1, 3)
            / per.clamp_min(1e-6)).clamp(0.7, 1.4)
    if not bool(torch.isfinite(gain).all()):
        return imgs
    k = int(smooth) | 1
    if n > k > 1:
        g = torch.nn.functional.pad(gain.T.unsqueeze(0), (k // 2, k // 2),
                                    mode="replicate")
        gain = torch.nn.functional.avg_pool1d(g, k, stride=1).squeeze(0).T[:n]
    out = imgs.clone()
    for i in range(0, n, 8):
        seg = imgs[i:i + 8]
        gs = gain[i:i + 8].view(-1, 1, 1, imgs.shape[-1])
        m = (((seg.clamp(0, 1).double() ** 2.2) * gs).clamp(0, 1)
             ** (1 / 2.2)).to(seg.dtype)
        out[i:i + 8] = seg + strength * (m - seg)
    return out


def _cc_apply(imgs, house_mu, house_cov, strength=1.0, block=8):
    """Level an IMAGE batch to the house colour statistics."""
    import torch
    if strength <= 0:
        return imgs
    mu_s, _cov_s = _cc_stats(imgs)
    gain = (house_mu.double() / mu_s.double().clamp_min(1e-6)).clamp(0.7, 1.4)
    if not bool(torch.isfinite(gain).all()):
        return imgs
    out = imgs.clone()
    for i in range(0, imgs.shape[0], max(1, int(block))):
        seg = imgs[i:i + block]
        lin = seg.clamp(0, 1).double() ** 2.2
        matched = ((lin * gain).clamp(0, 1) ** (1 / 2.2)).to(seg.dtype)
        out[i:i + block] = seg + strength * (matched - seg)
    return out
