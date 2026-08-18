# -*- coding: utf-8 -*-
"""Decoded-shot quality controls and continuity diagnostics."""

try:
    from .h3_audio import _aud_env
except Exception:
    from h3_audio import _aud_env

try:
    from .h3_color import (
        _cc_apply,
        _cc_stats,
        _cg_flatten,
        _cg_gauss,
        _cg_lap_var,
        _cg_sigma_for,
    )
except Exception:
    from h3_color import (
        _cc_apply,
        _cc_stats,
        _cg_flatten,
        _cg_gauss,
        _cg_lap_var,
        _cg_sigma_for,
    )

try:
    from .h3_notify import h3_warning as _h3_warning
except Exception:
    try:
        from h3_notify import h3_warning as _h3_warning
    except Exception:
        def _h3_warning(*_args, **_kwargs):
            return False


def h3_apply_shot_quality(imgs, si, color_level, chain_gain_control,
                          continuity, cg_win, house_frame, cc_mu, cc_cov,
                          cg_ref, cg_last_raw):
    # --- colour levelling to the FIXED house reference ---------------------
    # before anchor extraction and bank ingest, so the next shot's conditioning
    # inherits corrected statistics (closed loop)
    if si == 0:
        house_frame = imgs[0:1].detach().clone()
    if color_level == "mvgd":   # per-shot, rolling house reference
        if si == 0:
            cc_mu, cc_cov = _cc_stats(imgs[-min(24, imgs.shape[0]):])
            print("[H3Memory] colour house stats set (shot 1 settled tail)",
                  flush=True)
        else:
            imgs = _cc_apply(imgs, cc_mu, cc_cov)
            print("[H3Memory] colour levelled to house", flush=True)

    if (si == 0 and chain_gain_control != "off"
            and continuity in ("context_pin", "latent_handoff")):
        if (chain_gain_control == "flatten_pin"
                and continuity == "context_pin"):
            print("[H3Memory] NOTE: flatten_pin levels decoded frames and "
                  "the bank, and also softens the raw context_pin latent "
                  "before it is pinned. It is experimental and not a "
                  "guaranteed ratchet fix.", flush=True)
        else:
            print("[H3Memory] NOTE: chain_gain_control corrects the decoded "
                  "frames and the bank, but continuity=%s carries the "
                  "previous shot's RAW LATENTS, which it cannot reach - the "
                  "texture ratchet rides the pin regardless (measured +142%% "
                  "over 3 shots with flatten ON). It is effective on "
                  "frame-carried modes (first_frame, cut)." % continuity,
                  flush=True)

    if chain_gain_control != "off":
        window = min(cg_win, imgs.shape[0])
        if si == 0:
            cg_ref = _cg_lap_var(imgs[-window:])
            print(f"[H3Memory] chain: house texture level {cg_ref:.5f}",
                  flush=True)
        if cg_ref and chain_gain_control in ("flatten", "flatten_pin"):
            # the RAW tail texture of this shot, before levelling: the
            # pixel-domain ratchet flatten_pin has to undo on the pin
            cg_last_raw = _cg_lap_var(imgs[-window:]) if si > 0 else cg_ref
            imgs, sigma = _cg_flatten(imgs, cg_ref)
            if sigma > 0:
                print(f"[H3Memory] chain: levelled (sigma {sigma:.2f})",
                      flush=True)
        elif cg_ref and chain_gain_control == "match_output" and si > 0:
            if _cg_lap_var(imgs[:window]) > cg_ref * 1.05:
                sigma = _cg_sigma_for(imgs[:window], cg_ref)
                if sigma > 0:
                    imgs = _cg_gauss(imgs, sigma)

    return imgs, house_frame, cc_mu, cc_cov, cg_ref, cg_last_raw


def h3_report_continuity_diagnostics(imgs, wav, sr, continuity, si, last_tail,
                                     dbg_pins, handoff_depth, ho_wav_tail):
    if continuity == "first_frame" and si > 0 and last_tail is not None:
        # did the model actually START on the handed-over frame?
        mad = float((imgs[0].detach().cpu().float()
                     - last_tail[-1].detach().cpu().float()).abs().mean())
        print("[H3Memory] first_frame handover: frame0 vs prev last mad "
              "%.4f -> %s"
              % (mad, "HELD" if mad < 0.03 else
                 "IGNORED (wrong checkpoint? fl2va is required)"),
              flush=True)
        if mad >= 0.03:
            msg = ("first_frame handover was ignored; frame0 differs from "
                   "the previous last frame (MAD %.4f). This usually means "
                   "the wrong checkpoint is loaded; fl2va is required." % mad)
            _h3_warning(msg, topic="first_frame", tag="H3Memory")

    if dbg_pins:
        # bracket adherence: the regenerated head frames should reproduce the
        # pinned tail frames. Catches weak holds (a prompt fighting the
        # bracket) and index misalignment (each pin also scored one frame
        # early/late).
        messages = []
        for index, src in dbg_pins:
            scores = {}
            for delta in (-1, 0, 1):
                k = index + delta
                if 0 <= k < imgs.shape[0]:
                    scores[delta] = float(
                        (imgs[k].detach().cpu().float()
                         - src.float()).abs().mean())
            if scores:
                best = min(scores, key=scores.get)
                messages.append("idx %d mad %.4f (best %+d: %.4f)"
                                % (index, scores.get(0, float("nan")),
                                   best, scores[best]))
        print("[H3Memory] bracket adherence: " + "; ".join(messages),
              flush=True)
        dbg_pins = []

    if (continuity == "latent_handoff" and si > 0
            and last_tail is not None):
        # replay fidelity: the locked span should re-diffuse the previous
        # tail; a high mad means the lock is too weak (or the row mapping is
        # off).
        if handoff_depth == "block":
            prev_tail, frame_offset, key_frames = last_tail[-17:], 5, (0, 8, 16)
        else:
            prev_tail, frame_offset, key_frames = last_tail[-5:], 0, (0, 2, 4)
        messages = []
        for key in key_frames:
            if frame_offset + key < imgs.shape[0] and key < prev_tail.shape[0]:
                messages.append("f%d mad %.4f" % (
                    frame_offset + key,
                    float((imgs[frame_offset + key].detach().cpu().float()
                           - prev_tail[key].detach().cpu().float())
                          .abs().mean())))
        print("[H3Memory] handoff replay fidelity: " + "; ".join(messages),
              flush=True)
        if ho_wav_tail is not None:
            # audio replay fidelity + speech-onset check: the head of this shot
            # should be a replay of the previous tail; a speech onset inside it
            # means the trim will chop the line's opening words.
            window = max(1, int(sr * 0.02))
            count = min(ho_wav_tail.shape[-1], wav.shape[-1])
            prev_env = _aud_env(ho_wav_tail[..., :count].cpu(), window)
            new_env = _aud_env(wav[..., :count].detach().cpu(), window)
            common = min(prev_env.shape[0], new_env.shape[0])
            mad = float((prev_env[:common] - new_env[:common]).abs().mean())

            def onset(env):
                idx = (env > 0.02).nonzero()
                return (float(idx[0]) * 0.02) if idx.numel() else -1.0

            print("[H3Memory] audio replay: env mad %.4f | speech onset "
                  "prev-tail %.2fs vs new-head %.2fs"
                  % (mad, onset(prev_env[:common]), onset(new_env[:common])),
                  flush=True)
            # the number that decides the join: first speech AFTER the
            # replay+guard span. < 1.33s means the model planned speech under
            # the lock and its opening was suppressed.
            env2 = _aud_env(wav[..., :int(sr * 2.5)].detach().cpu(), window)
            post = env2[int(0.925 / 0.02):]
            post_idx = (post > 0.02).nonzero()
            t2 = (0.925 + float(post_idx[0]) * 0.02) if post_idx.numel() \
                else -1.0
            print("[H3Memory] new-line onset %.2fs (guard ends 1.33s, trim "
                  "keeps from 0.88s) -> %s"
                  % (t2, "CLEAN" if (t2 < 0 or t2 >= 1.30)
                     else "SUPPRESSED-START RISK"), flush=True)

    return dbg_pins
