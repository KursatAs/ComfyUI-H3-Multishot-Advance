# -*- coding: utf-8 -*-
"""Pre-render setup helpers for the H3 Multishot memory sampler."""

try:
    from .h3_notify import (
        h3_error as _h3_error,
        h3_info as _h3_info,
        h3_warning as _h3_warning,
    )
except Exception:
    try:
        from h3_notify import (
            h3_error as _h3_error,
            h3_info as _h3_info,
            h3_warning as _h3_warning,
        )
    except Exception:
        def _h3_noop(*_args, **_kwargs):
            return False
        _h3_info = _h3_warning = _h3_error = _h3_noop

try:
    from .h3_upscale import (
        _load_up_model,
        _load_upscaler_utils,
        _up_model_factor,
    )
except Exception:
    from h3_upscale import (
        _load_up_model,
        _load_upscaler_utils,
        _up_model_factor,
    )


def _h3_fail(message, exc_type=RuntimeError, title="H3 Multishot", tag=None):
    msg = str(message)
    _h3_error(msg, title=title, tag=tag)
    return exc_type(msg)


def h3_apply_sampler_overrides(sampler_name, scheduler, sampler_override,
                               scheduler_override):
    if sampler_override and str(sampler_override).strip():
        sampler_name = str(sampler_override).strip()
    if scheduler_override and str(scheduler_override).strip():
        scheduler = str(scheduler_override).strip()
    return sampler_name, scheduler


def h3_prepare_shots(script, shot_count, parse_script):
    shots = parse_script(script)
    n = shot_count if shot_count > 0 else len(shots)
    if len(shots) > n:
        shots = shots[:n]
    while len(shots) < n:
        shots.append(shots[-1])
    return shots, n


def h3_prepare_sigmas_and_sampler(model, scheduler, steps, sigmas,
                                  sampler_name, ncs):
    if sigmas is not None and len(sigmas) > 1:
        # a supplied schedule wins: some turbo LoRAs only converge on the
        # exact curve they shipped with, and silently re-deriving one from
        # steps+scheduler would make them look broken rather than misused
        steps = int(len(sigmas)) - 1
        print("[%s] custom sigmas supplied (%d steps): the steps and "
              "scheduler widgets are ignored for this run."
              % ("H3Memory", steps), flush=True)
    else:
        sigmas = ncs.BasicScheduler().get_sigmas(model, scheduler,
                                                 steps, 1.0)[0]
    sampler = ncs.KSamplerSelect().get_sampler(sampler_name)[0]
    return steps, sigmas, sampler


def h3_prepare_two_pass(two_pass_upscale, continuity, width, height,
                        frames_per_shot, steps, sigmas, upscale_factor,
                        pass1_fraction, upscale_audio_denoise, mmh3):
    # The raw-latent continuity modes pin the previous shot's latents into
    # THIS shot's grid. Pass 1 runs on a smaller grid, so the pin does not
    # fit - and resampling it would destroy the bit-identical hand-off that
    # is the entire reason those modes exist. Refuse rather than degrade.
    _tp_tr = _tp_w1 = _tp_h1 = _tp_sig_hi = _tp_sig_lo = None
    _tp_lat_th = _tp_lat_tw = None
    _tp_lat_h1 = _tp_lat_w1 = None   # pass-1 latent grid (pin target)
    if two_pass_upscale:
        if continuity == "latent_handoff":
            _msg = (
                "two_pass_upscale is not compatible with continuity="
                "'latent_handoff'. That mode renoises the previous tail "
                "into the CURRENT trajectory at every step, and a "
                "two-pass render is two trajectories - the handoff would "
                "be dropped halfway. Use continuity=context_pin, which "
                "does support two_pass_upscale (the pin is resampled "
                "onto the pass-1 grid), or turn two_pass_upscale off.")
            raise _h3_fail(_msg, ValueError, "H3 incompatible settings",
                           tag="H3Memory")
        _tp_tr = _load_upscaler_utils()
        _f = max(1.0, float(upscale_factor))
        _tp_w1 = max(32, int(round(width / _f / 32)) * 32)
        _tp_h1 = max(32, int(round(height / _f / 32)) * 32)
        _k = int(round(steps * float(pass1_fraction)))
        _k = max(1, min(steps - 1, _k))
        _tp_sig_hi, _tp_sig_lo = sigmas[:_k + 1], sigmas[_k:]
        _probe, _ = mmh3._empty_av_latent(width, height, frames_per_shot)
        _pv = _probe["samples"]
        _pv = _pv.unbind()[0] if getattr(_pv, "is_nested", False) else _pv
        _tp_lat_th, _tp_lat_tw = int(_pv.shape[-2]), int(_pv.shape[-1])
        del _probe, _pv
        print(f"[H3Memory] two-pass: {_tp_w1}x{_tp_h1} for {_k} steps -> "
              f"latent x{width / _tp_w1:.2f} -> {width}x{height} for "
              f"{steps - _k} steps (audio_denoise "
              f"{upscale_audio_denoise}).", flush=True)
    return (_tp_tr, _tp_w1, _tp_h1, _tp_sig_hi, _tp_sig_lo,
            _tp_lat_th, _tp_lat_tw, _tp_lat_h1, _tp_lat_w1)


def h3_reference_cap(bank_pinned, memory_frames):
    # H3 allows at most 3 video references, so the bank is capped there.
    return max(1, min(3, int(bank_pinned) + int(memory_frames)))


def _has_batch(x):
    try:
        return x is not None and int(x.shape[0]) > 0
    except Exception:
        return x is not None


def h3_reference_rows_expected(n, cap, reference_images, reference_video,
                               voice_ref, start_image, self_anchor_voice):
    # KursatAs 2026-08-17 09:18: checkpoint warnings must follow real ref-row
    # sources, not only the memory_frames widget.
    return (
        (n > 1 and cap > 0)
        or _has_batch(reference_images)
        or _has_batch(reference_video)
        or voice_ref is not None
        or start_image is not None
        or (self_anchor_voice and n > 1)
    )


def h3_report_checkpoint_setup(model, continuity, ref_rows_expected,
                               start_image):
    # task/checkpoint guard: continuity mode dictates the checkpoint.
    ckpt = str(getattr(getattr(model, "model", None),
                       "h3_checkpoint_name", "") or "").lower()
    is_fl = "fl2va" in ckpt
    is_ref = "ref2va" in ckpt
    if ckpt:
        if continuity == "first_frame" and is_fl:
            msg = ("fl2va quality path active: continuity=first_frame is "
                   "the intended hand-off mode for this checkpoint. "
                   "Reference-row features are not required for this path.")
            print("[H3Memory] " + msg, flush=True)
            _h3_info(msg, topic="fl2va_quality",
                     tag="H3Memory", timeout_ms=6000)
        elif continuity == "first_frame" and is_ref:
            msg = ("continuity=first_frame hands the previous last frame "
                   "over as the fl2va task, but a ref2va checkpoint is "
                   "loaded. The hand-off will be weak (soft keyframe "
                   "only). Load an fl2va checkpoint.")
            print("[H3Memory] WARNING: " + msg, flush=True)
            _h3_warning(msg, topic="checkpoint", tag="H3Memory")
        elif continuity != "first_frame" and is_fl and ref_rows_expected:
            msg = ("fl2va checkpoint loaded: this is the quality/first-"
                   "frame path, but the current workflow also enables "
                   "ref2va-only reference rows (memory bank, reference "
                   "images, start_image identity refs, or voice/video "
                   "refs). Those slots will be ignored by fl2va. Use "
                   "continuity=first_frame for the fl2va path, or load "
                   "ref2va when you want reference-row conditioning.")
            print("[H3Memory] WARNING: " + msg, flush=True)
            _h3_warning(msg, topic="checkpoint", tag="H3Memory")
        if start_image is not None and is_fl:
            # Reported from the field: start_image connected, fl2va loaded,
            # continuity=first_frame, and shot 1 does not open on the supplied
            # picture. Nothing was misconfigured - this input is an identity
            # REFERENCE ROW here, and fl2va has no reference rows, so it is
            # built and then ignored.
            msg = ("start_image on this node is an identity/reference "
                   "image, not a first-frame input. fl2va has no "
                   "reference rows, so this input has no effect on the "
                   "fl2va path and shot 1 will not open on it. To force "
                   "shot 1 to start on a specific picture in this Advance "
                   "pack, use continuity=flf_chain with keyframe_images "
                   "(N+1 stills for N shots). To use start_image as an "
                   "identity reference, load a ref2va checkpoint.")
            print("[H3Memory] NOTE: " + msg, flush=True)
            _h3_info(msg, topic="start_image", tag="H3Memory")
    return ckpt, is_fl, is_ref


def h3_make_stream_assembler(low_ram_master, join_fx, color_level):
    if not bool(low_ram_master):
        return None
    blockers = []
    # join_blend is NOT a blocker: it only runs in seamless_tail/
    # latent_handoff, and the stream assembler defers each shot one step so
    # the blend can still mutate the previous tail before staging.
    if str(join_fx) not in ("off", "none", ""):
        blockers.append("join_fx")
    if str(color_level) == "scene":
        blockers.append("color_level=scene")
    if blockers:
        print("[H3Memory] low_ram_master: falling back to the RAM path - "
              "%s read neighbouring shots and are not streamable in v1."
              % "+".join(blockers), flush=True)
        return None
    try:
        import os
        try:
            from .h3_stream_master import ShotStreamAssembler
        except ImportError:
            import importlib.util as _ilu
            _sp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "h3_stream_master.py")
            _spec = _ilu.spec_from_file_location("h3_stream_master", _sp)
            _mod = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            ShotStreamAssembler = _mod.ShotStreamAssembler
        import folder_paths as _fp
        _tdir = os.path.join(_fp.get_output_directory(), "video",
                             "H3CHAIN_STREAM", "tmp_%d" % os.getpid())
        stream_assembler = ShotStreamAssembler(_tdir, fps=24)
        print("[H3Memory] low_ram_master ON: shots stage to %s, peak host "
              "RAM = one shot." % _tdir, flush=True)
        return stream_assembler
    except Exception as e:  # noqa: BLE001
        print("[H3Memory] low_ram_master unavailable (%s) - using the RAM "
              "path." % e, flush=True)
        return None


def h3_load_upscale_model(upscale_model_name, output_scale, width, height,
                          frames_per_shot, shot_count):
    if upscale_model_name in ("(none)", "", None):
        return None
    # load it NOW, before a single step is sampled. The first version of this
    # loader raised inside the per-shot upscale, i.e. AFTER shot 1 had rendered.
    try:
        upscale_model = _load_up_model(upscale_model_name)
    except Exception as e:
        msg = (
            "upscale_model_name=%r could not be loaded: %s. Fix the name or "
            "set it to (none) - failing now rather than after the first shot "
            "has rendered." % (upscale_model_name, e))
        raise _h3_fail(msg, RuntimeError, "H3 upscale model", tag="H3Memory")
    print("[H3Memory] upscale model: %s (loaded and verified)"
          % upscale_model_name, flush=True)
    # Predict the cost NOW, not after the first shot has been sampled and
    # upscaled. Frames accumulate on the host until the master is joined, so
    # the total is per-shot x shots.
    try:
        import psutil as _ps
        ram = _ps.virtual_memory().total / 2**30
    except Exception:
        ram = 0.0
    f = (float(output_scale) if output_scale
         and abs(float(output_scale) - 1.0) > 1e-6
         else _up_model_factor(upscale_model))
    ow, oh = int(round(width * f)), int(round(height * f))
    per = ow * oh * 3 * 2 / 2**30          # fp16 on the host
    tot = per * frames_per_shot * max(1, shot_count)
    print("[H3Memory] upscale will produce %dx%d (%.1fx): %.0f MB per "
          "frame, %.1f GB per shot, %.1f GB for %d shot(s)%s"
          % (ow, oh, f, per * 1024, per * frames_per_shot, tot,
             max(1, shot_count),
             (" against %.1f GB of RAM" % ram) if ram else ""),
          flush=True)
    if ram and tot > ram * 0.8:
        limit = max(
            1.0,
            (ram * 0.7 / (frames_per_shot * max(1, shot_count)
                          * width * height * 3 * 2 / 2**30)) ** 0.5)
        msg = ("that will not fit. Frames are held in system RAM until the "
               "master is joined, so this dies at the join after every shot "
               "has been paid for. Set output_scale to %.2f or lower, or "
               "turn the upscaler off." % limit)
        print("[H3Memory] WARNING: " + msg, flush=True)
        _h3_warning(msg, topic="ram", tag="H3Memory")
    return upscale_model


def h3_handoff_geometry(handoff_depth, continuity):
    tail_k = 2            # bracket depth: pixel frames -9/-5/-1 -> idx 0/4/8
    ov = 1 + 4 * tail_k   # overlap frames regenerated by the next shot
    # latent_handoff geometry. Video latent: 2 bootstrap rows for the first 5
    # frames, then 5 rows per 17-frame block. The bootstrap rows have a
    # different encoding structure than block rows, so the lock covers the
    # first FULL block and the free 5-frame head is warmup that gets trimmed.
    ho_rows = 5 if handoff_depth == "block" else 2
    ho_r0 = 2 if handoff_depth == "block" else 0
    ho_acols = 37 if handoff_depth == "block" else 8
    ov_ho = 22 if handoff_depth == "block" else 5
    ho_guard = 16         # onset-guard cols (0.4s of locked room tone)
    trim = {"cut": 0, "seamless": 1, "first_frame": 1, "flf_chain": 1,
            "seamless_tail": ov, "latent_handoff": ov_ho,
            "context_pin": 22}.get(continuity, 0)
    return {
        "tail_k": tail_k,
        "ov": ov,
        "ho_rows": ho_rows,
        "ho_r0": ho_r0,
        "ho_acols": ho_acols,
        "ov_ho": ov_ho,
        "ho_guard": ho_guard,
        "trim": trim,
    }


def h3_report_chain_setup(chain_gain_control, continuity, n, bank_pinned, cap,
                          is_fl, bank_clip_frames, jb_grid):
    if (chain_gain_control == "flatten_pin"
            and continuity == "context_pin" and n > 1):
        # KursatAs 2026-08-17 20:48: flatten_pin is useful for A/B tests, but
        # it touches the raw context_pin latent before sampling and can add to
        # the same VRAM peak we are trying to reduce.
        msg = ("chain_gain_control=flatten_pin is experimental and can raise "
               "VRAM because it high-pass filters the raw context_pin latent "
               "before sampling. Default remains off; switch to flatten or "
               "off if VRAM climbs.")
        print("[H3Memory] WARNING: " + msg, flush=True)
        _h3_warning(msg, topic="experimental", tag="H3Memory")

    if bank_pinned == 0 and n > 4:
        msg = ("bank_pinned=0 on a %d-shot chain. With no pinned slot the "
               "conditioning is pure recency - each shot hears only the one "
               "before it - and audio COLLAPSES: measured 84-92%% loss of "
               "4-10 kHz energy by shot 8 (five-arm A/B, 2026-08-11). Set "
               "bank_pinned=1, and keep chains short if the voice matters."
               % n)
        print("[H3Memory] WARNING: " + msg, flush=True)
        _h3_warning(msg, topic="continuity", tag="H3Memory")
    if continuity == "first_frame":
        print("[H3Memory] first_frame mode: memory-bank reference rows are "
              "bypassed; continuity comes from the fl2va previous-frame "
              "handoff.", flush=True)
    else:
        bank_suffix = (
            "active ref2va reference-row path."
            if not is_fl else
            "configured, but fl2va ignores reference-row bank slots unless "
            "you switch to ref2va."
        )
        print(f"[H3Memory] memory-bank mode: no keyframe, "
              f"{bank_pinned} pinned + {cap - min(bank_pinned, cap)} recent "
              f"slot(s), {jb_grid(bank_clip_frames)}f clips; {bank_suffix}",
              flush=True)
