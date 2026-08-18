# -*- coding: utf-8 -*-
"""Runtime continuity injection helpers for the H3 Multishot sampler."""

try:
    from .h3_notify import h3_error as _h3_error
except Exception:
    try:
        from h3_notify import h3_error as _h3_error
    except Exception:
        def _h3_error(*_args, **_kwargs):
            return False

try:
    from .h3_upscale import _upscale_av_exact
except Exception:
    from h3_upscale import _upscale_av_exact


def _is_solattn_h3_layout_wrapper(fn):
    """Is PackedLayout.__init__ wrapped by ComfyUI-SolAttn_triton's H3 hook?"""
    where = str(getattr(fn, "__module__", ""))
    globs = getattr(fn, "__globals__", {}) or {}
    return (where.endswith("._morton_h3")
            and "solattn" in where.lower()
            and "_SPANS" in globs
            and "_video_span" in globs)


def _solattn_inner_layout_init(fn):
    """Best-effort unwrap of Sol-Attn's closure-held stock constructor."""
    for cell in getattr(fn, "__closure__", ()) or ():
        try:
            val = cell.cell_contents
        except ValueError:
            continue
        if callable(val) and getattr(val, "__name__", "") == "__init__":
            return val
    return None


def h3_motion_context_allow_solattn_layout(mc_cls):
    """Allow H3 Motion-Context to compose with Sol-Attn's neutral layout hook.

    Motion-Context normally refuses any foreign PackedLayout wrapper. Sol-Attn's
    H3 wrapper is different: it calls the original constructor unchanged and
    only records the target video span for its attention hook. Letting
    Motion-Context install over that wrapper preserves both behaviors.
    """
    try:
        import comfy.ldm.minimax.model as mmodel
        import sys as sys_mod

        layout_cls = getattr(mmodel, "PackedLayout", None)
        init = getattr(layout_cls, "__init__", None)
        if (init is None
                or getattr(init, "_h3_motion_context_layout_patch", False)
                or not _is_solattn_h3_layout_wrapper(init)):
            return

        mc_mod = sys_mod.modules.get(getattr(mc_cls, "__module__", ""))
        is_applied = getattr(mc_mod, "_layout_patch_applied", None)
        if callable(is_applied) and is_applied():
            return

        apply_patch = getattr(mc_mod, "_apply_layout_patch", None)
        globals_dict = getattr(apply_patch, "__globals__", None)
        if not callable(apply_patch) or not isinstance(globals_dict, dict):
            return

        orig_already = globals_dict.get("_already_patched")
        orig_probe = globals_dict.get("_probe_frame_count")
        if not callable(orig_already):
            return

        def already_patched_solattn_ok():
            cur = getattr(getattr(mmodel, "PackedLayout", None),
                          "__init__", None)
            if _is_solattn_h3_layout_wrapper(cur):
                return None
            return orig_already()

        def probe_frame_count_solattn_aware(init_):
            if callable(orig_probe) and _is_solattn_h3_layout_wrapper(init_):
                inner = _solattn_inner_layout_init(init_)
                if inner is not None:
                    return orig_probe(inner)
            return orig_probe(init_) if callable(orig_probe) else False

        globals_dict["_already_patched"] = already_patched_solattn_ok
        if callable(orig_probe):
            globals_dict["_probe_frame_count"] = (
                probe_frame_count_solattn_aware)
        try:
            ok = apply_patch()
        finally:
            globals_dict["_already_patched"] = orig_already
            if callable(orig_probe):
                globals_dict["_probe_frame_count"] = orig_probe

        if ok:
            print("[H3Memory] context_pin: Motion-Context layout patch "
                  "stacked over Sol-Attn's H3 layout span wrapper.",
                  flush=True)
        else:
            print("[H3Memory] context_pin: Motion-Context/Sol-Attn layout "
                  "compatibility pre-apply returned false; Motion-Context "
                  "will report the exact reason.", flush=True)
    except Exception as err:
        print("[H3Memory] context_pin: could not pre-apply "
              "Motion-Context/Sol-Attn layout compatibility (%s); "
              "falling back to Motion-Context's own checks" % err,
              flush=True)


def _motion_context_class_or_fail():
    import nodes as core_nodes
    mc_cls = core_nodes.NODE_CLASS_MAPPINGS.get("MiniMaxH3MotionContext")
    if mc_cls is not None:
        return mc_cls

    # ethanfel's ComfyUI-MiniMaxH3-Contex-Loop is a fork of the same project,
    # but it deliberately does NOT re-register MiniMaxH3MotionContext - that id
    # stays with upstream, by its own design.
    fork = any(
        key in core_nodes.NODE_CLASS_MAPPINGS
        for key in ("MiniMaxH3ChainLoopStart",
                    "MiniMaxH3LoopTrim",
                    "MiniMaxH3ChainAssemble"))
    msg = (
        "continuity=context_pin needs the ComfyUI-H3-Motion-Context pack by "
        "NikoDemon80 (github.com/NikoDemon80/ComfyUI-H3-Motion-Context) - it "
        "provides the MiniMaxH3MotionContext node."
        + (" You appear to have ethanfel's ComfyUI-MiniMaxH3-Contex-Loop fork "
           "installed. That is a COMPLEMENT, not a replacement: it "
           "deliberately leaves the MiniMaxH3MotionContext id to upstream, so "
           "install NikoDemon80's pack as well - the two are designed to "
           "coexist and this pack works with either one's runtime patches."
           if fork else
           " If you installed a fork instead, note that forks may leave that "
           "node id to upstream on purpose."))
    _h3_error(msg, topic="context_pin", tag="H3Memory")
    raise RuntimeError(msg)


def _renorm_context_pin(pin_src, pin_sig0):
    # BEFORE the noise: pin_noise is scaled to sigma, so the renorm has to land
    # first or it would be measuring a sigma the noise is about to change.
    zr = pin_src["samples"]
    v0 = zr.unbind()[0] if getattr(zr, "is_nested", False) else zr
    sigma = float(v0.float().std())
    if pin_sig0 is None:
        pin_sig0 = sigma
        print("[H3Memory] context_pin: pin sigma anchor %.4f (shot 1)"
              % sigma, flush=True)
    elif sigma > 1e-6:
        gain = pin_sig0 / sigma
        if abs(gain - 1.0) > 1e-4:
            pin_src = dict(pin_src)
            if getattr(zr, "is_nested", False):
                import comfy.nested_tensor as nested_tensor
                comps = list(zr.unbind())
                comps[0] = (comps[0].float() * gain).to(comps[0].dtype)
                pin_src["samples"] = nested_tensor.NestedTensor(comps)
            else:
                pin_src["samples"] = (zr.float() * gain).to(zr.dtype)
            print("[H3Memory] context_pin: pin renormed sigma %.4f -> %.4f "
                  "(x%.4f)" % (sigma, pin_sig0, gain), flush=True)
    return pin_src, pin_sig0


def _flatten_context_pin(pin_src, pin_hf0, cg_ref, cg_last_raw):
    # THE OTHER HALF OF FLATTEN. chain_gain_control=flatten levels decoded
    # frames and the bank, but context_pin carries the previous shot's RAW
    # LATENTS - and their accreted fine detail is what the next shot conditions
    # on. Level the pin's high-frequency energy to shot 1's tail before it is
    # pinned. Video half only. After renorm, before pin_noise.
    import torch.nn.functional as functional

    zr = pin_src["samples"]
    nested = getattr(zr, "is_nested", False)
    v0 = zr.unbind()[0] if nested else zr
    vf = v0.float()
    batch, channels, timesteps, height, width = vf.shape
    flat = vf.permute(0, 2, 1, 3, 4).reshape(
        -1, channels, height, width)
    low = functional.avg_pool2d(
        flat, 3, stride=1, padding=1, count_include_pad=False)
    high = flat - low
    tail = max(1, min(timesteps, 8))
    high_tail = high.reshape(
        batch, timesteps, channels, height, width)[:, -tail:]
    energy = float(high_tail.pow(2).mean())
    if pin_hf0 is None:
        pin_hf0 = energy
        print("[H3Memory] context_pin: fine-detail anchor %.5f "
              "(shot 1 tail)" % energy, flush=True)
    elif energy > 1e-12:
        latent_gain = (pin_hf0 / energy) ** 0.5
        pixel_gain = 1.0
        try:
            if cg_ref and cg_last_raw and cg_last_raw > cg_ref:
                pixel_gain = (float(cg_ref) / float(cg_last_raw)) ** 0.5
        except Exception:
            pixel_gain = 1.0
        gain = max(0.5, min(1.0, latent_gain, pixel_gain))
        print("[H3Memory] context_pin: flatten_pin gains - latent %.3f, "
              "pixel %.3f -> using %.3f"
              % (latent_gain, pixel_gain, gain), flush=True)
        if gain < 0.999:
            new_video = (low + high * gain).reshape(
                batch, timesteps, channels, height, width)
            new_video = new_video.permute(0, 2, 1, 3, 4).to(v0.dtype)
            pin_src = dict(pin_src)
            if nested:
                import comfy.nested_tensor as nested_tensor
                comps = list(zr.unbind())
                comps[0] = new_video
                pin_src["samples"] = nested_tensor.NestedTensor(comps)
            else:
                pin_src["samples"] = new_video
            print("[H3Memory] context_pin: flatten_pin - pin fine-detail "
                  "energy %.5f vs anchor %.5f -> high-pass x%.3f "
                  "(only softens)" % (energy, pin_hf0, gain), flush=True)
        else:
            print("[H3Memory] context_pin: flatten_pin - pin already at "
                  "anchor (%.5f vs %.5f), untouched"
                  % (energy, pin_hf0), flush=True)
    return pin_src, pin_hf0


def _noise_context_pin(pin_src, pin_noise, shot_seed):
    # noised clean condition, applied to the carrier. The pin is this model's
    # own output; left pristine the model treats it as ground truth and adds
    # detail on top of detail, once per hop. Seeded so same-seed A/B arms are
    # comparable. Only the VIDEO half is noised; noising the audio component
    # dulls the voice.
    import torch

    t = float(pin_noise)
    sigmas = []

    def noise_one(z):
        # Variance-preserving, scaled to the latent's OWN standard deviation.
        generator = torch.Generator(device=z.device).manual_seed(
            shot_seed ^ 0x91EE)
        sigma = z.float().std()
        sigmas.append(float(sigma))
        noise = torch.randn(
            z.shape, generator=generator, device=z.device,
            dtype=torch.float32)
        out = ((1.0 - t * t) ** 0.5) * z.float() + t * sigma * noise
        return out.to(z.dtype)

    samples = pin_src["samples"]
    pin_src = dict(pin_src)
    if getattr(samples, "is_nested", False):
        import comfy.nested_tensor as nested_tensor
        comps = list(samples.unbind())
        comps[0] = noise_one(comps[0])
        pin_src["samples"] = nested_tensor.NestedTensor(comps)
        what = "video half of the AV pin"
    else:
        pin_src["samples"] = noise_one(samples)
        what = "pin"
    print("[H3Memory] context_pin: %s noised %.3f of its own sigma=%.4f "
          "(anti-ratchet, variance-preserving)"
          % (what, t, sigmas[0] if sigmas else float("nan")),
          flush=True)
    return pin_src


def h3_apply_context_pin(cond, continuity, si, cp_prev, video_vae, latent,
                         two_pass_upscale, tp_tr, tp_lat_h1, tp_lat_w1,
                         pin_frames, pin_renorm, pin_sig0,
                         chain_gain_control, pin_hf0, cg_ref, cg_last_raw,
                         pin_noise, shot_seed, audio_pin_frames):
    if not (continuity == "context_pin" and si > 0 and cp_prev is not None):
        return cond, 0, pin_sig0, pin_hf0

    mc_cls = _motion_context_class_or_fail()
    h3_motion_context_allow_solattn_layout(mc_cls)

    # The pin is raw latents from the PREVIOUS shot, sampled at full
    # resolution. Pass 1 runs on a smaller grid, so the pin has to be resampled
    # onto that grid or the shapes do not meet.
    pin_src = cp_prev
    if two_pass_upscale and cp_prev is not None:
        pin_src = _upscale_av_exact(tp_tr, cp_prev, tp_lat_h1, tp_lat_w1)
        print("[H3Memory] two-pass: pin resampled to the pass-1 grid "
              "(%dx%d latent)" % (tp_lat_h1, tp_lat_w1), flush=True)

    picture_frames = (str(pin_frames) if str(pin_frames) in
                      ("5", "22", "39", "56") else "22")

    if pin_renorm and isinstance(pin_src, dict) and "samples" in pin_src:
        pin_src, pin_sig0 = _renorm_context_pin(pin_src, pin_sig0)

    if (chain_gain_control == "flatten_pin"
            and isinstance(pin_src, dict)
            and "samples" in pin_src):
        pin_src, pin_hf0 = _flatten_context_pin(
            pin_src, pin_hf0, cg_ref, cg_last_raw)

    if pin_noise > 0 and isinstance(pin_src, dict) and "samples" in pin_src:
        pin_src = _noise_context_pin(pin_src, pin_noise, shot_seed)

    audio_frames = int(audio_pin_frames or 0) or int(picture_frames)
    if audio_frames != int(picture_frames):
        print("[H3Memory] context_pin: audio reference window %d frames "
              "(%.1f s), picture pin %s frames - the head trim follows the "
              "picture pin only."
              % (audio_frames, audio_frames / 24.0, picture_frames),
              flush=True)

    cond, cp_trim = mc_cls().apply(
        conditioning=cond, vae=video_vae, latent=latent,
        context_length=picture_frames, audio_context_length=audio_frames,
        context_latent=pin_src)
    print("[H3Memory] context_pin: previous shot's tail pinned as raw latents "
          "(%sf video + %df audio ref / %.1f s, trim %d on decode)"
          % (picture_frames, audio_frames, audio_frames / 24.0, cp_trim),
          flush=True)
    return cond, cp_trim, pin_sig0, pin_hf0


def h3_patch_handoff_guider(guider, model, latent, spine, si, frames_per_shot,
                            trim, continuity, ho_v, ho_a, ho_guard,
                            audio_lock, ho_r0, handoff_taper,
                            ho_taper_src, handoff_release):
    if not (spine is not None
            or (continuity == "latent_handoff" and ho_v is not None)):
        return guider

    # one denoise trajectory: every model call sees the previous shot's tail
    # latents, renoised to the CURRENT sigma, sitting in the overlap slots of
    # both streams. The prediction then continues that state.
    import torch
    from comfy.ldm.minimax.model import time_shift_sigma as time_shift_sigma

    model_sampling = model.get_model_object("model_sampling")
    try:
        audio_scale = float(
            getattr(model_sampling, "audio_scale", 1.0) or 1.0)
    except Exception:
        audio_scale = 1.0
    if abs(audio_scale - 1.0) > 1e-6:
        print("[H3Memory] audio injections scaled x%.3f onto the sampler's "
              "audio domain (ModelSamplingAV)" % audio_scale, flush=True)

    diffusion_model = getattr(getattr(model, "model", None),
                              "diffusion_model", None)
    sigma_shift_video = float(
        getattr(diffusion_model, "sigma_shift_video", 12.0))
    sigma_shift_audio = float(
        getattr(diffusion_model, "sigma_shift_audio", 3.0))
    orig_predict_noise = guider.predict_noise
    comps0 = latent["samples"].unbind()
    video_shape = tuple(comps0[0].shape)
    audio_shape = tuple(comps0[1].shape) if len(comps0) > 1 else None
    spine_segment = None
    if spine is not None and audio_shape is not None:
        # this shot's time-slice of the spine: shots advance by
        # (frames_per_shot - trim) in output time, and the trim depends on the
        # continuity mode.
        audio_start = int(round(si * (frames_per_shot - trim) / 24.0 * 40.0))
        audio_start = max(0, min(
            audio_start, max(0, spine.shape[-1] - 1)))
        spine_segment = spine[..., audio_start:audio_start + audio_shape[-1]]

    video_size = 1
    for dim in video_shape[1:]:
        video_size *= dim
    audio_size = 0
    if audio_shape is not None:
        audio_size = 1
        for dim in audio_shape[1:]:
            audio_size *= dim

    audio_locked = bool(audio_lock) or spine_segment is not None

    def patched_predict_noise(x, timestep, model_options={}, seed=None,
                              _orig=orig_predict_noise, _hv=ho_v, _ha=ho_a,
                              _hg=ho_guard, _hs=spine_segment,
                              _ms=model_sampling, _al=audio_locked,
                              _r0=ho_r0, _tss=time_shift_sigma,
                              _shv=sigma_shift_video,
                              _sha=sigma_shift_audio,
                              _tp=int(handoff_taper),
                              _tsrc=ho_taper_src,
                              _rel=float(handoff_release),
                              _vs=video_shape, _as=audio_shape,
                              _Nv=video_size, _Na=audio_size,
                              _asc=audio_scale,
                              _state={"logged": False}):
        try:
            sig = float(timestep.flatten()[0])
        except Exception:
            sig = float(timestep)
        if (x.ndim == 3 and x.shape[1] == 1 and x.shape[2] >= _Nv + _Na):
            try:
                x = x.clone()
                step_video = torch.tensor([sig], device=x.device,
                                          dtype=x.dtype)
                sig_audio = _tss(sig, _shv, _sha)
                step_audio = torch.tensor([sig_audio], device=x.device,
                                          dtype=x.dtype)
                if sig > _rel and _hv is not None:
                    xv = x[:, 0, :_Nv].reshape((x.shape[0],) + _vs[1:])
                    tv = _hv.to(device=x.device, dtype=x.dtype)
                    xv[:, :, _r0:_r0 + tv.shape[2]] = _ms.noise_scaling(
                        step_video, torch.randn_like(tv), tv)
                    if _tp > 0 and _tsrc is not None:
                        # graded taper: bias the rows AFTER the hard lock
                        # toward the previous tail at linearly decaying
                        # strength, so the pose has a ramp instead of a cliff.
                        t0 = _r0 + tv.shape[2]
                        tn = min(_tp, xv.shape[2] - t0)
                        if tn > 0:
                            tsrc = _tsrc.to(device=x.device, dtype=x.dtype)
                            for index in range(tn):
                                weight = (tn - index) / (tn + 1.0)
                                target = _ms.noise_scaling(
                                    step_video, torch.randn_like(tsrc),
                                    tsrc)[:, :, 0]
                                xv[:, :, t0 + index] = (
                                    (1.0 - weight) * xv[:, :, t0 + index]
                                    + weight * target)

                # AUDIO stays locked through EVERY step: video release exists
                # so detail steps can reconcile texture, but audio content must
                # be exact.
                if _Na and _hs is not None:
                    # spine mode: the WHOLE audio stream is a locked slice of
                    # one continuous track - nothing left for the model to plan.
                    xa = x[:, 0, _Nv:_Nv + _Na].reshape(
                        (x.shape[0],) + _as[1:])
                    ts = _hs.to(device=x.device, dtype=x.dtype) * _asc
                    count = min(ts.shape[-1], xa.shape[-1])
                    xa[..., :count] = _ms.noise_scaling(
                        step_audio, torch.randn_like(ts[..., :count]),
                        ts[..., :count])
                elif _Na and _al and _ha is not None:
                    xa = x[:, 0, _Nv:_Nv + _Na].reshape(
                        (x.shape[0],) + _as[1:])
                    ta = _ha.to(device=x.device, dtype=x.dtype) * _asc
                    if _hg is not None:
                        ta = torch.cat(
                            [ta, _hg.to(device=x.device,
                                        dtype=x.dtype) * _asc],
                            dim=-1)
                    xa[..., :ta.shape[-1]] = _ms.noise_scaling(
                        step_audio, torch.randn_like(ta), ta)
                if not _state["logged"]:
                    _state["logged"] = True
                    print("[H3Memory] handoff injection ACTIVE "
                          "(packed path, sigma %.3f)" % sig, flush=True)
            except Exception as err:
                print("[H3Memory] handoff injection FAILED: %r" % (err,),
                      flush=True)
        return _orig(x, timestep, model_options=model_options, seed=seed)

    guider.predict_noise = patched_predict_noise
    print("[H3Memory] latent handoff armed: %d video rows, %d audio cols + "
          "%d guard cols%s, release below sigma %.2f"
          % (0 if ho_v is None else ho_v.shape[2],
             0 if ho_a is None else ho_a.shape[-1],
             0 if ho_guard is None else ho_guard.shape[-1],
             "" if spine_segment is None else
             " + SPINE %d cols" % spine_segment.shape[-1],
             float(handoff_release)),
          flush=True)
    return guider
