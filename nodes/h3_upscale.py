# -*- coding: utf-8 -*-
"""Upscale and per-shot output helpers for H3 Multishot Advance."""

try:
    from .h3_notify import h3_error as _h3_error
except Exception:
    try:
        from h3_notify import h3_error as _h3_error
    except Exception:
        def _h3_error(*_args, **_kwargs):
            return False


def _h3_fail(message, exc_type=RuntimeError, title="H3 Multishot", tag=None):
    msg = str(message)
    _h3_error(msg, title=title, tag=tag)
    return exc_type(msg)


def _up_model_list():
    try:
        import folder_paths
        return folder_paths.get_filename_list("upscale_models")
    except Exception:
        return []


def _load_up_model(name):
    """Load an upscale model by NAME, through ComfyUI's own loader node.

    Do not reimplement this. The first version here copied the loader's body
    from an older ComfyUI - state dict, spandrel, eval() - and missed that the
    current one also attaches a CoreModelPatcher, which ImageUpscaleWithModel
    then reads as upscale_model.patcher.load_device. It crashed AFTER shot 1
    had rendered. Calling the real node means this cannot drift out of sync
    with whatever ComfyUI does next.
    """
    from comfy_extras.nodes_upscale_model import UpscaleModelLoader
    out = UpscaleModelLoader().load_model(name)
    # the V3 node API returns a NodeOutput; older returns a plain tuple
    for attr in ("result", "results"):
        if hasattr(out, attr):
            out = getattr(out, attr)
            break
    while isinstance(out, (tuple, list)):
        out = out[0]
    if not hasattr(out, "patcher"):
        _msg = (
            "upscale_model_name=%r loaded, but the result has no .patcher - "
            "ComfyUI's upscale loader has changed shape again. Wire a Load "
            "Upscale Model node instead, or set upscale_model_name to "
            "(none)." % name)
        raise _h3_fail(_msg, RuntimeError, "H3 upscale model",
                       tag="H3Upscale")
    return out


def _write_shot_mp4(imgs, wav, sr, prefix, label, tag):
    """Write one decoded shot to disk immediately, and never let that kill
    the render.

    A long chain represents hours of GPU time that only becomes a file at
    the very end, when the master is muxed. Anything that fails after the
    last shot - a mux OOM, a full disk, a cancelled tab - has historically
    destroyed every shot at once (issue #13). Each shot written as it
    decodes turns that from lost work into a joining job.

    Returns the path written, or None (already reported) on failure.
    """
    try:
        import os
        from fractions import Fraction
        import folder_paths
        from comfy_api.latest import InputImpl, Types
        w = wav if wav.ndim == 3 else wav.unsqueeze(0)
        folder, fname, counter, _sub, _pfx = folder_paths.get_save_image_path(
            prefix, folder_paths.get_output_directory(),
            imgs.shape[2], imgs.shape[1])
        path = os.path.join(folder, f"{fname}_{counter:05}_.mp4")
        InputImpl.VideoFromComponents(Types.VideoComponents(
            images=imgs.detach().cpu(),
            audio={"waveform": w.detach().cpu(), "sample_rate": sr},
            frame_rate=Fraction(24))).save_to(path)
        print(f"[{tag}] {label} -> {path}", flush=True)
        return path
    except Exception as e:
        print(f"[{tag}] {label} FAILED to save (render continues): {e}",
              flush=True)
        return None


def _up_model_factor(model, default=4.0):
    """The fixed enlargement factor of a loaded upscale model.

    Needed to predict output size before anything is upscaled. ESRGAN-family
    models expose it as `scale`; fall back to 4x, which is what almost every
    model in circulation is, rather than refusing to predict.
    """
    for attr in ("scale", "scale_factor", "upscale_factor"):
        v = getattr(model, attr, None)
        if v is None:
            v = getattr(getattr(model, "model", None), attr, None)
        try:
            if v and float(v) > 0:
                return float(v)
        except (TypeError, ValueError):
            pass
    return default


def _upscale_frames(imgs, scale, model, tag):
    """Enlarge decoded frames, AFTER sampling. Pixels, never latents.

    The old two-pass path interpolated the raw latent between passes; H3's
    latent is not spatially smooth, so that landed off-manifold and produced
    colour noise no matter how the schedule was split. Anything done here is
    downstream of the VAE, so it cannot leave the manifold - the worst case is
    a soft picture, not a broken one.

    Per shot, so peak memory is one shot's frames rather than the whole chain.
    """
    if model is None and (not scale or abs(scale - 1.0) < 1e-6):
        return imgs
    import comfy.utils
    h, w = int(imgs.shape[1]), int(imgs.shape[2])
    if model is not None:
        from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
        import comfy.model_management as _mm
        import os as _os
        import torch as _t
        # Chunked. One call with every frame allocates the whole upscaled batch
        # in fp32 on the CPU on top of the input - 11 GB for 243 frames at
        # 1472x2560, which is where a 12-shot 736x1280 run died after two hours.
        # Writing into a preallocated output holds (output + one chunk) instead.
        _n = int(imgs.shape[0])
        _ch = max(1, int(_os.environ.get("H3_UPSCALE_CHUNK", "16")))
        if _n > _ch:
            _out = None
            for _s in range(0, _n, _ch):
                _p = ImageUpscaleWithModel().upscale(model, imgs[_s:_s + _ch])[0]
                if _out is None:
                    _out = _t.empty((_n,) + tuple(_p.shape[1:]), dtype=_p.dtype)
                _out[_s:_s + int(_p.shape[0])] = _p
                del _p
            imgs = _out
        else:
            imgs = ImageUpscaleWithModel().upscale(model, imgs)[0]
        # Free it AT ONCE. Left resident it survives into the next shot, and
        # the DiT can then only load partially - measured 18.5 s/it on shot 1
        # (full load) against 349 s/it on shot 2 (431 MB offloaded), a 19x
        # collapse from ~65 MB of upscaler weights holding the door open.
        try:
            _mm.free_memory(_mm.get_total_memory(_mm.get_torch_device()),
                            _mm.get_torch_device(), [getattr(model, "patcher", None)])
            if hasattr(model, "patcher"):
                model.patcher.model.to(_mm.unet_offload_device())
            _mm.soft_empty_cache()
        except Exception as _e:
            print("[%s] upscale: could not free the upscaler (%s) - the next "
                  "shot may load the DiT partially and run far slower"
                  % (tag, _e), flush=True)
        print("[%s] upscale: model %dx%d -> %dx%d" %
              (tag, w, h, int(imgs.shape[2]), int(imgs.shape[1])), flush=True)
        if scale and abs(scale - 1.0) > 1e-6:
            # the model has a fixed factor; land on the size actually asked for
            th, tw = int(round(h * scale)), int(round(w * scale))
            x = imgs.movedim(-1, 1)
            x = comfy.utils.common_upscale(x, tw, th, "lanczos", "disabled")
            imgs = x.movedim(1, -1)
            print("[%s] upscale: resized to %dx%d" % (tag, tw, th), flush=True)
        return imgs
    th, tw = int(round(h * scale)), int(round(w * scale))
    x = imgs.movedim(-1, 1)
    x = comfy.utils.common_upscale(x, tw, th, "lanczos", "disabled")
    print("[%s] upscale: lanczos %dx%d -> %dx%d" % (tag, w, h, tw, th), flush=True)
    return x.movedim(1, -1)


_UPSCALER_UTILS = None


def _load_upscaler_utils():
    """Load ComfyUI-MiniMaxH3_LatentUpscaler's utils.py by path (cached).

    Two-pass upscale reuses that pack's NestedTensor upscale + CONST re-noise
    math rather than duplicating it. Works whether this file is installed
    loose in custom_nodes or inside ComfyUI-H3-Multishot-Advance/.
    """
    global _UPSCALER_UTILS
    if _UPSCALER_UTILS is not None:
        return _UPSCALER_UTILS
    import importlib.util
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    custom_nodes = os.path.dirname(root)
    candidates = [
        os.path.join(here, "ComfyUI-MiniMaxH3_LatentUpscaler", "utils.py"),
        os.path.join(root,
                     "ComfyUI-MiniMaxH3_LatentUpscaler", "utils.py"),
        os.path.join(custom_nodes,
                     "ComfyUI-MiniMaxH3_LatentUpscaler", "utils.py"),
    ]
    try:
        import folder_paths
        for d in folder_paths.get_folder_paths("custom_nodes"):
            candidates.append(
                os.path.join(d, "ComfyUI-MiniMaxH3_LatentUpscaler", "utils.py"))
    except Exception:
        pass
    for path in candidates:
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location(
                "h3_latent_upscaler_utils", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _UPSCALER_UTILS = mod
            return mod
    _msg = (
        "two_pass_upscale needs the ComfyUI-MiniMaxH3_LatentUpscaler pack "
        "(github.com/Tr1dae/ComfyUI-MiniMaxH3_LatentUpscaler) installed in "
        "custom_nodes - it provides the NestedTensor upscale/re-noise math.")
    raise _h3_fail(_msg, RuntimeError, "H3 two-pass upscaler",
                   tag="H3Upscale")


def _upscale_av_exact(tr, latent_dict, target_h, target_w):
    """Spatially upscale the VIDEO member of an AV latent to EXACT latent dims.

    The pack's own upscaler works from a single scale_by, which can round to a
    grid one cell off the requested resolution; sampling to a fixed widget
    resolution needs exact dims so every shot decodes to width x height.
    """
    import torch.nn.functional as F
    members, was_nested = tr.extract_tensor(latent_dict["samples"])
    v = members[0]
    orig = tuple(v.shape)
    x = v
    if len(orig) > 4:  # [B,C,T,H,W] -> [B*T,C,H,W]
        x = x.reshape(orig[0], orig[1], -1, orig[-2], orig[-1])
        x = x.movedim(2, 1).reshape(-1, orig[1], orig[-2], orig[-1])
    x = F.interpolate(x, size=(target_h, target_w), mode="bilinear",
                      align_corners=False)
    if len(orig) > 4:
        x = x.reshape(orig[0], -1, orig[1], target_h, target_w).movedim(2, 1)
    out = latent_dict.copy()
    out["samples"] = tr.wrap_tensor([x, *members[1:]], was_nested=was_nested)
    return out
