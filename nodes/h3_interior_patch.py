"""Enable interior keyframe anchors in ComfyUI's MiniMax H3 layout, at runtime.

Stock ComfyUI supports keyframes at the FIRST and LAST frame only. Anything in
between raises:

    ValueError: only first/last keyframe anchors are supported

That restriction is a positional-math limitation, not a model limitation. In
`PackedLayout.__init__` the two supported cases are computed as:

    p = 0        ->  cond_t = text_len
    p = fc - 1   ->  cond_t = text_len + sum(_video_t_spans(latent_t)) - FRAME_RESCALE

and because `sum(_video_t_spans(latent_t)) == FRAME_RESCALE * frame_count` on
every valid 17k+5 frame count, BOTH collapse to one general expression:

    cond_t = text_len + FRAME_RESCALE * p

which is defined for every p, not just the endpoints. This module rewrites that
branch in memory so arbitrary anchors become expressible.

Verified empirically on an RTX 5090 (243 frames, anchor at pixel frame 121):
the rendered frame most resembling the anchor image was frame 122 of 243 -- the
requested position, off by one frame -- reached by continuous motion with no cut
(peak frame-to-frame delta 2.3x median). Audio ran unbroken across it.

One caveat worth knowing: if the two images have no continuous path between them
(different rooms, say), H3 satisfies the anchor with a hard CUT rather than a
move. Stock first/last does exactly the same thing with such a pair, so this is
the model being sensible, not a defect of interior anchors.

Nothing here touches disk. The patch is applied to the imported module object,
is idempotent, and self-tests against the stock formula before it commits: if
the recompiled code does not reproduce stock first/last positions exactly, it is
discarded and stock behaviour is left in place.
"""

import inspect
import logging
import textwrap

_OLD = """                if pixel_index == 0:
                    cond_t = float(text_len)
                elif frame_count is not None and pixel_index == frame_count - 1:
                    cond_t = float(text_len) + sum(_video_t_spans(latent_t)) - FRAME_RESCALE
                else:
                    raise ValueError("only first/last keyframe anchors are supported")
"""

_NEW = """                # PATCHED (h3_interior_patch): general form of the two stock
                # cases, so anchors between the endpoints are expressible.
                cond_t = float(text_len) + FRAME_RESCALE * float(pixel_index)
"""

_MARK = "_h3_interior_keyframes_patched"
_MC_LAYOUT_MARK = "_h3_motion_context_layout_patch"
_state = {"done": False, "ok": False, "msg": ""}


def _selftest(module, cls):
    """Stock first/last positions must survive the rewrite byte for byte."""
    frame_rows_probe = []
    for frame_count, latent_t in ((124, 37), (243, 72), (362, 107)):
        text_len = 11
        stock_first = float(text_len)
        stock_last = (float(text_len)
                      + sum(module._video_t_spans(latent_t)) - module.FRAME_RESCALE)
        for pixel_index, expected in ((0, stock_first),
                                      (frame_count - 1, stock_last)):
            kf = [{"resolved_frame_index": pixel_index}]
            lay = cls(text_len, latent_t, 4, 4, 8, keyframes=kf,
                      frame_count=frame_count)
            got = float(lay.position_ids[text_len, 0])
            if abs(got - expected) > 1e-9:
                return False, (f"self-test FAILED at frame_count={frame_count} "
                               f"p={pixel_index}: got {got}, stock {expected}")
            frame_rows_probe.append(got)
    return True, f"self-test passed ({len(frame_rows_probe)} stock positions reproduced)"


def _motion_context_status():
    """Return Motion-Context registration and active layout-patch status.

    KursatAs 2026-08-17 09:18: avoid folder-name false positives; a disabled
    or broken pack directory does not mean the runtime patch is active.

    KursatAs 2026-08-17 09:54: separate node registration from active layout
    ownership. The node can be registered while its runtime patch is not active.

    A custom_nodes folder name is not enough: disabled/broken packs can leave a
    directory behind without registering their node or patching PackedLayout.
    """
    registered = None
    layout_owner = None
    try:
        import nodes
        mappings = getattr(nodes, "NODE_CLASS_MAPPINGS", {}) or {}
        cls = mappings.get("MiniMaxH3MotionContext")
        if cls is not None:
            registered = (getattr(cls, "__module__", None)
                          or "MiniMaxH3MotionContext")
    except Exception:
        pass
    try:
        import comfy.ldm.minimax.model as mm
        layout_cls = getattr(mm, "PackedLayout", None)
        init = getattr(layout_cls, "__init__", None)
        if (getattr(init, _MC_LAYOUT_MARK, False)
                or getattr(layout_cls, _MC_LAYOUT_MARK, False)):
            layout_owner = (getattr(init, "__module__", None)
                            or "Motion-Context runtime patch")
    except Exception:
        pass
    return {"registered": registered, "layout": layout_owner}


def _motion_context_present():
    """Return the active Motion-Context layout owner, if any."""
    return _motion_context_status()["layout"]


def ensure_interior_keyframes(verbose=True):
    """Idempotent. Returns (ok, message)."""
    if _state["done"]:
        return _state["ok"], _state["msg"]
    _state["done"] = True

    _mc_status = _motion_context_status()
    _mc = _mc_status["layout"]
    if _mc:
        _state["ok"] = True
        _state["msg"] = (
            f"standing down: {_mc} owns the active keyframe layout patch "
            f"(its version is a superset). Interior anchors come from that "
            f"runtime; first/last anchors work either way.")
        if verbose:
            logging.info("[H3InteriorPatch] " + _state["msg"])
        return True, _state["msg"]

    try:
        import comfy.ldm.minimax.model as mm
    except Exception as e:                                    # pragma: no cover
        _state["msg"] = f"MiniMax H3 not present in this ComfyUI ({e})"
        return False, _state["msg"]

    cls = getattr(mm, "PackedLayout", None)
    if cls is None:
        _state["msg"] = "comfy.ldm.minimax.model.PackedLayout not found"
        return False, _state["msg"]

    if getattr(cls, _MARK, False):
        _state["ok"] = True
        _state["msg"] = "already patched"
        return True, _state["msg"]

    try:
        # match against the RAW source: the anchor is indented as a method body,
        # and dedenting first would strip the class indent off it and never match
        raw = inspect.getsource(cls.__init__)
    except Exception as e:                                    # pragma: no cover
        _state["msg"] = f"could not read PackedLayout source ({e})"
        return False, _state["msg"]

    if "FRAME_RESCALE * float(pixel_index)" in raw:
        # someone already generalised it (e.g. a hand-edited core file)
        cls._h3_interior_keyframes_patched = True
        _state["ok"] = True
        _state["msg"] = "core already supports interior anchors; nothing to do"
        return True, _state["msg"]

    if _OLD not in raw:
        _state["msg"] = ("anchor text not found in PackedLayout.__init__ - "
                         "ComfyUI changed this code. Interior keyframes are "
                         "DISABLED; first/last still work normally.")
        return False, _state["msg"]

    new_src = textwrap.dedent(raw.replace(_OLD, _NEW, 1))
    ns = {}
    try:
        exec(compile(new_src, "<h3_interior_patch>", "exec"), mm.__dict__, ns)
        patched = ns["__init__"]
    except Exception as e:                                    # pragma: no cover
        _state["msg"] = f"recompile failed ({e}); stock behaviour kept"
        return False, _state["msg"]

    original = cls.__init__
    cls.__init__ = patched
    ok, msg = _selftest(mm, cls)
    if not ok:
        cls.__init__ = original                                # roll back
        _state["msg"] = msg + " - rolled back, stock behaviour kept"
        return False, _state["msg"]

    cls._h3_interior_keyframes_patched = True
    _state["ok"] = True
    _state["msg"] = f"interior keyframe anchors ENABLED ({msg})"
    if _mc_status["registered"]:
        _state["msg"] += (
            f"; Motion-Context node is registered ({_mc_status['registered']}) "
            f"but its layout patch was not active")
    if verbose:
        logging.info("[H3InteriorPatch] " + _state["msg"])
    return True, _state["msg"]
