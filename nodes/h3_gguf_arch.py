"""Teach ComfyUI-GGUF the `minimax_h3` architecture, in memory, at import time.

ComfyUI-GGUF validates a UNET GGUF's architecture against a hardcoded set
(`IMG_ARCH_LIST` in its loader.py). Upstream does not include `minimax_h3`, so
every MiniMax-H3 DiT GGUF is rejected before a single tensor is read:

    ValueError: Unexpected architecture type in GGUF file: 'minimax_h3'

The pack used to ship `apply_gguf_arch_patch.py`, which edited that file on
disk. That worked but required the user to find it, run it, and re-run it after
every ComfyUI-GGUF update - and most people never got that far, which produced
a steady trickle of "your quants are broken" reports that were really "our
install instructions were a footnote".

This module does the same thing without touching the filesystem: it finds the
loaded ComfyUI-GGUF loader module and adds the architecture to the live set.
`gguf_sd_loader` reads that global on every call, so mutating the set in place
is enough, it survives ComfyUI-GGUF updates, and it cannot corrupt a file.

Idempotent and never fatal: if ComfyUI-GGUF is absent or has been restructured,
this logs and returns False rather than breaking the pack's other nodes.
"""

import logging
import sys

ARCH = "minimax_h3"
_done = False


def _find_gguf_loader():
    """The ComfyUI-GGUF loader module, whatever mangled name it loaded under."""
    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        try:
            if (hasattr(mod, "IMG_ARCH_LIST") and hasattr(mod, "TXT_ARCH_LIST")
                    and isinstance(getattr(mod, "IMG_ARCH_LIST"), set)):
                return mod
        except Exception:
            continue
    return None


def ensure_minimax_arch(quiet=False):
    """Add `minimax_h3` to ComfyUI-GGUF's IMG_ARCH_LIST. Safe to call repeatedly."""
    global _done
    if _done:
        return True
    mod = _find_gguf_loader()
    if mod is None:
        if not quiet:
            logging.info(
                "[H3] ComfyUI-GGUF not loaded yet - deferring the minimax_h3 "
                "architecture patch until a GGUF is actually loaded.")
        return False
    try:
        if ARCH not in mod.IMG_ARCH_LIST:
            mod.IMG_ARCH_LIST.add(ARCH)
            logging.info(
                "[H3] taught ComfyUI-GGUF the '%s' architecture (in memory; "
                "no files modified)", ARCH)
        _done = True
        return True
    except Exception as e:                                  # never fatal
        logging.warning(
            "[H3] could not patch ComfyUI-GGUF's architecture list (%s). "
            "MiniMax-H3 DiT GGUFs will fail to load with \"Unexpected "
            "architecture type\"; as a fallback run apply_gguf_arch_patch.py "
            "from this pack's folder.", e)
        return False


ensure_minimax_arch(quiet=True)
