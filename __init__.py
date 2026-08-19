"""MiniMax-H3 Multishot Advance node pack.

Every optional module is imported defensively. One module failing (a
missing third-party dependency, a ComfyUI API change) must not take the
whole pack down with it - otherwise every node vanishes from the menu at
once and saved workflows open as a wall of red boxes.
"""
import logging

from .nodes import h3_multishot_sampler as _core

# KursatAs 2026-08-19 21:05: keep runtime version aligned with the Project Node release.
__version__ = "1.1.0"
VERSION = __version__
WEB_DIRECTORY = "web/js"
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS",
           "WEB_DIRECTORY", "__version__", "VERSION"]


def _advance_key(key):
    key = str(key)
    # KursatAs 2026-08-19 19:10: allow new Advance-native nodes to keep their
    # exact public node id instead of receiving a second Advance suffix.
    return key if (key.endswith("Advance")
                   or key.startswith("H3Advance")) else f"{key}Advance"


def _advance_display(key, name):
    name = str(name or key)
    if name.startswith("[deprecated]"):
        return None
    # KursatAs 2026-08-19 19:10: H3 Advance Project is the first node whose
    # source module already owns the exact Advance display name.
    if name.startswith("H3 Advance "):
        return name
    # KursatAs 2026-08-18 12:49: allow nodes to opt into an exact Advance
    # display name instead of the package-wide "H3 Advance ..." prefix.
    if name.endswith(" Advance"):
        return name
    if name.startswith("H3 "):
        return "H3 Advance " + name[3:]
    return "H3 Advance " + name


def _advance_class(name, cls):
    return type(name, (cls,), {
        "__module__": cls.__module__,
        "__doc__": cls.__doc__,
    })


def _register(mapping, display_mapping):
    for key, cls in (mapping or {}).items():
        adv_key = _advance_key(key)
        if adv_key is None:
            continue
        display = _advance_display(key, (display_mapping or {}).get(key))
        if display is None:
            continue
        NODE_CLASS_MAPPINGS[adv_key] = _advance_class(adv_key, cls)
        NODE_DISPLAY_NAME_MAPPINGS[adv_key] = display


def _merge(modname):
    try:
        mod = __import__(f"{__name__}.nodes.{modname}", fromlist=["*"])
    except Exception as e:                                # pragma: no cover
        logging.warning("[H3-Multishot-Advance] %s not loaded (%s) - its nodes are "
                        "unavailable; the rest of the pack still works",
                        modname, e)
        return
    _register(getattr(mod, "NODE_CLASS_MAPPINGS", {}),
              getattr(mod, "NODE_DISPLAY_NAME_MAPPINGS", {}))


_register(_core.NODE_CLASS_MAPPINGS, _core.NODE_DISPLAY_NAME_MAPPINGS)


for _m in ("h3_multiloader",     # complete MiniMax-H3 stack loader
           "h3_project",         # project manifest and future cache workspace
           "h3_cartridge",       # portable character cartridges
           "h3_episode_tools",   # H3Controls
           "h3_speed_boosters"): # switch panel for optional accelerators
    _merge(_m)

# Teaches ComfyUI-GGUF the minimax_h3 architecture, in memory, at startup.
# apply_gguf_arch_patch.py is the on-disk fallback for installs where this
# import cannot reach ComfyUI-GGUF. Harmless when GGUF is not installed.
try:
    from .nodes import h3_gguf_arch          # noqa: F401
except Exception as _e:                                   # pragma: no cover
    logging.info("[H3-Multishot-Advance] GGUF arch hook not installed (%s). "
                 "Safetensors checkpoints are unaffected; for GGUF, install "
                 "ComfyUI-GGUF and run apply_gguf_arch_patch.py once.", _e)
