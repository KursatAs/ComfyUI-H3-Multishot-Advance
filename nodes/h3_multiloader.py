# -*- coding: utf-8 -*-
"""Complete MiniMax-H3 stack loader node for H3 Multishot Advance."""

try:
    from .h3_notify import h3_error as _h3_error
except Exception:
    try:
        from h3_notify import h3_error as _h3_error
    except Exception:
        def _h3_error(*_args, **_kwargs):
            return False

try:
    from .h3_model_runtime import _H3ModelLoaderSupport
except Exception:
    from h3_model_runtime import _H3ModelLoaderSupport

try:
    from .h3_clip_runtime import _H3ClipLoaderSupport
except Exception:
    from h3_clip_runtime import _H3ClipLoaderSupport


def _h3_fail(message, exc_type=RuntimeError, title="H3 Multishot", tag=None):
    msg = str(message)
    _h3_error(msg, title=title, tag=tag)
    return exc_type(msg)


class H3MultiLoader:
    """Load the full MiniMax-H3 model stack from one node."""

    @staticmethod
    def _vae_names():
        import folder_paths
        try:
            names = list(folder_paths.get_filename_list("vae"))
        except Exception:
            names = []
        return sorted(set(names)) or ["(no VAE files found)"]

    @staticmethod
    def _pick_vae(names, kind):
        lk = str(kind).lower()
        for name in names:
            low = str(name).lower()
            if ("minimax" in low and "h3" in low and lk in low
                    and "vae" in low):
                return name
        for name in names:
            low = str(name).lower()
            if lk in low and "vae" in low:
                return name
        for name in names:
            if lk in str(name).lower():
                return name
        return names[0]

    @staticmethod
    def _clip_type_default(spec):
        if not isinstance(spec, tuple) or not spec:
            return spec
        values = spec[0]
        options = dict(spec[1]) if len(spec) > 1 and isinstance(spec[1], dict) else {}
        try:
            if "minimax" in values:
                options["default"] = "minimax"
        except TypeError:
            pass
        return (values, options, *spec[2:])

    @classmethod
    def INPUT_TYPES(cls):
        model_inputs = _H3ModelLoaderSupport.input_types()
        clip_inputs = _H3ClipLoaderSupport.input_types()
        vaes = cls._vae_names()
        # KursatAs 2026-08-18 09:11: One H3 stack loader keeps the canvas
        # clean while preserving the exact proven model/CLIP/VAE loaders.
        return {"required": {
            "model_name": model_inputs["required"]["model_name"],
            "clip_name": clip_inputs["required"]["clip_name"],
            "clip_type": cls._clip_type_default(
                clip_inputs["required"]["type"]),
            "video_vae_name": (vaes, {
                "default": cls._pick_vae(vaes, "video"),
                "tooltip": "MiniMax-H3 video VAE. Defaults to the first "
                           "minimax/h3/video VAE match when available."}),
            "audio_vae_name": (vaes, {
                "default": cls._pick_vae(vaes, "audio"),
                "tooltip": "MiniMax-H3 audio VAE. Defaults to the first "
                           "minimax/h3/audio VAE match when available."}),
        }, "optional": {
            "activation_reserve_gb":
                model_inputs["optional"]["activation_reserve_gb"],
            "mmproj_name": clip_inputs["optional"]["mmproj_name"],
        }}

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "VAE")
    RETURN_NAMES = ("model", "clip", "video_vae", "audio_vae")
    FUNCTION = "load"
    CATEGORY = "loaders/minimax"
    DESCRIPTION = ("Load the complete MiniMax-H3 stack: diffusion model, "
                   "text encoder, video VAE, and audio VAE.")

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def load(self, model_name, clip_name, clip_type, video_vae_name,
             audio_vae_name, activation_reserve_gb=0.0,
             mmproj_name="(auto)"):
        import nodes as core_nodes
        if str(video_vae_name).startswith("(no VAE"):
            raise _h3_fail("No VAE files found. Install the MiniMax-H3 video "
                           "and audio VAE files, then restart ComfyUI.",
                           RuntimeError, "H3 VAE missing",
                           tag="H3MultiLoader")
        if str(audio_vae_name).startswith("(no VAE"):
            raise _h3_fail("No VAE files found. Install the MiniMax-H3 video "
                           "and audio VAE files, then restart ComfyUI.",
                           RuntimeError, "H3 VAE missing",
                           tag="H3MultiLoader")

        model = _H3ModelLoaderSupport().load(
            model_name, activation_reserve_gb)[0]
        clip = _H3ClipLoaderSupport().load(
            clip_name, clip_type, mmproj_name)[0]
        vae_loader = core_nodes.VAELoader()
        try:
            video_vae = vae_loader.load_vae(video_vae_name)[0]
        except Exception as e:
            raise _h3_fail("Video VAE could not be loaded: %s" % e,
                           RuntimeError, "H3 video VAE missing",
                           tag="H3MultiLoader")
        try:
            audio_vae = vae_loader.load_vae(audio_vae_name)[0]
        except Exception as e:
            raise _h3_fail("Audio VAE could not be loaded: %s" % e,
                           RuntimeError, "H3 audio VAE missing",
                           tag="H3MultiLoader")
        print("[H3MultiLoader] loaded model=%r, clip=%r, video_vae=%r, "
              "audio_vae=%r" % (model_name, clip_name, video_vae_name,
                                audio_vae_name), flush=True)
        return (model, clip, video_vae, audio_vae)


NODE_CLASS_MAPPINGS = {"H3MultiLoader": H3MultiLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"H3MultiLoader": "H3 Multi Model Loader"}
