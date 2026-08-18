# -*- coding: utf-8 -*-
"""CLIP/text encoder loading support for H3 Multishot Advance."""

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



def _mmproj_postprocess(gg_loader, vsd, label):
    """Everything gguf_mmproj_loader does AFTER gguf_sd_loader.

    The explicit-file path skipped all of it, so the vision tower loaded under
    raw llama.cpp names (v.blk.*, mm.*) that nothing downstream consumes -
    user-reported as a matmul shape error mid-render, cured by switching back
    to (auto). Same file, different loader.

    Uses upstream's own map and helpers rather than reimplementing the rename,
    so if their key map changes this follows it. If their internals move, fail
    loudly here with an instruction rather than silently returning half a
    vision tower.
    """
    import torch
    try:
        sd_map_replace = gg_loader.sd_map_replace
        CLIP_VISION_SD_MAP = gg_loader.CLIP_VISION_SD_MAP
        dequantize_tensor = gg_loader.dequantize_tensor
        is_quantized = gg_loader.is_quantized
    except AttributeError as e:
        _msg = (
            f"This ComfyUI-GGUF build does not expose the vision key map "
            f"({e}), so an explicitly chosen mmproj cannot be renamed to the "
            f"layout the encoder expects. Set mmproj_name back to (auto) and "
            f"keep the mmproj beside the encoder with a matching name.")
        raise _h3_fail(_msg, RuntimeError, "H3 mmproj incompatible",
                       tag="H3ClipLoader")

    # 1. 4D patch_embd pair -> 5D
    if "v.patch_embd.weight.1" in vsd:
        w1 = dequantize_tensor(vsd.pop("v.patch_embd.weight"),
                               dtype=torch.float32)
        w2 = dequantize_tensor(vsd.pop("v.patch_embd.weight.1"),
                               dtype=torch.float32)
        vsd["v.patch_embd.weight"] = torch.stack([w1, w2], dim=2)

    # 2. the rename that makes the tower addressable at all
    vsd = sd_map_replace(vsd, CLIP_VISION_SD_MAP)

    # 3. split q/k/v -> fused qkv
    if "visual.blocks.0.attn_q.weight" in vsd:
        attns = {}
        for k, v in vsd.items():
            if any(x in k for x in ("attn_q", "attn_k", "attn_v")):
                k_attn, k_name = k.rsplit(".attn_", 1)
                k_attn += ".attn.qkv." + k_name.split(".")[-1]
                attns.setdefault(k_attn, {})[k_name] = dequantize_tensor(
                    v, dtype=(torch.bfloat16 if is_quantized(v)
                              else torch.float16))
        for k, v in attns.items():
            sfx = k.split(".")[-1]
            vsd[k] = torch.cat([v[f"q.{sfx}"], v[f"k.{sfx}"], v[f"v.{sfx}"]],
                               dim=0)

    if not any(k.startswith("visual.") for k in vsd):
        _msg = (
            f"The mmproj '{label}' loaded but produced no visual.* tensors, so "
            f"it is not a vision sidecar for this encoder. Pick the -mmproj "
            f"file that belongs to the encoder you selected, or set "
            f"mmproj_name to (auto).")
        raise _h3_fail(_msg, RuntimeError, "H3 mmproj invalid",
                       tag="H3ClipLoader")
    return vsd


class _H3ClipLoaderSupport:
    """Internal CLIP-loading support used by H3MultiLoader."""

    @classmethod
    def input_types(cls):
        import os
        import folder_paths
        files = set(folder_paths.get_filename_list("text_encoders"))
        for d in folder_paths.get_folder_paths("text_encoders"):
            if not os.path.isdir(d):
                continue
            # RECURSIVE, same reason as the model loader above.
            for root, _dirs, fs in os.walk(d):
                for f in fs:
                    if f.lower().endswith(".gguf") and "mmproj" not in f.lower():
                        files.add(os.path.relpath(os.path.join(root, f), d))
        mm = ["(auto)"]
        for d in folder_paths.get_folder_paths("text_encoders"):
            if not os.path.isdir(d):
                continue
            for root, _dirs, fs in os.walk(d):
                for f in fs:
                    if f.lower().endswith(".gguf") and "mmproj" in f.lower():
                        mm.append(os.path.relpath(os.path.join(root, f), d))
        import nodes as core_nodes
        types = core_nodes.CLIPLoader.INPUT_TYPES()["required"]["type"]
        return {"required": {
            "clip_name": (sorted(files), {
                "tooltip": "safetensors or GGUF - routed automatically. GGUF "
                           "encoders auto-pair their -mmproj vision sidecar."}),
            "type": types,
        }, "optional": {
            # NEW WIDGETS APPEND LAST - inserting above shifts every saved
            # workflow's values by one slot.
            "mmproj_name": (mm, {
                "default": "(auto)",
                "tooltip": "Vision sidecar for a GGUF encoder. '(auto)' uses "
                           "ComfyUI-GGUF's pairing, which matches on FILENAME "
                           "inside the encoder's own folder - rename either "
                           "file, or split them across folders, and the match "
                           "fails. If auto finds nothing and exactly one "
                           "mmproj sits beside the encoder, that one is used "
                           "anyway. Pick a file here to override entirely; "
                           "then names and folders do not matter."}),
        }}

    # llama.cpp/qwen2vl-era names -> the H3 encoder's exact visual.* layout
    # (established 2026-08-04 against the official int8 file). Ordered, and
    # chosen so no rule can re-hit another rule's output.
    _VISION_FIXES = [
        ("visual.merger.ln_q.", "visual.merger.norm."),
        ("attn_qkv.", "attn.qkv."),
        ("mlp.up_proj.", "mlp.linear_fc1."),
        ("mlp.down_proj.", "mlp.linear_fc2."),
        (".fc1.", ".linear_fc1."),
        (".fc2.", ".linear_fc2."),
        ("v.position_embd.weight", "visual.pos_embed.weight"),
    ]

    def load(self, clip_name, type, mmproj_name="(auto)"):
        import os
        import re
        import sys
        import nodes as core_nodes
        if not clip_name.lower().endswith(".gguf"):
            return core_nodes.CLIPLoader().load_clip(clip_name, type=type)

        gg_cls = core_nodes.NODE_CLASS_MAPPINGS.get("CLIPLoaderGGUF")
        if gg_cls is None:
            _msg = "ComfyUI-GGUF not loaded - install/enable it and restart."
            raise _h3_fail(_msg, RuntimeError, "H3 GGUF missing",
                           tag="H3ClipLoader")
        gg = sys.modules[gg_cls.__module__]
        # gguf_mmproj_loader lives in their loader module, which nodes.py
        # does not re-export - resolve it from where gguf_clip_loader is
        # actually defined.
        gg_loader = sys.modules[gg.gguf_clip_loader.__module__]

        import folder_paths
        import comfy.sd
        import comfy.model_management
        clip_path = folder_paths.get_full_path("clip", clip_name)

        # --- text side: their mapper, then truncate to the official H3
        # shape (Qwen3-VL-32B cut to 50 layers; no final norm, no lm_head).
        sd = gg.gguf_clip_loader(clip_path)
        drop = re.compile(r"model\.layers\.(5[0-9]|6[0-9])\.")
        sd = {k: v for k, v in sd.items()
              if not drop.match(k) and k not in ("model.norm.weight",
                                                 "lm_head.weight")}

        # --- vision side: their sidecar loader, then correct the names to
        # H3's layout (their map is qwen2vl-era: wrong merger keys, missing
        # deepstack and qkv rules).
        # Upstream pairs the sidecar by FILENAME inside the encoder's own
        # folder (gguf_mmproj_loader: strip quant suffix, substring match),
        # so a rename on either file - or splitting them across folders -
        # breaks the pair. Upstream then logs an error and returns {}, which
        # renders as "the model ignores my reference image". Three ways out,
        # in order, and a hard failure rather than a silent one.
        vsd = None
        if mmproj_name and mmproj_name != "(auto)":
            mm_path = folder_paths.get_full_path("text_encoders", mmproj_name)
            if not mm_path:
                _msg = (
                    f"mmproj_name '{mmproj_name}' is not in the "
                    f"text_encoders folder any more.")
                raise _h3_fail(_msg, RuntimeError, "H3 mmproj missing",
                               tag="H3ClipLoader")
            vsd, _ = gg_loader.gguf_sd_loader(mm_path, is_text_model=True)
            vsd = _mmproj_postprocess(gg_loader, vsd, mmproj_name)
            print(f"[H3ClipLoader] vision sidecar (explicit): {mmproj_name}",
                  flush=True)
        else:
            vsd = gg_loader.gguf_mmproj_loader(clip_path)
            if not vsd:
                # upstream's name match failed; if exactly one mmproj sits
                # beside the encoder the intent is unambiguous, so use it
                _dir = os.path.dirname(clip_path)
                cands = [f for f in os.listdir(_dir)
                         if f.lower().endswith(".gguf")
                         and "mmproj" in f.lower()]
                if len(cands) == 1:
                    vsd, _ = gg_loader.gguf_sd_loader(
                        os.path.join(_dir, cands[0]), is_text_model=True)
                    print(f"[H3ClipLoader] filename pairing failed, but "
                          f"'{cands[0]}' is the only mmproj beside the "
                          f"encoder - using it. Set mmproj_name to silence "
                          f"this.", flush=True)
        if not vsd:
            _msg = (
                f"No vision sidecar (-mmproj) resolved for '{clip_name}'. "
                f"The H3 encoder NEEDS its vision tower for image "
                f"references and shot chaining. Either keep the mmproj file "
                f"beside the encoder with a matching name, or just pick it "
                f"explicitly in this node's mmproj_name widget - with that "
                f"set, names and folders do not matter.")
            raise _h3_fail(_msg, RuntimeError, "H3 mmproj missing",
                           tag="H3ClipLoader")
        # merger mlp indices -> linear_fc1/2 by ascending index
        idxs = sorted({m.group(1) for k in vsd
                       for m in [re.match(r"visual\.merger\.mlp\.(\d+)\.", k)]
                       if m})
        # deepstack mergers: llama.cpp indexes them by the vision layer they
        # tap (8/16/24), the H3 encoder by list position (0/1/2) - remap
        # ascending, and sort NUMERICALLY (lexically 16 < 8).
        ds = sorted({int(m.group(1)) for k in vsd
                     for m in [re.match(r"v\.deepstack\.(\d+)\.", k)]
                     if m})
        fixed = {}
        for k, v in vsd.items():
            for i, name in zip(idxs, ("linear_fc1", "linear_fc2")):
                k = k.replace(f"visual.merger.mlp.{i}.",
                              f"visual.merger.{name}.")
            for pos, layer in enumerate(ds):
                k = k.replace(f"v.deepstack.{layer}.",
                              f"visual.deepstack_merger_list.{pos}.")
            for a, b in self._VISION_FIXES:
                k = k.replace(a, b)
            fixed[k] = v
        sd.update(fixed)

        clip = comfy.sd.load_text_encoder_state_dicts(
            clip_type=getattr(comfy.sd.CLIPType, type.upper(),
                              comfy.sd.CLIPType.STABLE_DIFFUSION),
            state_dicts=[sd],
            model_options={
                "custom_operations": gg.GGMLOps,
                "initial_device":
                    comfy.model_management.text_encoder_offload_device(),
            },
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        clip.patcher = gg.GGUFModelPatcher.clone(clip.patcher)
        return (clip,)
