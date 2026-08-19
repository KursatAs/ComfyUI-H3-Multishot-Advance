# -*- coding: utf-8 -*-
"""Shot-cache support for H3 Multishot Advance."""

import json
import os

try:
    from .h3_notify import h3_info as _h3_info
except Exception:
    try:
        from h3_notify import h3_info as _h3_info
    except Exception:
        def _h3_info(*_args, **_kwargs):
            return False


_H3_SHOT_CACHE_VERSION = 1
_H3_CACHE_KIND = "__h3_cache_kind__"


def _h3_cache_root():
    import os
    try:
        import folder_paths
        base = folder_paths.get_user_directory()
    except Exception:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # KursatAs 2026-08-19 20:37: use a package-specific cache root so
    # Multishot Advance cache files do not mix with legacy H3 cache folders.
    return os.path.abspath(os.path.join(base,
                                        "multishot_advance_shot_cache"))


def _h3_cache_tensor_bytes(t):
    t = t.detach().cpu().contiguous()
    try:
        return memoryview(t.numpy()).tobytes()
    except TypeError:
        return memoryview(t.float().numpy()).tobytes()


def _h3_cache_fingerprint(obj):
    import hashlib
    h = hashlib.sha256()

    def feed(x):
        try:
            import torch
            if torch.is_tensor(x):
                tx = x.detach().cpu().contiguous()
                h.update(b"T")
                h.update(str(tuple(tx.shape)).encode("utf-8"))
                h.update(str(tx.dtype).encode("utf-8"))
                h.update(_h3_cache_tensor_bytes(tx))
                return
        except Exception:
            pass
        if getattr(x, "is_nested", False):
            h.update(b"N")
            for y in x.unbind():
                feed(y)
        elif isinstance(x, dict):
            h.update(b"D")
            for k in sorted(x, key=lambda v: str(v)):
                h.update(str(k).encode("utf-8", "replace"))
                feed(x[k])
        elif isinstance(x, (list, tuple)):
            h.update(b"L" if isinstance(x, list) else b"U")
            h.update(str(len(x)).encode("ascii"))
            for y in x:
                feed(y)
        else:
            h.update(b"R")
            h.update(repr(x).encode("utf-8", "replace"))

    feed(obj)
    return h.hexdigest()


def _h3_cache_pack(obj):
    try:
        import torch
        if torch.is_tensor(obj):
            return obj.detach().cpu().contiguous()
    except Exception:
        pass
    if getattr(obj, "is_nested", False):
        return {_H3_CACHE_KIND: "nested",
                "items": [_h3_cache_pack(x) for x in obj.unbind()]}
    if isinstance(obj, dict):
        return {k: _h3_cache_pack(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return {_H3_CACHE_KIND: "tuple",
                "items": [_h3_cache_pack(x) for x in obj]}
    if isinstance(obj, list):
        return [_h3_cache_pack(x) for x in obj]
    return obj


def _h3_cache_unpack(obj):
    if isinstance(obj, dict) and obj.get(_H3_CACHE_KIND) == "nested":
        import comfy.nested_tensor as _nt
        return _nt.NestedTensor([_h3_cache_unpack(x) for x in obj["items"]])
    if isinstance(obj, dict) and obj.get(_H3_CACHE_KIND) == "tuple":
        return tuple(_h3_cache_unpack(x) for x in obj["items"])
    if isinstance(obj, dict):
        return {k: _h3_cache_unpack(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_h3_cache_unpack(x) for x in obj]
    return obj


def _h3_cache_key(obj):
    import hashlib
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      default=str).encode("utf-8", "replace")
    return hashlib.sha256(blob).hexdigest()


def _h3_safe_rmtree(path, root):
    import os
    import shutil
    root = os.path.abspath(root)
    path = os.path.abspath(path)
    try:
        inside = os.path.commonpath([root, path]) == root
    except ValueError:
        inside = False
    if inside and path != root and os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def _h3_shot_cache_mode(value):
    mode = str(value or "off").strip().lower()
    if mode == "read_write":
        return "use_cache"
    if mode == "clear":
        return "rebuild_cache"
    return mode


class _H3ShotCacheSession:
    """CPU/disk prefix cache for H3 multi-shot sampler state.

    This class owns cache mode/path/restore/save mechanics. The sampler still
    owns the render state itself, because those locals are live algorithm
    state and should not be hidden behind cache plumbing.
    """

    def __init__(self, mode, shots, total_shots, cache_dir=None,
                 restore_prefix_limit=None, cache_label="shot_cache",
                 write_allowed=True):
        self.mode = _h3_shot_cache_mode(mode)
        if not write_allowed and self.mode == "rebuild_cache":
            # KursatAs 2026-08-19 19:18: read-only project mode may inspect
            # existing prefixes but must never clear or rewrite project cache.
            self.mode = "use_cache"
        self.shots = list(shots)
        self.total_shots = int(total_shots)
        self.enabled = self.mode in ("use_cache", "rebuild_cache")
        self.write_enabled = False
        self.base_key = None
        self.override_cache_dir = (
            os.path.abspath(cache_dir) if cache_dir else None)
        self.restore_prefix_limit = (
            None if restore_prefix_limit is None
            else max(0, min(self.total_shots, int(restore_prefix_limit))))
        self.cache_label = str(cache_label or "shot_cache")
        self.write_allowed = bool(write_allowed)
        self.cache_dir = None

    def disable_for_streaming(self):
        self.enabled = False
        self.write_enabled = False
        print("[H3Memory] shot_cache disabled for this run: "
              "low_ram_master is using the streaming path.",
              flush=True)

    def disable_due_to_error(self, exc):
        self.enabled = False
        self.write_enabled = False
        print("[H3Memory] shot_cache disabled: %s" % exc, flush=True)

    def prefix_key(self, idx):
        return _h3_cache_key({"base": self.base_key,
                              "prompts": self.shots[:idx]})

    def path(self, idx):
        import os
        key = self.prefix_key(idx)
        return os.path.join(self.cache_dir, "prefix_%02d_%s.pt"
                            % (idx, key[:16]))

    def configure(self, base_obj):
        import os
        import torch as _torch
        _torch  # preserve the old early torch dependency check for cache runs
        self.base_key = _h3_cache_key(base_obj)
        root = _h3_cache_root()
        if self.override_cache_dir:
            # KursatAs 2026-08-19 19:14: project cache writes directly under
            # user/multishot_advance_projects/<project>/cache so the project
            # folder owns the
            # editable render state; completed videos are not archived here.
            self.cache_dir = self.override_cache_dir
            delete_root = os.path.dirname(self.cache_dir)
        else:
            self.cache_dir = os.path.join(root, self.base_key[:16])
            delete_root = root
        if self.mode == "rebuild_cache":
            _h3_safe_rmtree(self.cache_dir, delete_root)
            print("[H3Memory] shot_cache: cleared %s"
                  % self.cache_dir, flush=True)
            _h3_info(
                "Shot cache: rebuilding cache for this configuration; "
                "existing compatible prefixes are ignored for this run.",
                topic="shot_cache", tag="H3Memory", timeout_ms=4000)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.write_enabled = self.write_allowed
        if self.override_cache_dir:
            limit = (self.total_shots if self.restore_prefix_limit is None
                     else self.restore_prefix_limit)
            print("[H3Memory] project cache: %s | safe prefix %d/%d"
                  % (self.cache_dir, limit, self.total_shots), flush=True)

    def restore_prefix(self):
        if self.mode == "rebuild_cache":
            return 0, None

        import os
        import torch as _torch
        limit = (self.total_shots if self.restore_prefix_limit is None
                 else self.restore_prefix_limit)
        for idx in range(limit, 0, -1):
            path = self.path(idx)
            if not os.path.isfile(path):
                continue
            try:
                try:
                    blob = _torch.load(path, map_location="cpu",
                                       weights_only=False)
                except TypeError:
                    blob = _torch.load(path, map_location="cpu")
                if (blob.get("version") != _H3_SHOT_CACHE_VERSION
                        or blob.get("base_key") != self.base_key
                        or blob.get("prefix_key") != self.prefix_key(idx)):
                    continue
                state = _h3_cache_unpack(blob["state"])
                resume_msg = (
                    "no sampling needed; assembling cached clips."
                    if idx >= self.total_shots else
                    "rendering starts at clip %d." % (idx + 1))
                print("[H3Memory] shot_cache HIT: restored "
                      "%d/%d completed shot(s); %s"
                      % (idx, self.total_shots, resume_msg), flush=True)
                _h3_info(
                    "Shot cache restored %d/%d clip(s); %s"
                    % (idx, self.total_shots, resume_msg),
                    topic="memory_sampler", tag="H3Memory",
                    timeout_ms=4000)
                return int(idx), state
            except Exception as exc:
                print("[H3Memory] shot_cache: ignored unreadable prefix "
                      "%d (%s)" % (idx, exc), flush=True)

        print("[H3Memory] shot_cache: no compatible prefix found; "
              "rendering from clip 1.", flush=True)
        _h3_info(
            "Shot cache: no compatible prefix found; rendering from clip 1.",
            topic="shot_cache", tag="H3Memory", timeout_ms=4000)
        return 0, None

    def save_prefix(self, idx, state):
        if not self.write_enabled:
            return None

        import os
        import torch as _torch
        path = self.path(idx)
        tmp = path + ".tmp"
        blob = {
            "version": _H3_SHOT_CACHE_VERSION,
            "base_key": self.base_key,
            "prefix_key": self.prefix_key(idx),
            "shot_index": int(idx),
            "state": _h3_cache_pack(state),
        }
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            _torch.save(blob, tmp)
            os.replace(tmp, path)
            print("[H3Memory] shot_cache: stored prefix %d/%d -> %s"
                  % (idx, self.total_shots, os.path.basename(path)),
                  flush=True)
            return path
        except Exception as exc:
            self.write_enabled = False
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass
            print("[H3Memory] shot_cache disabled: could not write "
                  "prefix %d (%s)" % (idx, exc), flush=True)
            return None


def _h3_restore_shot_cache_state(state, bank):
    bank._entries = list(state["bank_entries"])
    return (
        list(state["frames_parts"]),
        list(state["audio_parts"]),
        list(state["lat_v_parts"]),
        list(state["lat_a_parts"]),
        state["sr"],
        state["voice_block"],
        state["cg_ref"],
        state["last_tail"],
        state["ho_v"],
        state["ho_a"],
        state["ho_taper_src"],
        state["ho_guard"],
        state["ho_wav_tail"],
        state["house_frame"],
        state["cc_mu"],
        state["cc_cov"],
        state["cp_prev"],
        state["pin_sig0"],
        state["pin_hf0"],
        state["cg_last_raw"],
        state["cp_trim"],
    )
