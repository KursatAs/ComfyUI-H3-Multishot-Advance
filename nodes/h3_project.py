# -*- coding: utf-8 -*-
"""Project manifest node for H3 Multishot Advance."""

import datetime as _datetime
import hashlib as _hashlib
import json as _json
import os as _os
import re as _re


_H3_ADVANCE_PROJECT_VERSION = 1
_H3_ADVANCE_PROJECT_TYPE = "H3_ADVANCE_PROJECT"


def _h3_project_root():
    try:
        import folder_paths
        base = folder_paths.get_user_directory()
    except Exception:
        base = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    # KursatAs 2026-08-19 20:37: keep Advance project manifests under a
    # package-specific user folder instead of the generic legacy H3 name.
    return _os.path.abspath(_os.path.join(base,
                                          "multishot_advance_projects"))


def _h3_project_slug(project_name):
    name = str(project_name or "").strip()
    if not name:
        name = "Untitled_Project"
    name = _re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", name)
    name = _re.sub(r"\s+", "_", name).strip(" ._")
    if not name:
        name = "Untitled_Project"
    return name[:96]


def _h3_project_sha256_text(text):
    return _hashlib.sha256(
        str(text or "").encode("utf-8", "replace")).hexdigest()


def _h3_project_parse_script(script):
    try:
        from .h3_multishot_sampler import _parse_script
    except Exception:
        try:
            from h3_multishot_sampler import _parse_script
        except Exception:
            _parse_script = None

    if _parse_script is not None:
        return [str(x) for x in _parse_script(script)]

    text = str(script or "").strip()
    if not text:
        return [""]
    if text.startswith("{") or text.startswith("["):
        data = _json.loads(text)
        if isinstance(data, dict):
            return [str(x) for x in data.get("prompts", [])] or [text]
        if isinstance(data, list):
            return [str(x) for x in data] or [text]
    return [b.strip().replace('\\"', '"')
            for b in _re.split(r"(?m)^---\s*$", text) if b.strip()] or [text]


def _h3_project_prepare_shots(script, shot_count=0):
    shots = _h3_project_parse_script(script)
    try:
        n = int(shot_count)
    except Exception:
        n = 0
    if n <= 0:
        return shots
    if len(shots) > n:
        return shots[:n]
    # KursatAs 2026-08-19 19:31: mirror sampler shot_count semantics exactly:
    # a short script repeats the last prompt, so the project manifest must
    # track the same logical shot list that will be cached by the sampler.
    while len(shots) < n:
        shots.append(shots[-1] if shots else "")
    return shots


def _h3_project_int(value, fallback=0):
    try:
        return int(value)
    except Exception:
        return int(fallback)


def _h3_project_manifest_path(project_dir):
    return _os.path.join(project_dir, "project.json")


def _h3_project_cache_dir(project_dir):
    return _os.path.join(project_dir, "cache")


def _h3_project_read_manifest(path):
    if not _os.path.isfile(path):
        return None, None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return _json.load(fh), None
    except Exception as exc:
        return None, str(exc)


def _h3_project_write_manifest(path, manifest):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        _json.dump(manifest, fh, ensure_ascii=False, indent=2,
                   sort_keys=True)
        fh.write("\n")
    _os.replace(tmp, path)


def _h3_project_first_changed_shot(previous_hashes, current_hashes):
    prev = list(previous_hashes or [])
    curr = list(current_hashes or [])
    for idx in range(min(len(prev), len(curr))):
        if prev[idx] != curr[idx]:
            return idx + 1
    if len(prev) != len(curr):
        return min(len(prev), len(curr)) + 1
    return None


def _h3_project_shots_from_manifest(manifest):
    shots = []
    for item in (manifest or {}).get("shots", []):
        if isinstance(item, dict):
            shots.append(str(item.get("prompt", "")))
    return shots


def _h3_project_script_from_shots(shots):
    return "\n\n---\n\n".join(str(shot or "") for shot in shots)


def _h3_project_apply_shot_override(shots, shot_index, override_prompt,
                                    enabled):
    if not enabled:
        return list(shots), False, ""
    idx = _h3_project_int(shot_index, 0)
    if idx <= 0:
        return list(shots), False, "override enabled but shot index is 0"
    if idx > len(shots):
        return (
            list(shots),
            False,
            "override shot %d skipped; project has %d shot(s)"
            % (idx, len(shots)),
        )
    out = list(shots)
    out[idx - 1] = str(override_prompt or "")
    return out, True, "override applied to shot %d" % idx


def _h3_project_relpath(path, root):
    try:
        rel = _os.path.relpath(path, root)
    except Exception:
        return str(path)
    if rel.startswith(".."):
        return str(path)
    return rel.replace("\\", "/")


def _h3_project_prefix_index(item):
    try:
        return int(item.get("shot_index", 0))
    except Exception:
        return 0


def _h3_project_prefix_disk_path(item, project_dir):
    path = str(item.get("path") or "")
    if not path:
        filename = str(item.get("filename") or "")
        if filename:
            path = _os.path.join("cache", filename)
    if not path:
        return None
    if _os.path.isabs(path):
        return _os.path.abspath(path)
    return _os.path.abspath(_os.path.join(project_dir, path.replace(
        "/", _os.sep)))


def _h3_project_preserved_prefixes(previous, safe_prefix_shots,
                                  project_dir=None):
    cache = (previous or {}).get("cache") or {}
    prefixes = cache.get("prefixes") or []
    kept = []
    missing = 0
    for item in prefixes:
        if not isinstance(item, dict):
            continue
        idx = _h3_project_prefix_index(item)
        if 0 < idx <= int(safe_prefix_shots):
            if project_dir:
                # KursatAs 2026-08-19 19:25: project metadata must not claim
                # a prefix is available after the user manually deletes cache
                # .pt files. The sampler still validates file contents later.
                path = _h3_project_prefix_disk_path(item, project_dir)
                if path and not _os.path.isfile(path):
                    missing += 1
                    continue
            kept.append(dict(item))
    kept.sort(key=_h3_project_prefix_index)
    return kept, missing


def _h3_project_last_prefix(prefixes):
    return max([_h3_project_prefix_index(item) for item in prefixes] or [0])


def _h3_project_cache_summary(prefixes, shot_count, first_changed_shot,
                              dirty_reason=None):
    shot_count = int(shot_count or 0)
    last_prefix = min(_h3_project_last_prefix(prefixes), shot_count)
    if first_changed_shot is not None:
        render_state = "dirty"
        needs_render_from = int(first_changed_shot)
    elif shot_count <= 0:
        render_state = "empty"
        needs_render_from = None
    elif last_prefix >= shot_count:
        render_state = "complete"
        needs_render_from = None
    elif last_prefix > 0:
        render_state = "partial"
        needs_render_from = last_prefix + 1
    else:
        render_state = "empty"
        needs_render_from = 1
    return {
        "render_state": render_state,
        "cached_prefix_shots": int(last_prefix),
        "total_shots": int(shot_count),
        "needs_render_from": needs_render_from,
        "dirty_reason": (
            str(dirty_reason or "") if render_state == "dirty" else ""),
    }


def _h3_project_make_manifest(project_name, project_slug, source_script_text,
                              script_text,
                              shots, shot_hashes, script_hash,
                              requested_shot_count, safe_prefix_shots,
                              first_changed_shot, mode, dirty_reason="",
                              prefixes=None, previous=None):
    now = _datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    prefixes = list(prefixes or [])
    previous = previous if isinstance(previous, dict) else {}
    summary = _h3_project_cache_summary(
        prefixes, len(shots), first_changed_shot, dirty_reason)
    manifest = {
        "type": _H3_ADVANCE_PROJECT_TYPE,
        "version": _H3_ADVANCE_PROJECT_VERSION,
        "project_name": str(project_name or ""),
        "project_slug": str(project_slug or ""),
        # KursatAs 2026-08-19 19:26: rebuilding the editable manifest must
        # not reset project identity/history every time the node executes.
        "created_at": str(previous.get("created_at") or now),
        "updated_at": now,
        "mode": str(mode or "load_or_create"),
        # KursatAs 2026-08-19 19:10: project files store editable render
        # state only; completed mp4 outputs intentionally stay outside.
        "outputs_are_not_archived": True,
        # KursatAs 2026-08-19 19:20: project loading must not require the
        # workflow string input to still contain the old script tomorrow.
        "script_text": str(script_text or ""),
        # KursatAs 2026-08-19 19:37: keep the user's original text separately;
        # script_text is the expanded effective script that should drive the
        # sampler and exactly match shot_count_out.
        "source_script_text": str(source_script_text or ""),
        "script_hash": script_hash,
        # KursatAs 2026-08-19 19:34: store the requested sampler shot_count
        # separately from the effective shot list so a named project can be
        # reopened with an empty script input and still restore repeated shots.
        "requested_shot_count": int(requested_shot_count),
        "effective_shot_count": len(shots),
        "shot_count": len(shots),
        "shots": [
            {"index": idx + 1, "hash": shot_hashes[idx], "prompt": shots[idx]}
            for idx in range(len(shots))
        ],
        "cache": {
            "kind": "prefix_state",
            "directory": "cache",
            "safe_prefix_shots": int(safe_prefix_shots),
            "first_changed_shot": first_changed_shot,
            # KursatAs 2026-08-19 19:18: prefix metadata is project state;
            # it tracks sampler-state cache files, not exported videos.
            # KursatAs 2026-08-19 19:30: keep a compact editor-style render
            # state so reopening a project can immediately say complete,
            # partial, empty, or dirty without inspecting every .pt blob.
            "render_state": summary["render_state"],
            # KursatAs 2026-08-19 19:35: dirty state needs a reason; editor
            # decisions later should know whether the cause was a prompt edit,
            # explicit rebuild, new project, shot_count, or sampler config.
            "dirty_reason": summary["dirty_reason"],
            "cached_prefix_shots": summary["cached_prefix_shots"],
            "total_shots": summary["total_shots"],
            "needs_render_from": summary["needs_render_from"],
            "last_completed_prefix": summary["cached_prefix_shots"],
            "prefixes": prefixes,
        },
    }
    sampler = previous.get("sampler")
    if isinstance(sampler, dict):
        # KursatAs 2026-08-19 19:26: Project node runs before the sampler in
        # the graph; preserving the last sampler block avoids erasing useful
        # render-config metadata simply because the project was reopened.
        manifest["sampler"] = dict(sampler)
    return manifest


def h3_advance_project_active(project):
    return (
        isinstance(project, dict)
        and project.get("type") == _H3_ADVANCE_PROJECT_TYPE
        and project.get("project_dir")
    )


def h3_advance_project_record_prefix(project, shot_index, total_shots,
                                     base_key, prefix_key, prefix_path):
    if not h3_advance_project_active(project):
        return False
    if str(project.get("mode") or "") == "read_only":
        return False

    manifest_path = project.get("manifest_path")
    project_dir = project.get("project_dir")
    if not manifest_path or not project_dir or not prefix_path:
        return False

    manifest, _read_error = _h3_project_read_manifest(manifest_path)
    if manifest is None:
        manifest = dict(project.get("manifest") or {})
    if not manifest:
        return False

    idx = int(shot_index)
    now = _datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    cache = manifest.setdefault("cache", {})
    prefixes = [
        dict(item)
        for item in cache.get("prefixes", [])
        if isinstance(item, dict) and _h3_project_prefix_index(item) < idx
    ]
    prefixes.append({
        "shot_index": idx,
        "total_shots": int(total_shots),
        "filename": _os.path.basename(prefix_path),
        "path": _h3_project_relpath(prefix_path, project_dir),
        "base_key": str(base_key or ""),
        "prefix_key": str(prefix_key or ""),
        "updated_at": now,
    })
    prefixes.sort(key=_h3_project_prefix_index)

    cache["kind"] = "prefix_state"
    cache["directory"] = "cache"
    cache["prefixes"] = prefixes
    summary = _h3_project_cache_summary(prefixes, int(total_shots), None)
    cache["render_state"] = summary["render_state"]
    cache["dirty_reason"] = summary["dirty_reason"]
    cache["cached_prefix_shots"] = summary["cached_prefix_shots"]
    cache["total_shots"] = summary["total_shots"]
    cache["needs_render_from"] = summary["needs_render_from"]
    cache["last_completed_prefix"] = summary["cached_prefix_shots"]
    cache["last_prefix_updated_at"] = now
    manifest["updated_at"] = now

    _h3_project_write_manifest(manifest_path, manifest)
    project["manifest"] = manifest
    project["render_state"] = cache["render_state"]
    project["cached_prefix_shots"] = cache["cached_prefix_shots"]
    project["needs_render_from"] = cache["needs_render_from"]
    return True


def _h3_project_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _h3_project_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_h3_project_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def h3_advance_project_record_sampler_config(project, base_key, cache_base):
    if not h3_advance_project_active(project):
        return False
    if str(project.get("mode") or "") == "read_only":
        return False

    manifest_path = project.get("manifest_path")
    if not manifest_path:
        return False
    manifest, _read_error = _h3_project_read_manifest(manifest_path)
    if manifest is None:
        manifest = dict(project.get("manifest") or {})
    if not manifest:
        return False

    now = _datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    previous_base_key = str(
        ((manifest.get("sampler") or {}).get("base_key") or ""))
    base_key = str(base_key or "")
    base_changed = bool(previous_base_key and previous_base_key != base_key)
    if base_changed:
        # KursatAs 2026-08-19 19:28: a changed sampler/cache base means the
        # old prefix .pt files represent a different render configuration.
        # The sampler would refuse them anyway; clearing metadata keeps the
        # project report honest.
        cache = manifest.setdefault("cache", {})
        cache["prefixes"] = []
        cache["last_completed_prefix"] = 0
        cache["cached_prefix_shots"] = 0
        cache["first_changed_shot"] = 1
        cache["safe_prefix_shots"] = 0
        cache["render_state"] = "dirty"
        cache["dirty_reason"] = "sampler_base_changed"
        cache["needs_render_from"] = 1
        cache["invalidated_by_sampler_base_change"] = {
            "previous_base_key": previous_base_key,
            "new_base_key": base_key,
            "updated_at": now,
        }
    # KursatAs 2026-08-19 19:22: the project manifest records the exact
    # sampler/cache identity that produced the prefix state. The .pt files are
    # still the source of truth; this is the human/debuggable project index.
    manifest["sampler"] = {
        "base_key": base_key,
        "configured_at": now,
        "cache_base": _h3_project_jsonable(cache_base),
    }
    manifest["updated_at"] = now
    _h3_project_write_manifest(manifest_path, manifest)
    project["manifest"] = manifest
    project["sampler_base_key"] = base_key
    project["sampler_base_changed"] = base_changed
    if base_changed:
        project["cached_prefix_shots"] = 0
    return True


class H3AdvanceProject:
    """Create or load a reusable H3 Advance project manifest."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "project_name": ("STRING", {
                "default": "My_H3_Project",
                "tooltip": "Project folder name under ComfyUI/user/"
                           "multishot_advance_projects. The name is "
                           "sanitized for disk."
            }),
            "script": ("STRING", {
                "multiline": True,
                "default": "",
                "tooltip": "Shot script. Plain text shots can be separated "
                           "with --- lines; JSON prompt lists are also "
                           "accepted like the sampler."
            }),
            "mode": (["load_or_create", "read_only", "rebuild_project"], {
                "default": "load_or_create",
                "tooltip": "load_or_create writes project.json; read_only "
                           "only reports; rebuild_project marks all current "
                           "shots dirty for the future project-aware sampler."
            }),
        }, "optional": {
            "shot_count": ("INT", {
                "default": 0, "min": 0, "max": 100, "step": 1,
                "tooltip": "Match the sampler's shot_count. 0 = use script "
                           "shot count. If greater than the script, the last "
                           "prompt is repeated exactly like the sampler."
            }),
            "apply_shot_override": ("BOOLEAN", {
                "default": False,
                "tooltip": "Enable replacing one effective shot prompt before "
                           "the project computes hashes/cache invalidation."
            }),
            "override_shot_index": ("INT", {
                "default": 0, "min": 0, "max": 100, "step": 1,
                "tooltip": "1-based shot number to replace when "
                           "apply_shot_override is enabled. 0 = disabled."
            }),
            "override_prompt": ("STRING", {
                "multiline": True,
                "default": "",
                "tooltip": "Replacement prompt for override_shot_index. "
                           "The edited effective script becomes the project "
                           "script so it can be reopened later."
            }),
        }}

    RETURN_TYPES = ("H3_ADVANCE_PROJECT", "STRING", "INT", "STRING")
    RETURN_NAMES = ("project", "script_out", "shot_count_out", "report")
    FUNCTION = "build"
    CATEGORY = "conditioning/minimax"

    def build(self, project_name, script, mode="load_or_create",
              shot_count=0, apply_shot_override=False,
              override_shot_index=0, override_prompt=""):
        mode = str(mode or "load_or_create")
        if mode not in ("load_or_create", "read_only", "rebuild_project"):
            mode = "load_or_create"

        slug = _h3_project_slug(project_name)
        root = _h3_project_root()
        project_dir = _os.path.join(root, slug)
        manifest_path = _h3_project_manifest_path(project_dir)
        cache_dir = _h3_project_cache_dir(project_dir)

        previous, read_error = _h3_project_read_manifest(manifest_path)
        input_script = str(script or "")
        input_has_script = bool(input_script.strip())
        requested_shot_count = max(0, _h3_project_int(shot_count, 0))
        loaded_shot_count_from_manifest = False
        loaded_script_from_manifest = False

        if (not input_has_script and previous is not None
                and read_error is None):
            shots = _h3_project_shots_from_manifest(previous)
            if shots:
                # KursatAs 2026-08-19 19:20: empty script input means "load
                # this named project", not "overwrite the project with one
                # blank shot". This is the actual reopen-next-day path.
                input_script = str(previous.get("script_text")
                                   or previous.get("source_script_text")
                                   or _h3_project_script_from_shots(shots))
                if requested_shot_count <= 0:
                    requested_shot_count = max(
                        0, _h3_project_int(
                            previous.get("requested_shot_count",
                                         previous.get("shot_count", 0)), 0))
                    loaded_shot_count_from_manifest = (
                        requested_shot_count > 0)
                loaded_script_from_manifest = True

        shots = _h3_project_prepare_shots(input_script, requested_shot_count)
        # KursatAs 2026-08-19 19:39: apply shot override before hashes are
        # computed. Otherwise a separate editor node could change the script
        # after Project has already decided which prefix cache is safe.
        shots, override_applied, override_report = (
            _h3_project_apply_shot_override(
                shots, override_shot_index, override_prompt,
                bool(apply_shot_override)))
        shot_hashes = [_h3_project_sha256_text(shot) for shot in shots]
        source_script_text = str(input_script or "")
        script_text = _h3_project_script_from_shots(shots)
        if override_applied:
            # The edited effective script is now the canonical project script.
            # Keeping the pre-edit source would lose overrides on reopen,
            # especially for repeated shot_count-expanded prompts.
            source_script_text = script_text
        script_hash = _h3_project_sha256_text(script_text)
        effective_shot_count = len(shots)

        previous_hashes = [
            str(item.get("hash", ""))
            for item in (previous or {}).get("shots", [])
            if isinstance(item, dict)
        ]
        previous_requested_shot_count = _h3_project_int(
            (previous or {}).get("requested_shot_count",
                                 (previous or {}).get("shot_count", 0)), 0)

        first_changed = _h3_project_first_changed_shot(
            previous_hashes, shot_hashes)
        dirty_reason = ""
        if previous is None and read_error is None:
            first_changed = 1 if shot_hashes else None
            dirty_reason = "new_project" if shot_hashes else ""
        elif first_changed is not None:
            if previous_requested_shot_count != requested_shot_count:
                dirty_reason = "shot_count_changed"
            elif override_applied:
                dirty_reason = "shot_override"
            else:
                dirty_reason = "prompt_changed"
        if mode == "rebuild_project" and shot_hashes:
            first_changed = 1
            dirty_reason = "rebuild_project"

        safe_prefix = len(shot_hashes) if first_changed is None else max(
            0, int(first_changed) - 1)

        preserved_prefixes, missing_prefixes = _h3_project_preserved_prefixes(
            previous, safe_prefix, project_dir)

        manifest = _h3_project_make_manifest(
            project_name, slug, source_script_text, script_text, shots,
            shot_hashes, script_hash,
            requested_shot_count, safe_prefix, first_changed, mode,
            dirty_reason,
            preserved_prefixes, previous)

        wrote_manifest = False
        write_error = None
        if mode != "read_only":
            if read_error and mode != "rebuild_project":
                write_error = ("Existing project.json is unreadable; choose "
                               "rebuild_project to replace it. Error: "
                               + read_error)
            else:
                try:
                    _os.makedirs(cache_dir, exist_ok=True)
                    _h3_project_write_manifest(manifest_path, manifest)
                    wrote_manifest = True
                except Exception as exc:
                    write_error = str(exc)

        report_lines = [
            "H3 Advance Project",
            f"project: {project_name} ({slug})",
            f"mode: {mode}",
            f"shots: {effective_shot_count}",
            f"folder: {project_dir}",
        ]
        if read_error:
            report_lines.append(f"manifest read error: {read_error}")
        elif previous is None:
            report_lines.append("previous manifest: none")
        else:
            report_lines.append("previous manifest: loaded")
        report_lines.append(
            "script source: %s"
            % ("loaded from project.json" if loaded_script_from_manifest
               else "input"))
        report_lines.append(
            "shot_count source: %s"
            % ("loaded from project.json" if loaded_shot_count_from_manifest
               else "input/default"))
        if override_report:
            report_lines.append("shot override: " + override_report)

        if first_changed is None:
            report_lines.append(
                f"script unchanged; safe prefix: {safe_prefix}/{len(shots)}")
        else:
            report_lines.append(
                "first changed shot: %d; safe prefix: %d/%d; rerender %d..%d"
                % (first_changed, safe_prefix, len(shots), first_changed,
                   len(shots)))
        if preserved_prefixes:
            report_lines.append(
                "cached prefix metadata: %d/%d"
                % (_h3_project_last_prefix(preserved_prefixes), len(shots)))
        if missing_prefixes:
            report_lines.append(
                "cached prefix metadata: dropped %d missing file(s)"
                % missing_prefixes)
        cache_summary = manifest.get("cache") or {}
        report_lines.append(
            "render state: %s%s" % (
                cache_summary.get("render_state", "unknown"),
                "" if cache_summary.get("needs_render_from") is None
                else "; next render from shot %s"
                % cache_summary.get("needs_render_from")))
        if cache_summary.get("dirty_reason"):
            report_lines.append(
                "dirty reason: %s" % cache_summary.get("dirty_reason"))
        if isinstance((previous or {}).get("sampler"), dict):
            report_lines.append("sampler config: preserved")

        if wrote_manifest:
            report_lines.append("manifest: written")
        elif mode == "read_only":
            report_lines.append("manifest: read-only, not written")
        elif write_error:
            report_lines.append("manifest write skipped: " + write_error)

        project = {
            "type": _H3_ADVANCE_PROJECT_TYPE,
            "version": _H3_ADVANCE_PROJECT_VERSION,
            "name": str(project_name or ""),
            "slug": slug,
            "mode": mode,
            "project_root": root,
            "project_dir": project_dir,
            "manifest_path": manifest_path,
            "cache_dir": cache_dir,
            "script_hash": script_hash,
            "script_text": script_text,
            "script_loaded_from_manifest": loaded_script_from_manifest,
            "requested_shot_count": requested_shot_count,
            "shot_count_loaded_from_manifest":
                loaded_shot_count_from_manifest,
            "shot_override_applied": override_applied,
            "shot_override_report": override_report,
            # KursatAs 2026-08-19 19:33: expose the effective shot_count so
            # project and sampler cannot silently disagree about repeated or
            # truncated shots.
            "effective_shot_count": effective_shot_count,
            "shot_prompts": list(shots),
            "shot_hashes": list(shot_hashes),
            "previous_shot_hashes": previous_hashes,
            "first_changed_shot": first_changed,
            "safe_prefix_shots": safe_prefix,
            "render_state": (manifest.get("cache") or {}).get(
                "render_state", "unknown"),
            "dirty_reason": (manifest.get("cache") or {}).get(
                "dirty_reason", ""),
            "cached_prefix_shots": (manifest.get("cache") or {}).get(
                "cached_prefix_shots", 0),
            "needs_render_from": (manifest.get("cache") or {}).get(
                "needs_render_from"),
            "missing_prefix_files": int(missing_prefixes),
            "manifest": manifest,
            "sampler_base_key": str(
                ((manifest.get("sampler") or {}).get("base_key") or "")),
            "manifest_written": wrote_manifest,
            "manifest_error": write_error or read_error,
        }
        return (project, script_text, effective_shot_count,
                "\n".join(report_lines))


NODE_CLASS_MAPPINGS = {"H3AdvanceProject": H3AdvanceProject}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3AdvanceProject": "H3 Advance Project",
}
