# -*- coding: utf-8 -*-
"""H3 multishot sampler node - single-script prompting for the
MiniMax H3 chained workflow.

Accepts these script formats:
  - JSON: {"prompts": ["shot 1 ...", "shot 2 ...", "shot 3 ..."]}
  - plain text with --- separators between shots
Feeds up to 4 shot prompts as separate STRING outputs. Missing shots fall
back to the previous shot's prompt so a 2-shot script still runs a 3-shot
graph without erroring.
"""
import json
import re

try:
    from .h3_notify import (
        h3_error as _h3_error,
        h3_info as _h3_info,
    )
except Exception:
    try:
        from h3_notify import (
            h3_error as _h3_error,
            h3_info as _h3_info,
        )
    except Exception:
        def _h3_noop(*_args, **_kwargs):
            return False
        _h3_info = _h3_error = _h3_noop

try:
    from .h3_cache import _H3ShotCacheSession
except Exception:
    from h3_cache import _H3ShotCacheSession

try:
    from .h3_references import (
        _h3_build_reference_image_slots,
        _h3_build_voice_anchor,
    )
except Exception:
    from h3_references import (
        _h3_build_reference_image_slots,
        _h3_build_voice_anchor,
    )

try:
    from .h3_memory_bank import _H3ChainBank
except Exception:
    from h3_memory_bank import _H3ChainBank

try:
    from .h3_audio import (
        _aud_env,
        _jb_audio_window,
        _wav_for_vae,
    )
except Exception:
    from h3_audio import (
        _aud_env,
        _jb_audio_window,
        _wav_for_vae,
    )

try:
    from .h3_upscale import (
        _upscale_av_exact,
        _upscale_frames,
        _write_shot_mp4,
    )
except Exception:
    from h3_upscale import (
        _upscale_av_exact,
        _upscale_frames,
        _write_shot_mp4,
    )

try:
    from .h3_model_runtime import (
        _auto_measure_begin,
        _auto_measure_end,
        _auto_set_payload,
    )
except Exception:
    from h3_model_runtime import (
        _auto_measure_begin,
        _auto_measure_end,
        _auto_set_payload,
    )

try:
    from .h3_join import (
        _jb_centre_clip,
        _jb_grid,
    )
except Exception:
    from h3_join import (
        _jb_centre_clip,
        _jb_grid,
    )

try:
    from .h3_sampler_schema import h3_memory_sampler_input_types
except Exception:
    from h3_sampler_schema import h3_memory_sampler_input_types

try:
    from .h3_sampler_cache import h3_build_sampler_cache_base
except Exception:
    from h3_sampler_cache import h3_build_sampler_cache_base

try:
    from .h3_project import (
        h3_advance_project_active,
        h3_advance_project_record_prefix,
        h3_advance_project_record_sampler_config,
    )
except Exception:
    try:
        from h3_project import (
            h3_advance_project_active,
            h3_advance_project_record_prefix,
            h3_advance_project_record_sampler_config,
        )
    except Exception:
        def h3_advance_project_active(_project):
            return False
        def h3_advance_project_record_prefix(*_args, **_kwargs):
            return False
        def h3_advance_project_record_sampler_config(*_args, **_kwargs):
            return False

try:
    from .h3_sampler_setup import (
        h3_apply_sampler_overrides,
        h3_handoff_geometry,
        h3_load_upscale_model,
        h3_make_stream_assembler,
        h3_prepare_shots,
        h3_prepare_sigmas_and_sampler,
        h3_prepare_two_pass,
        h3_reference_cap,
        h3_reference_rows_expected,
        h3_report_chain_setup,
        h3_report_checkpoint_setup,
    )
except Exception:
    from h3_sampler_setup import (
        h3_apply_sampler_overrides,
        h3_handoff_geometry,
        h3_load_upscale_model,
        h3_make_stream_assembler,
        h3_prepare_shots,
        h3_prepare_sigmas_and_sampler,
        h3_prepare_two_pass,
        h3_reference_cap,
        h3_reference_rows_expected,
        h3_report_chain_setup,
        h3_report_checkpoint_setup,
    )

try:
    from .h3_sampler_refs import (
        h3_build_shot_keyframes,
        h3_build_shot_refs,
    )
except Exception:
    from h3_sampler_refs import (
        h3_build_shot_keyframes,
        h3_build_shot_refs,
    )

try:
    from .h3_sampler_conditioning import h3_build_shot_conditioning
except Exception:
    from h3_sampler_conditioning import h3_build_shot_conditioning

# KursatAs 2026-08-19 04:45: speech guard is sampler-level, not writer-level.
# Silent shots keep visual references but must not receive old voice/audio refs
# that can make H3 invent random-language dialogue.
try:
    from .h3_speech_guard import (
        h3_apply_no_speech_guard,
        h3_detect_foreground_speech,
    )
except Exception:
    from h3_speech_guard import (
        h3_apply_no_speech_guard,
        h3_detect_foreground_speech,
    )

try:
    from .h3_sampler_runtime import (
        h3_apply_context_pin,
        h3_patch_handoff_guider,
    )
except Exception:
    from h3_sampler_runtime import (
        h3_apply_context_pin,
        h3_patch_handoff_guider,
    )

try:
    from .h3_sampler_vram import (
        h3_evict_dit_before_text_encoder,
        h3_prepare_sampling_memory,
    )
except Exception:
    from h3_sampler_vram import (
        h3_evict_dit_before_text_encoder,
        h3_prepare_sampling_memory,
    )

try:
    from .h3_sampler_decode import h3_decode_shot_outputs
except Exception:
    from h3_sampler_decode import h3_decode_shot_outputs

try:
    from .h3_sampler_quality import (
        h3_apply_shot_quality,
        h3_report_continuity_diagnostics,
    )
except Exception:
    from h3_sampler_quality import (
        h3_apply_shot_quality,
        h3_report_continuity_diagnostics,
    )

try:
    from .h3_sampler_joining import h3_apply_join_handling
except Exception:
    from h3_sampler_joining import h3_apply_join_handling

try:
    from .h3_sampler_output import h3_finalize_sampler_outputs
except Exception:
    from h3_sampler_output import h3_finalize_sampler_outputs

try:
    from .h3_sampler_state import (
        H3SamplerState,
        h3_build_sampler_cache_state,
        h3_restore_sampler_cache_state,
        h3_update_post_decode_state,
    )
except Exception:
    from h3_sampler_state import (
        H3SamplerState,
        h3_build_sampler_cache_state,
        h3_restore_sampler_cache_state,
        h3_update_post_decode_state,
    )


def _h3_fail(message, exc_type=RuntimeError, title="H3 Multishot", tag=None):
    """Build a hard-fail exception after sending the same text as a red toast.

    KursatAs 2026-08-18 05:47: H3's known operator-fixable failures should
    not live only in the console or ComfyUI queue error panel during long runs.
    """
    msg = str(message)
    _h3_error(msg, title=title, tag=tag)
    return exc_type(msg)



def _repair_json(text):
    """Parse JSON, auto-closing unterminated brackets/quotes.

    Long multi-prompt scripts get truncated or lose their final brace all the
    time (a 4,500-char script with the closing '}' missing is not a typo the
    author can see). Returns (data, note): data is None on real failure and
    note carries the error; note is a description when a repair was applied,
    or "" when the text parsed clean.
    """
    try:
        return json.loads(text), ""
    except json.JSONDecodeError as e:
        first_err = str(e)   # bind now; Python clears the except-name on exit

    # walk the text tracking string state, then close what is still open
    stack, in_str, esc = [], False, False
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack and ((ch == "}" and stack[-1] == "{") or
                          (ch == "]" and stack[-1] == "[")):
                stack.pop()

    candidate = text.rstrip()
    fixes = []
    if in_str:
        candidate += '"'
        fixes.append("closed an open string")
    if candidate.endswith(","):
        candidate = candidate[:-1]
        fixes.append("dropped a trailing comma")
    # trailing comma before a closer, e.g.  ["a","b",]  or  {"k":1,}
    cleaned = re.sub(r",(\s*[\]}])", r"\1", candidate)
    if cleaned != candidate:
        candidate = cleaned
        fixes.append("removed comma(s) before a closing bracket")
    for opener in reversed(stack):
        candidate += "}" if opener == "{" else "]"
    if stack:
        fixes.append("added " + "".join("}" if o == "{" else "]"
                                        for o in reversed(stack)))
    if not fixes:
        return None, first_err
    try:
        return json.loads(candidate), ", ".join(fixes)
    except json.JSONDecodeError as e:
        return None, str(e)

def _parse_script(text):
    """Script -> list of shot prompts. JSON {"prompts": [...]} or
    plain text with --- separators. Malformed JSON fails LOUD."""
    text = (text or "").strip()
    shots = []
    if text.startswith("{") or text.startswith("["):
        data, repaired = _repair_json(text)
        if data is None:
            _msg = (
                f"H3 script looks like JSON but does not parse ({repaired}). "
                f"Auto-repair of unclosed brackets/quotes was attempted and "
                f"failed. Common cause: a doubled {{ on the first lines, or a "
                f"missing comma between prompts. Fix the script or use plain "
                f"prompts separated by --- lines.")
            raise _h3_fail(_msg, ValueError, "H3 script parse",
                           tag="H3Script")
        if repaired:
            print(f"[H3Multishot] script JSON was incomplete; auto-repaired "
                  f"({repaired}). Consider fixing the source.", flush=True)
        if isinstance(data, dict):
            shots = [str(p) for p in data.get("prompts", [])]
        elif isinstance(data, list):
            shots = [str(p) for p in data]
    if not shots:
        shots = [b.strip().replace('\\"', '"')
                 for b in re.split(r"(?m)^---\s*$", text) if b.strip()]
    if not shots:
        shots = [text]
    return shots


class H3MultishotMemorySampler:
    """Multishot with a MEMORY BANK.

    There is no keyframe here. Shots are not continued pixel-wise from their
    predecessor; each is generated fresh and held together by a bank of past
    shots injected as reference conditioning. That architecture is why chains
    do not accumulate texture drift: a shot is
    never a pure function of the shot before it, because the bank always
    contains the beginning of the episode.

    Bank slot = a short video clip from the MIDDLE of a shot + the audio under
    it, injected as an H3 `video_audio` reference. The first `bank_pinned`
    slots are never evicted; the rest is a bounded recency window.

    REQUIRES a ref2va checkpoint - reference rows are what this node is built
    on, and fl2va was not trained with them.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return h3_memory_sampler_input_types()

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT", "LATENT", "LATENT", "INT",
                    "STRING")
    RETURN_NAMES = ("master_frames", "master_audio", "shots_rendered",
                    "video_latents", "audio_latents", "head_frames",
                    "master_path")
    OUTPUT_TOOLTIPS = (
        "The joined master, seams trimmed.",
        "Master audio.",
        "How many shots rendered.",
        "Every shot's video latent EXACTLY as sampled, batched along dim 0 "
        "(one entry per shot), UNTRIMMED. Shots after the first open with "
        "head_frames of replayed material from the previous shot's tail - "
        "that is what makes the join seamless, and it is removed at decode, "
        "not in the latent. So this does NOT line up with master_frames "
        "until you trim it yourself. Latent temporal rows for F frames are "
        "5*((F-5)//17)+2.",
        "The matching audio latent per shot, same batching, same caveat.",
        "Frames of replayed head carried by every shot AFTER the first "
        "(0 for shot 1, and 0 for continuity modes that do not pin). Trim "
        "this many frames off the front of each later shot after you decode.",
        "low_ram_master only: the finished master file on disk. Empty string "
        "when low_ram_master is off (use master_frames as always).")
    FUNCTION = "run"
    CATEGORY = "sampling/minimax"

    @classmethod
    def VALIDATE_INPUTS(cls, memory_frames):
        # Naming memory_frames here hands ITS range check to us and leaves
        # every other input on ComfyUI's own validation.
        #
        # Out of range almost never means the user typed it. v1.2 inserted
        # seed_per_shot at widget index 8, ahead of memory_frames, so a
        # workflow saved on v1.0/v1.1 reads its old anchor_frames (0-9) into
        # memory_frames (0-3) - and the stock message, "Value 4 bigger than
        # max of 3", sends people hunting a dial they never touched. The JS
        # extension repairs that on load; this is the backstop for anyone
        # running with custom frontend extensions disabled.
        if not 0 <= int(memory_frames) <= 3:
            return (f"memory_frames is {memory_frames}, but the range is 0-3 "
                    f"(H3 holds at most 3 references, pinned + recent).\n"
                    f"If you did not set it: this workflow was saved before "
                    f"v1.2, which added seed_per_shot ahead of memory_frames "
                    f"and shifted every dial after it by one. Re-open the "
                    f"shipped H3_Seamless_Chain_v2.json, or right-click the "
                    f"sampler -> Fix node (recreate) and re-enter your "
                    f"settings, then save.")
        return True

    def run(self, model, clip, video_vae, audio_vae, script, shot_count, width,
            height, frames_per_shot, seed, steps, memory_frames, anchor_frames,
            seed_per_shot=True, start_image=None,
            sampler_name="res_multistep", scheduler="simple",
            bank_pinned=1, chain_gain_control="off", bank_clip_frames=22,
            continuity="cut", color_level="off", join_anchor_noise=0.0,
            join_blend=False, handoff_release=0.30, bank_ref_noise=0.0,
            end_anchor=False, join_fx="off", audio_lock=True,
            handoff_taper=0, handoff_depth="block", guide_audio=None,
            keyframe_images=None, reference_images=None, voice_ref=None,
            sampler_override=None, scheduler_override=None,
            self_anchor_voice=False, reference_image_size="match",
            preview_first_shot=False,
            save_every_shot=False, sigmas=None, output_scale=1.0,
            upscale_model_name="(none)",
            master_normalize="luma+contrast", pin_frames="22", pin_noise=0.0,
            pin_renorm=False, reference_subjects="",
            reference_video=None, reference_video_audio=None,
            low_ram_master=False, audio_pin_frames=0, shot_cache="use_cache",
            project=None, prompt=None, extra_pnginfo=None):
        # Keep the hidden PROMPT before anything can shadow it: the shot loop
        # rebinds `prompt` to this shot's conditioning TEXT, so by finalize()
        # the API graph is gone and the streamed master was tagged with the
        # last shot's script (24 GB test-lab finding F014, 2026-08-16 - core SaveVideo wrote
        # a 42-node dict, the streamed master a 1254-char string, same run).
        _api_prompt, _api_pnginfo = prompt, extra_pnginfo
        # two_pass_upscale is REMOVED as of 2.1.3. It spatially interpolated
        # the raw latent between passes, and H3's latent is not a spatially
        # smooth representation - interpolated values land off-manifold and
        # pass 2, running at low sigma, has no room to pull them back. Three
        # A/Bs at the previously documented recipe (14 steps, beta57) came
        # back as colour-noise mush against clean single-pass controls, with
        # shot 1 - which carries no pin at all - destroyed identically, so it
        # was never about the chaining. Upscaling now happens AFTER decode
        # (output_scale / upscale_model), where it cannot leave the manifold.
        # The branches below are kept only so the diff stays readable; they
        # are unreachable.
        two_pass_upscale = False
        upscale_factor, pass1_fraction, upscale_audio_denoise = 1.5, 0.4, 0.35
        sampler_name, scheduler = h3_apply_sampler_overrides(
            sampler_name, scheduler, sampler_override, scheduler_override)
        import torch
        import node_helpers
        from comfy_extras import nodes_custom_sampler as ncs
        from comfy_extras import nodes_minimax_h3 as mmh3
        from comfy_extras.nodes_audio import vae_decode_audio
        import comfy.model_management as _mm

        shots, n = h3_prepare_shots(script, shot_count, _parse_script)
        steps, sigmas, sampler = h3_prepare_sigmas_and_sampler(
            model, scheduler, steps, sigmas, sampler_name, ncs)

        # --- voice anchor: encode ONCE, ride in every shot's conditioning ---
        # The bank already carries voice from shot 2 on, but shot 1 renders
        # against an empty bank; an explicit ref covers the whole chain.
        voice_block = _h3_build_voice_anchor(audio_vae, voice_ref)

        # --- subject/character reference images: encode ONCE, fixed slots ---
        ref_image_items, ref_image_blocks = _h3_build_reference_image_slots(
            mmh3, video_vae, reference_images, reference_image_size,
            width, height)

        (_tp_tr, _tp_w1, _tp_h1, _tp_sig_hi, _tp_sig_lo,
         _tp_lat_th, _tp_lat_tw, _tp_lat_h1, _tp_lat_w1) = h3_prepare_two_pass(
            two_pass_upscale, continuity, width, height, frames_per_shot,
            steps, sigmas, upscale_factor, pass1_fraction,
            upscale_audio_denoise, mmh3)

        cap = h3_reference_cap(bank_pinned, memory_frames)

        _ref_rows_expected = h3_reference_rows_expected(
            n, cap, reference_images, reference_video, voice_ref, start_image,
            self_anchor_voice)
        _, _is_fl, _ = h3_report_checkpoint_setup(
            model, continuity, _ref_rows_expected, start_image)

        bank = _H3ChainBank(num_fix=bank_pinned, max_size=cap)
        frames_parts, audio_parts = [], []
        _stream_assembler = h3_make_stream_assembler(
            low_ram_master, join_fx, color_level)

        _lat_v_parts, _lat_a_parts = [], []   # issue #12: raw per-shot latents
        upscale_model = h3_load_upscale_model(
            upscale_model_name, output_scale, width, height, frames_per_shot,
            len(shots))
        state = H3SamplerState()
        _CG_WIN = 24
        _geom = h3_handoff_geometry(handoff_depth, continuity)
        _TAIL_K = _geom["tail_k"]
        _OV = _geom["ov"]
        _HO_ROWS = _geom["ho_rows"]
        _HO_R0 = _geom["ho_r0"]
        _HO_ACOLS = _geom["ho_acols"]
        _OV_HO = _geom["ov_ho"]
        _spine = None          # encoded audio spine (guide_audio)
        _TRIM = _geom["trim"]
        if guide_audio is not None:
            _gw3, _g_sr = _wav_for_vae(audio_vae, guide_audio, "audio spine")
            _spine = audio_vae.encode(_gw3.movedim(1, -1)).detach()
            print("[H3Memory] audio spine: %d cols (%.1fs) - every shot's "
                  "audio is locked to a slice of it, so the VOICE cannot "
                  "change between shots (fl2va has no reference rows to "
                  "carry a voice; this is how you keep one performance)."
                  % (_spine.shape[-1],
                     _gw3.shape[-1] / float(_g_sr)),
                  flush=True)
            if two_pass_upscale:
                # the spine lock lives in a predict_noise patch built around
                # ONE guider; pass 2 runs its own guider, so half the
                # trajectory would sample unlocked audio and the voice would
                # move exactly where the spine exists to hold it still
                _msg = (
                    "two_pass_upscale cannot be combined with an audio spine "
                    "(guide_audio). The spine locks audio through every "
                    "sampling step of a single trajectory; a two-pass render "
                    "is two trajectories. Disconnect guide_audio, or turn "
                    "two_pass_upscale off.")
                raise _h3_fail(_msg, ValueError, "H3 incompatible settings",
                               tag="H3Memory")
        h3_report_chain_setup(
            chain_gain_control, continuity, n, bank_pinned, cap, _is_fl,
            bank_clip_frames, _jb_grid)

        # KursatAs 2026-08-18 06:24: shot_cache is a CPU/disk prefix cache,
        # not a GPU cache. A cache hit restores the exact sampler state after
        # the last unchanged --- prompt and starts rendering at the first
        # changed shot, so prompt edits near the end do not re-pay shot 1..N.
        _project_active = h3_advance_project_active(project)
        _project_cache_dir = project.get("cache_dir") if _project_active else None
        _project_safe_prefix = (
            project.get("safe_prefix_shots") if _project_active else None)
        _project_write_allowed = not (
            _project_active and str(project.get("mode") or "") == "read_only")
        if _project_active:
            # KursatAs 2026-08-19 19:14: safe project editing means the
            # sampler may restore only the unchanged prefix before the first
            # changed shot. Shot K changed => restore at most K-1.
            print("[H3Memory] project '%s': safe prefix %s/%d"
                  % (project.get("slug") or project.get("name"),
                     _project_safe_prefix, n), flush=True)
        _cache = _H3ShotCacheSession(
            shot_cache, shots, n, cache_dir=_project_cache_dir,
            restore_prefix_limit=_project_safe_prefix,
            cache_label="project_cache" if _project_active else "shot_cache",
            write_allowed=_project_write_allowed)
        _cache_start = 0

        if _cache.enabled:
            if _stream_assembler is not None:
                _cache.disable_for_streaming()
            else:
                try:
                    _base_obj = h3_build_sampler_cache_base(
                        width=width, height=height,
                        frames_per_shot=frames_per_shot, seed=seed,
                        steps=steps, seed_per_shot=seed_per_shot,
                        sampler_name=sampler_name, scheduler=scheduler,
                        sigmas=sigmas, memory_frames=memory_frames,
                        anchor_frames=anchor_frames, bank_pinned=bank_pinned,
                        chain_gain_control=chain_gain_control,
                        bank_clip_frames=bank_clip_frames,
                        continuity=continuity, color_level=color_level,
                        join_anchor_noise=join_anchor_noise,
                        join_blend=join_blend,
                        handoff_release=handoff_release,
                        bank_ref_noise=bank_ref_noise, end_anchor=end_anchor,
                        join_fx=join_fx, audio_lock=audio_lock,
                        handoff_taper=handoff_taper,
                        handoff_depth=handoff_depth,
                        self_anchor_voice=self_anchor_voice,
                        reference_image_size=reference_image_size,
                        preview_first_shot=preview_first_shot,
                        save_every_shot=save_every_shot,
                        output_scale=output_scale,
                        upscale_model_name=upscale_model_name,
                        master_normalize=master_normalize,
                        pin_frames=pin_frames, pin_noise=pin_noise,
                        pin_renorm=pin_renorm,
                        reference_subjects=reference_subjects,
                        audio_pin_frames=audio_pin_frames,
                        model=model, clip=clip, video_vae=video_vae,
                        audio_vae=audio_vae, start_image=start_image,
                        keyframe_images=keyframe_images,
                        reference_images=reference_images,
                        guide_audio=guide_audio, voice_ref=voice_ref,
                        reference_video=reference_video,
                        reference_video_audio=reference_video_audio,
                        project=project if _project_active else None)
                    _cache.configure(_base_obj)
                    if _project_active:
                        h3_advance_project_record_sampler_config(
                            project, _cache.base_key, _base_obj)
                    _cache_start, _state = _cache.restore_prefix()
                    if _state is not None:
                        (
                            frames_parts,
                            audio_parts,
                            _lat_v_parts,
                            _lat_a_parts,
                            voice_block,
                            state,
                        ) = h3_restore_sampler_cache_state(_state, bank)
                except Exception as _e:
                    _cache.disable_due_to_error(_e)

        for si, prompt in enumerate(shots[_cache_start:], start=_cache_start):
            shot_seed = (seed + si) if seed_per_shot else seed
            # KursatAs 2026-08-19 04:45: decide per-shot whether foreground
            # speech is explicitly requested. Defaulting to silent is deliberate
            # because the failure mode is invented speech when the prompt only
            # asked for action/ambience.
            speech_active, speech_reason = h3_detect_foreground_speech(prompt)
            if not speech_active:
                prompt = h3_apply_no_speech_guard(prompt)
            # KursatAs 2026-08-17 10:25: short per-clip toast confirms the
            # browser notification bridge is alive without requiring an error.
            _h3_info(f"Clip {si + 1}/{n} running",
                     topic="memory_sampler", tag="H3Memory",
                     timeout_ms=3000)
            if two_pass_upscale:
                latent, frame_count = mmh3._empty_av_latent(
                    _tp_w1, _tp_h1, frames_per_shot)
                _p1v, _p1n = _tp_tr.extract_tensor(latent["samples"])
                _tp_lat_h1 = int(_p1v[0].shape[-2])
                _tp_lat_w1 = int(_p1v[0].shape[-1])
            else:
                latent, frame_count = mmh3._empty_av_latent(width, height,
                                                            frames_per_shot)
            ref_items, ref_blocks = h3_build_shot_refs(
                mmh3, video_vae, audio_vae, si, width, height,
                anchor_frames, start_image, ref_image_items, ref_image_blocks,
                voice_block, reference_video, reference_video_audio,
                continuity, bank, speech_active=speech_active)
            prompt, kf_vision, keyframes = h3_build_shot_keyframes(
                mmh3, si, prompt, continuity, keyframe_images,
                state.last_tail,
                width, height, frame_count, frames_per_shot, _TAIL_K,
                handoff_depth, end_anchor, state.house_frame, state.dbg_pins)

            print("[H3Memory] shot %d/%d (%df @ %dx%d) | bank %s%s | "
                  "speech %s (%s)"
                  % (si + 1, n, frames_per_shot, width, height, bank.describe(),
                     " + identity ref" if (start_image is not None
                                           and anchor_frames > 0) else "",
                     "on" if speech_active else "off", speech_reason),
                  flush=True)

            h3_evict_dit_before_text_encoder(si)

            prompt, cond, cond_hi = h3_build_shot_conditioning(
                clip, node_helpers, mmh3, video_vae, prompt, si,
                ref_items, ref_blocks, kf_vision, keyframes,
                reference_subjects, two_pass_upscale, width, height,
                frame_count, join_anchor_noise, shot_seed,
                pass1_width=_tp_w1, pass1_height=_tp_h1,
                speech_active=speech_active)

            # context_pin: reuse the Motion-Context node as a library via
            # the registry - OUR features (bank, colour levels, join fx)
            # stay; THEIR mechanism (interior latent pin + timeline audio
            # ref + payload coexistence patches) rides on the conditioning.
            state.cp_trim = 0
            cond, state.cp_trim, state.pin_sig0, state.pin_hf0 = \
                h3_apply_context_pin(
                cond, continuity, si, state.cp_prev, video_vae, latent,
                two_pass_upscale, _tp_tr, _tp_lat_h1, _tp_lat_w1,
                pin_frames, pin_renorm, state.pin_sig0,
                chain_gain_control, state.pin_hf0, state.cg_ref,
                state.cg_last_raw, pin_noise, shot_seed, audio_pin_frames)

            h3_prepare_sampling_memory(
                si, clip, model, video_vae, audio_vae, _mm)

            guider = ncs.BasicGuider().get_guider(model, cond)[0]
            guider_hi = (ncs.BasicGuider().get_guider(model, cond_hi)[0]
                         if two_pass_upscale else None)
            h3_patch_handoff_guider(
                guider, model, latent, _spine, si, frames_per_shot, _TRIM,
                continuity, state.ho_v, state.ho_a, state.ho_guard,
                audio_lock, _HO_R0, handoff_taper, state.ho_taper_src,
                handoff_release)
            noise = ncs.RandomNoise().get_noise(shot_seed)[0]
            # payload signature: continuity mode + position + bank/spine
            # decide the conditioning payload, and with it the real pool
            # KursatAs 2026-08-19 04:45: speech guard changes the number and
            # type of audio refs, so reserve/cache payload signatures must not
            # collapse silent and speaking shots into the same bucket.
            _audio_ref_count = sum(
                1 for item in ref_items if item["type"] == "audio")
            _auto_set_payload(
                "%s%d_k%dr%da%d%s%s%s" % (
                    continuity[:4], 1 if si > 0 else 0,
                    len(keyframes), len(ref_blocks), _audio_ref_count,
                    "sp" if speech_active else "sl",
                    "s" if _spine is not None else "",
                    "2p" if two_pass_upscale else ""))
            _mb = _auto_measure_begin()
            try:
                if two_pass_upscale:
                    out1, _d1 = ncs.SamplerCustomAdvanced().sample(
                        noise, guider, sampler, _tp_sig_hi, latent)
                    up = _upscale_av_exact(_tp_tr, out1, _tp_lat_th,
                                           _tp_lat_tw)
                    _s = max(0.0, min(1.0, float(upscale_audio_denoise)))
                    _members, _was_nested = _tp_tr.extract_tensor(up["samples"])
                    if _was_nested and len(_members) >= 2:
                        if _s <= 0.0:
                            _ridx, _rstr = (0,), None
                        elif _s >= 1.0:
                            _ridx, _rstr = (0, 1), None
                        else:
                            _ridx, _rstr = (0, 1), {0: 1.0, 1: _s}
                    else:
                        _ridx = _rstr = None
                    noise2 = ncs.RandomNoise().get_noise(shot_seed + 977)[0]
                    up = _tp_tr.add_noise_nested_latent(
                        model, noise2, _tp_sig_lo, up,
                        renoise_indices=_ridx, noise_strengths=_rstr)
                    up = _tp_tr.finalize_latent_for_handoff(up)
                    out, _d = ncs.SamplerCustomAdvanced().sample(
                        ncs.DisableNoise().get_noise()[0], guider_hi,
                        sampler, _tp_sig_lo, up)
                else:
                    out, _d = ncs.SamplerCustomAdvanced().sample(
                        noise, guider, sampler, sigmas, latent)
            finally:
                _auto_measure_end(_mb, model, steps=steps)

            (imgs, wav, sr, state.cp_prev, state.ho_taper_src, state.ho_v,
             state.ho_a) = h3_decode_shot_outputs(
                out, continuity, handoff_taper, _HO_ROWS, _HO_ACOLS,
                audio_lock, _lat_v_parts, _lat_a_parts, video_vae,
                audio_vae, vae_decode_audio, cp_prev=state.cp_prev,
                ho_taper_src=state.ho_taper_src, ho_v=state.ho_v,
                ho_a=state.ho_a)
            state.sr = sr

            (imgs, state.house_frame, state.cc_mu, state.cc_cov,
             state.cg_ref, state.cg_last_raw) = h3_apply_shot_quality(
                imgs, si, color_level, chain_gain_control, continuity,
                _CG_WIN, state.house_frame, state.cc_mu, state.cc_cov,
                state.cg_ref, state.cg_last_raw)

            state.dbg_pins = h3_report_continuity_diagnostics(
                imgs, wav, sr, continuity, si, state.last_tail,
                state.dbg_pins, handoff_depth, state.ho_wav_tail)

            if self_anchor_voice and voice_block is None and speech_active:
                # KursatAs 2026-08-19 04:45: self-anchor now starts from the
                # first speaking shot, not blindly shot 1. A silent shot can
                # contain ambience, but that must never become the chain voice.
                # THE self-anchor: the first rendered SPEAKING shot becomes
                # the reference for later speaking shots. Silent shots must
                # not seed the voice anchor with ambience, and they must not
                # receive that anchor back as conditioning. The bank carries voice as
                # part of a video_audio slot that keeps rolling; this pins
                # the ORIGINAL performance and never moves. The decoded audio
                # is already at the VAE's rate and stereo - just trim and
                # encode.
                _aw = wav[:1] if wav.ndim == 3 else wav.unsqueeze(0)[:1]
                _alim = 15 * sr
                if _aw.shape[-1] > _alim:
                    _aw = _aw[..., :_alim]
                _avz = audio_vae.encode(_aw.movedim(1, -1))
                voice_block = {"kind": "audio", "ref_audio_t": _avz.shape[-1],
                               "audio_latent": _avz}
                print("[H3Memory] self-anchor: shot %d voice (%.1fs) is now "
                      "<Audio 1> for the remaining %d shot(s)."
                      % (si + 1, _aw.shape[-1] / sr, max(0, n - si - 1)),
                      flush=True)
            # store this shot as a bank slot: centre clip + the audio under it
            clip_imgs, clip_start = _jb_centre_clip(imgs, bank_clip_frames)
            if bank_ref_noise > 0:
                # noised-clean-condition for the bank: the texture ratchet
                # rides reference clips exactly as it rode keyframes - the
                # model copies its own "pristine" output and adds ~1.2x
                # detail (worst on faces). A little seeded noise makes the
                # clip read as capture, so texture is regenerated, not
                # enhanced. The noised clip never reaches the final cut.
                _gn = torch.Generator().manual_seed(shot_seed ^ 0xBA9C)
                clip_imgs = (clip_imgs + bank_ref_noise * torch.randn(
                    clip_imgs.shape, generator=_gn).to(
                    clip_imgs.device, clip_imgs.dtype)).clamp(0, 1)
            bank.add((clip_imgs.clone(),
                      _jb_audio_window(wav, sr, clip_start,
                                       clip_imgs.shape[0])))

            # Upscale AFTER the bank has taken its clip: the bank must keep
            # base-resolution reference clips or the conditioning payload -
            # and the VRAM it costs - grows with output_scale for no gain.
            # Downstream of the VAE, so unlike the old two-pass path this
            # cannot leave the latent manifold. Frame COUNT is untouched, so
            # every seam-trim index below still means what it meant.
            imgs = _upscale_frames(imgs, output_scale, upscale_model,
                                   "H3Memory")
            if si == 0 and preview_first_shot:
                # shot 1 as early as possible, so a bad take can be cancelled
                # instead of waited out
                _write_shot_mp4(imgs, wav, sr,
                                "video/H3_FIRSTSHOT/firstshot",
                                "FIRST-SHOT PREVIEW saved", "H3Memory")
            if save_every_shot:
                # before the seam trim, so consecutive files overlap ~1s - a
                # chain that dies at the mux can still be joined by hand
                # KursatAs 2026-08-19 20:37: align per-shot debug videos with
                # the Multishot Advance output folder naming convention.
                _write_shot_mp4(imgs, wav, sr,
                                "video/multishot_advance_shots/shot",
                                f"shot {si + 1}/{n} saved", "H3Memory")

            h3_update_post_decode_state(
                state, imgs, wav, sr, continuity, _OV, _OV_HO, audio_lock,
                audio_vae, _aud_env)

            imgs, wav = h3_apply_join_handling(
                imgs, wav, si=si, continuity=continuity,
                cp_trim=state.cp_trim, ov=_OV, ov_ho=_OV_HO,
                frames_parts=frames_parts, audio_parts=audio_parts,
                stream_assembler=_stream_assembler, join_blend=join_blend,
                audio_lock=audio_lock, spine_present=_spine is not None,
                shot_seed=shot_seed, seed=seed, join_fx=join_fx, sr=sr)

            # fp16: the encoder quantises to uint8 downstream, and this
            # timeline is what exhausted host RAM at 6 shots x 243f.
            if _stream_assembler is not None:
                # streaming: the shot goes to lossless disk NOW and its RAM is
                # returned; only 1-D statistics stay behind.
                _stream_assembler.add(imgs.cpu().float())
            else:
                frames_parts.append(imgs.cpu().half())
            audio_parts.append((wav if wav.ndim == 3 else wav.unsqueeze(0)).cpu())
            _saved_prefix = _cache.save_prefix(
                si + 1,
                h3_build_sampler_cache_state(
                    bank, frames_parts, audio_parts, _lat_v_parts,
                    _lat_a_parts, voice_block, state))
            if _project_active and _saved_prefix:
                h3_advance_project_record_prefix(
                    project, si + 1, n, _cache.base_key,
                    _cache.prefix_key(si + 1), _saved_prefix)

        sr = state.sr
        return h3_finalize_sampler_outputs(
            frames_parts, audio_parts, _lat_v_parts, _lat_a_parts,
            stream_assembler=_stream_assembler, color_level=color_level,
            keyframe_images=keyframe_images, master_normalize=master_normalize,
            sr=sr, total_shots=n, cp_trim=state.cp_trim,
            api_prompt=_api_prompt, api_pnginfo=_api_pnginfo)

NODE_CLASS_MAPPINGS = {"H3MultishotMemorySampler": H3MultishotMemorySampler}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3MultishotMemorySampler": "H3 Multishot Sampler + Memory (long form)"}
