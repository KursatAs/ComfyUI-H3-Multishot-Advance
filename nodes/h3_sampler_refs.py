# -*- coding: utf-8 -*-
"""Per-shot reference and keyframe preparation for H3 Multishot."""

try:
    from .h3_audio import _encode_ref_audio_compat
except Exception:
    from h3_audio import _encode_ref_audio_compat

try:
    from .h3_join import _jb_grid
except Exception:
    from h3_join import _jb_grid

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


def _h3_fail(message, exc_type=RuntimeError, title="H3 Multishot", tag=None):
    msg = str(message)
    _h3_error(msg, title=title, tag=tag)
    return exc_type(msg)


def h3_build_shot_refs(mmh3, video_vae, audio_vae, si, width, height,
                       anchor_frames, start_image, ref_image_items,
                       ref_image_blocks, voice_block, reference_video,
                       reference_video_audio, continuity, bank,
                       speech_active=True):
    ref_items, ref_blocks = [], []

    # operator-supplied refs go FIRST and never move: the bank grows from shot
    # to shot, so anything appended after it would change <Picture n> /
    # <Audio n> numbering mid-chain and break the prompt's bindings. Items and
    # blocks stay in the same sequence.
    for item, block in zip(ref_image_items, ref_image_blocks):
        ref_items.append(item)
        ref_blocks.append(block)
    # KursatAs 2026-08-19 04:45: voice identity remains stored in voice_block,
    # but it only enters conditioning for shots that explicitly speak.
    if voice_block is not None and speech_active:
        ref_items.append({"type": "audio"})
        ref_blocks.append(voice_block)

    # identity reference image(s): seed identity early, then let the bank carry
    # it afterwards.
    if start_image is not None and anchor_frames > 0:
        img = start_image[:1]
        ih, iw = int(img.shape[1]), int(img.shape[2])
        import math as _math
        sc = min(1.0, _math.sqrt((width * height) / max(iw * ih, 1)))
        tw = max(32, round(iw * sc / 32) * 32)
        th = max(32, round(ih * sc / 32) * 32)
        rz = mmh3._resize(img, tw, th, "disabled")
        ref_items.append({"type": "image", "data": rz})
        ref_blocks.append({"kind": "image", "latent_h": th // 16,
                           "latent_w": tw // 16,
                           "latent": video_vae.encode(rz)})

    # bank slots -> video_audio references, built the way core does.
    # first_frame mode runs on an fl2va checkpoint, which has no reference rows
    # - the hand-off frame carries continuity.
    #
    # 2.2.4: a user-supplied reference_video is prepended as just another
    # (frames, audio) pair, so it travels this exact proven path instead of a
    # parallel one. Speaking shots pair video references with audio references;
    # silent shots deliberately keep them video-only so old voice/audio cannot
    # trigger invented dialogue.
    extra_clips = []
    if reference_video is not None and int(reference_video.shape[0]):
        # KursatAs 2026-08-19 04:45: for silent shots, keep the visual
        # reference clip but withhold its audio so it cannot act as dialogue
        # conditioning.
        rv_audio = reference_video_audio if speech_active else None
        rv_n = int(reference_video.shape[0])
        if rv_audio is None and speech_active:
            import torch as _t
            sr = 32000
            dur = rv_n / float(mmh3.FPS)
            rv_audio = {
                "waveform": _t.zeros(1, 2, max(1, int(dur * sr))),
                "sample_rate": sr,
            }
            if si == 1:
                print("[H3Memory] reference_video: %d frame(s), no audio "
                      "supplied - pairing with silence" % rv_n, flush=True)
        elif rv_audio is not None and si == 1:
            print("[H3Memory] reference_video: %d frame(s) + audio" % rv_n,
                  flush=True)
        elif not speech_active and reference_video_audio is not None and si == 1:
            print("[H3Memory] reference_video: %d frame(s), audio withheld "
                  "by speech guard for this silent shot" % rv_n, flush=True)
        if si == 1 and rv_n > 64:
            print("[H3Memory] reference_video is %d frames; it is "
                  "subsampled to 2 fps but that is still a lot of reference "
                  "tokens on EVERY step. Trim the clip before wiring it."
                  % rv_n, flush=True)
        extra_clips.append((reference_video, rv_audio))

    for clip_frames, clip_audio in (
            extra_clips
            + ([] if continuity == "first_frame" else bank.frames())):
        vh, vw = int(clip_frames.shape[1]), int(clip_frames.shape[2])
        cw, ch = mmh3.adapt_canvas(vw, vh)
        if vw * vh < cw * ch:
            cw = max(32, round(vw / 32) * 32)
            ch = max(32, round(vh / 32) * 32)
        fr = mmh3._resize(clip_frames, cw, ch, "disabled")
        fr = fr[:_jb_grid(fr.shape[0])]
        z = video_vae.encode(fr)
        a_lat, a_t = (None, 0)
        # KursatAs 2026-08-19 04:45: bank clips become video_audio only for
        # speaking shots. Silent shots use kind=video to preserve continuity
        # without injecting previous speech.
        if speech_active and clip_audio is not None:
            # KursatAs 2026-08-15 18:37: route through ref-audio compat.
            a_lat, a_t = _encode_ref_audio_compat(mmh3, audio_vae, clip_audio)
            # the soundtrack takes its own <Audio j>, emitted before <Video k>
            ref_items.append({"type": "audio"})
        idx = list(range(0, fr.shape[0], mmh3.FPS // 2))
        ref_items.append({"type": "video", "data": fr[idx],
                          "timestamps": [i / 2.0 for i in range(len(idx))]})
        ref_blocks.append({"kind": "video_audio" if a_t else "video",
                           "latent_t": z.shape[2],
                           "latent_h": ch // 16, "latent_w": cw // 16,
                           "ref_audio_t": a_t, "latent": z,
                           "audio_latent": a_lat})

    return ref_items, ref_blocks


def _load_interior_keyframe_patch():
    try:
        from .h3_interior_patch import (_motion_context_status,
                                        ensure_interior_keyframes)
        return _motion_context_status, ensure_interior_keyframes
    except ImportError:
        try:
            from h3_interior_patch import (_motion_context_status,
                                           ensure_interior_keyframes)
            return _motion_context_status, ensure_interior_keyframes
        except ImportError:
            # loose install: the module sits beside this file but is not
            # importable by name.
            import importlib.util as _ilu
            import os as _os
            path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                 "h3_interior_patch.py")
            spec = _ilu.spec_from_file_location("h3_interior_patch", path)
            mod = _ilu.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod._motion_context_status, mod.ensure_interior_keyframes


def h3_build_shot_keyframes(mmh3, si, prompt, continuity, keyframe_images,
                            last_tail, width, height, frame_count,
                            frames_per_shot, tail_k, handoff_depth,
                            end_anchor, house_frame, dbg_pins):
    kf_vision = []     # first_frame mode: images -> vision tokens
    keyframes = []

    if continuity == "flf_chain" and keyframe_images is None:
        # a silent no-op here renders a full unanchored chain and the operator
        # finds out hours later - fail loudly instead.
        msg = (
            "continuity=flf_chain but keyframe_images is empty. Wire N+1 "
            "boundary plates (and enable their gate) for N shots, or switch "
            "continuity to context_pin.")
        raise _h3_fail(msg, ValueError, "H3 FLF chain config",
                       tag="H3Memory")

    if continuity == "flf_chain" and keyframe_images is not None:
        # TRUE FFLF: shot i runs between boundary image i and i+1.
        # The join is ONE shared picture used as the end of one shot and the
        # start of the next, so there is nothing to drift and nothing to
        # colour-correct at the boundary.
        n_kf = keyframe_images.shape[0]
        a = keyframe_images[min(si, n_kf - 1):min(si, n_kf - 1) + 1]
        kf_a = mmh3._resize(a, width, height, "disabled")
        kf_vision.append(kf_a)
        keyframes.append({"resolved_frame_index": 0, "image": kf_a})
        if si + 1 < n_kf:
            b = keyframe_images[si + 1:si + 2]
            kf_b = mmh3._resize(b, width, height, "disabled")
            kf_vision.append(kf_b)
            keyframes.append({"resolved_frame_index": frame_count - 1,
                              "image": kf_b})
            # the documented FL2VA alignment instruction, first line
            prompt = (
                "How the reference pictures align with the target video - "
                "Picture 1 (from Shot 1) aligns with the 0.00-second mark of "
                "the target video; Picture 2 (from Shot 1) aligns with the "
                "%.2f-second mark of the target video.\n\n"
                % (frame_count / 24.0)
            ) + prompt
        print("[H3Memory] FFLF shot %d: pinned between boundary keyframes %d "
              "and %d" % (si + 1, si, min(si + 1, n_kf - 1)), flush=True)
    elif last_tail is not None and continuity == "first_frame":
        # the model's own hand-off: the previous last frame goes in BOTH ways
        # the stock Image-to-Video node sends it - vision tokens through the
        # text encoder AND the frame-0 keyframe latent. A keyframe latent alone
        # is a weak hint; the vision path is the conditioning fl2va was trained
        # on.
        kf_img = mmh3._resize(last_tail[-1:], width, height, "disabled")
        kf_vision.append(kf_img)
        keyframes.append({"resolved_frame_index": 0, "image": kf_img})
    elif last_tail is not None and continuity == "seamless":
        kf_img = mmh3._resize(last_tail[-1:], width, height, "disabled")
        keyframes.append({"resolved_frame_index": 0, "image": kf_img})
    elif last_tail is not None and continuity == "seamless_tail":
        # tail bracket: previous pixel frames -9/-5/-1 pinned at keyframe
        # indices 0/4/8 (one per latent block on the 4x temporal grid). The
        # join is over-determined: position, exposure and velocity are all
        # specified by real frames.
        #
        # Indices 4 and 8 are INTERIOR anchors, which stock comfy rejects. Our
        # layout patch generalises the math.
        #
        # KursatAs 2026-08-17 09:54: only an ACTIVE Motion-Context layout patch
        # owns this site. A registered node alone should not block
        # seamless_tail; a live MC layout patch should, since it only serves
        # rows carrying its own marker and these raw keyframes would fall
        # through to stock mid-render.
        motion_context_status, ensure_interior_keyframes = (
            _load_interior_keyframe_patch())
        mc_status = motion_context_status()
        mc_pack = mc_status["layout"]
        if mc_pack:
            msg = (
                "continuity=seamless_tail needs interior keyframe anchors, "
                f"and {mc_pack} owns that patch site but only serves its own "
                "nodes - the chain would crash mid-render. Use "
                "continuity=context_pin (better, and it is what that pack is "
                "for), or first_frame, or remove that pack to use "
                "seamless_tail.")
            _h3_error(msg, topic="seamless_tail_conflict", tag="H3Memory")
            raise ValueError(msg)
        if si == 1 and mc_status["registered"]:
            # KursatAs 2026-08-17 09:54: a registered Motion-Context node is
            # not enough to block seamless_tail; only an active layout patch
            # owns the site.
            msg = ("seamless_tail: Motion-Context node is registered but its "
                   "layout patch is not active; using H3's "
                   "interior-keyframe patch.")
            print("[H3Memory] " + msg, flush=True)
            _h3_info(msg, topic="seamless_tail", tag="H3Memory",
                     timeout_ms=7000)
        ik_ok, ik_msg = ensure_interior_keyframes(verbose=False)
        if not ik_ok:
            msg = (
                "continuity=seamless_tail needs interior keyframe anchors "
                f"and the layout patch failed: {ik_msg}. Use "
                "continuity=first_frame or context_pin instead.")
            _h3_error(msg, topic="seamless_tail_unavailable", tag="H3Memory")
            raise ValueError(msg)
        for j in range(tail_k + 1):
            pi = -(1 + 4 * (tail_k - j))          # -9, -5, -1
            src = last_tail[pi:pi + 1] if pi != -1 else last_tail[-1:]
            kf_img = mmh3._resize(src, width, height, "disabled")
            dbg_pins.append((4 * j, kf_img[0].detach().cpu().clone()))
            keyframes.append({"resolved_frame_index": 4 * j,
                              "image": kf_img})

    if (last_tail is not None and continuity == "latent_handoff"
            and handoff_depth == "bootstrap"):
        # bootstrap depth: the 2-row latent lock is a weak video anchor - back
        # it with a frame-0 keyframe pin of the previous last frame (soft, but
        # end_anchor and the bank carry the rest)
        kf_img = mmh3._resize(last_tail[-1:], width, height, "disabled")
        keyframes.append({"resolved_frame_index": 0, "image": kf_img})

    if (end_anchor and continuity == "first_frame" and si > 0
            and house_frame is not None):
        # fl2va reads first+last as "travel from A to B" and invents a camera
        # move to fill the middle (render-verified: shot 2 pushed into an
        # extreme close-up and back out). Hand it ONLY the first frame and let
        # it continue.
        if si == 1:
            msg = ("end_anchor ignored in first_frame mode: a last-frame pin "
                   "makes fl2va plan a camera MOVE between the two frames. "
                   "Control drift with prompt wording instead.")
            print("[H3Memory] " + msg, flush=True)
            _h3_info(msg, topic="end_anchor", tag="H3Memory")
    elif end_anchor and house_frame is not None and si > 0:
        # return-to-house DOUBLE pin at the shot's tail: closes the compounding
        # push-in creep so the next join inherits a tail the text agrees with.
        # One pin at the last frame gets outvoted by committed motion
        # (render-verified: a tail lean-in ran straight through it); a second
        # pin half a second earlier makes the hold bracket-strength and reads
        # as her settling for the beat. Rides through the same encode loop
        # below, so join_anchor_noise applies too.
        kf_img = mmh3._resize(house_frame, width, height, "disabled")
        keyframes.append({"resolved_frame_index": frames_per_shot - 1,
                          "image": kf_img})
        if frames_per_shot > 21:
            keyframes.append({"resolved_frame_index": frames_per_shot - 13,
                              "image": kf_img})

    return prompt, kf_vision, keyframes
