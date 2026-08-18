# -*- coding: utf-8 -*-
"""Final scene assembly and sampler output packaging."""

try:
    from .h3_audio import _xfade_audio
except Exception:
    from h3_audio import _xfade_audio

try:
    from .h3_color import _cc_apply_perframe, _cc_stats, _mn_normalize
except Exception:
    from h3_color import _cc_apply_perframe, _cc_stats, _mn_normalize


def _h3_apply_scene_color(frames_parts, color_level, keyframe_images):
    if color_level != "scene" or len(frames_parts) <= 1:
        return frames_parts

    import torch

    # SCENE-WIDE match: ONE reference for the whole piece, applied once at the
    # end. The per-shot mode matched each shot to a rolling "house" and still
    # left a hard step at every join. Driving every shot to a single scene-wide
    # median removes the step, because all shots are pulled to the same target.
    if keyframe_images is not None:
        scene_mu, _scene_cov = _cc_stats(keyframe_images)
        source = "boundary keyframes"
    else:
        scene_mu, _scene_cov = _cc_stats(torch.cat(
            [p[::max(1, p.shape[0] // 24)] for p in frames_parts],
            dim=0))
        source = "scene median"
    before = [float(_cc_stats(p)[0][0] / _cc_stats(p)[0][2]
                    .clamp_min(1e-6)) for p in frames_parts]
    for idx in range(len(frames_parts)):
        # PER-FRAME: a per-shot gain cannot fix a shot with a warm head and a
        # cool body - it scales the head too
        frames_parts[idx] = _cc_apply_perframe(frames_parts[idx], scene_mu)
    after = [float(_cc_stats(p)[0][0] / _cc_stats(p)[0][2]
                   .clamp_min(1e-6)) for p in frames_parts]
    print("[H3Memory] scene colour match (per-frame, target = %s) "
          "| warmth before " % source
          + "/".join("%.2f" % v for v in before) + "  after "
          + "/".join("%.2f" % v for v in after), flush=True)
    return frames_parts


def _h3_batch_latents(parts, what):
    import torch

    if not parts:
        return {"samples": torch.zeros(0)}
    shapes = {tuple(x.shape[1:]) for x in parts}
    if len(shapes) > 1:
        print(f"[H3Memory] {what}: shots do not share a grid "
              f"({sorted(shapes)}) - returning shot 1 only.",
              flush=True)
        return {"samples": parts[0]}
    return {"samples": torch.cat(parts, dim=0)}


def _h3_finalize_streaming_output(stream_assembler, master_normalize,
                                  waveform, sr, total_shots, lat_v, lat_a,
                                  cp_trim, api_prompt, api_pnginfo):
    import os
    import torch
    import folder_paths as folder_paths_module

    output_dir = os.path.join(folder_paths_module.get_output_directory(),
                              "video", "H3CHAIN_STREAM")
    os.makedirs(output_dir, exist_ok=True)
    index = 1
    while os.path.exists(os.path.join(output_dir, "master_%05d.mp4" % index)):
        index += 1
    master_path = os.path.join(output_dir, "master_%05d.mp4" % index)
    stream_assembler.finalize(master_path, master_normalize, waveform, sr,
                              prompt=api_prompt, extra_pnginfo=api_pnginfo)
    placeholder = torch.zeros((1, stream_assembler.shots[0]["h"],
                               stream_assembler.shots[0]["w"], 3),
                              dtype=torch.half)
    print(f"[H3Memory] done (streamed): {total_shots} shots -> {master_path}. "
          "master_frames is a 1-frame placeholder; wire master_path.",
          flush=True)
    return (placeholder, {"waveform": waveform, "sample_rate": sr},
            total_shots, lat_v, lat_a, int(cp_trim), master_path)


def _h3_assemble_master_frames(frames_parts):
    import torch

    # Assemble in place. torch.cat allocated a second full timeline while the
    # first was still alive - 2x peak, and a 33.8 GB contiguous request that a
    # 64 GB box cannot satisfy. Same bytes, one buffer.
    total_frames = sum(int(part.shape[0]) for part in frames_parts)
    master = torch.empty((total_frames,) + tuple(frames_parts[0].shape[1:]),
                         dtype=frames_parts[0].dtype)
    offset = 0
    for idx in range(len(frames_parts)):
        part = frames_parts[idx]
        master[offset:offset + part.shape[0]].copy_(part)
        offset += int(part.shape[0])
        frames_parts[idx] = None
        del part
    return master


def h3_finalize_sampler_outputs(frames_parts, audio_parts, lat_v_parts,
                                lat_a_parts, *, stream_assembler,
                                color_level, keyframe_images,
                                master_normalize, sr, total_shots, cp_trim,
                                api_prompt=None, api_pnginfo=None):
    frames_parts = _h3_apply_scene_color(
        frames_parts, color_level, keyframe_images)

    # issue #12: batch the raw per-shot latents along dim 0. Shapes match when
    # every shot shares one grid, which is the normal case; if a run ever mixes
    # grids the batch is impossible, and saying so beats returning something
    # that silently is not what it claims to be.
    lat_v = _h3_batch_latents(lat_v_parts, "video_latents")
    lat_a = _h3_batch_latents(lat_a_parts, "audio_latents")
    print("[H3Memory] latents out: video %s, audio %s, head_frames=%d "
          "(UNTRIMMED - shots 2+ open with the replayed head)"
          % (tuple(lat_v["samples"].shape),
             tuple(lat_a["samples"].shape) if lat_a_parts else "none",
             cp_trim), flush=True)

    # always the short 40ms weld: a long crossfade CONSUMES its overlap, which
    # shortened audio 375ms per join and walked lip sync off from shot 2 onward.
    waveform = _xfade_audio(audio_parts, sr, ms=40)

    if stream_assembler is not None:
        return _h3_finalize_streaming_output(
            stream_assembler, master_normalize, waveform, sr, total_shots,
            lat_v, lat_a, cp_trim, api_prompt, api_pnginfo)

    if master_normalize != "off":
        frames_parts, normalize_msg = _mn_normalize(
            frames_parts, master_normalize)
        if normalize_msg:
            print("[H3Memory] master normalize (%s): %s"
                  % (master_normalize, normalize_msg), flush=True)

    master = _h3_assemble_master_frames(frames_parts)
    print(f"[H3Memory] done: {total_shots} shots, {master.shape[0]} frames "
          f"(~{master.shape[0] / 24.0:.1f}s).", flush=True)
    return (master, {"waveform": waveform, "sample_rate": sr}, total_shots,
            lat_v, lat_a, int(cp_trim), "")
