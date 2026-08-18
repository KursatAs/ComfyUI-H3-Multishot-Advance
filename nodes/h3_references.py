# -*- coding: utf-8 -*-
"""Reference preparation helpers for H3 Multishot Advance."""


def _h3_build_voice_anchor(audio_vae, voice_ref):
    if voice_ref is None:
        return None

    waveform = voice_ref["waveform"]
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    waveform = waveform[:1]
    if waveform.shape[1] == 1:          # mono crashes the packed layout
        waveform = waveform.repeat(1, 2, 1)
    elif waveform.shape[1] > 2:
        waveform = waveform[:, :2]

    sample_rate = int(voice_ref["sample_rate"])
    vae_sample_rate = getattr(audio_vae, "audio_sample_rate", 32000)
    if sample_rate != vae_sample_rate:
        import torchaudio
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, vae_sample_rate)
        sample_rate = vae_sample_rate

    limit = 15 * sample_rate               # ref rows cost speed EVERY step
    if waveform.shape[-1] > limit:
        waveform = waveform[..., :limit]

    audio_latent = audio_vae.encode(waveform.movedim(1, -1))
    print(f"[H3Memory] voice anchor: {waveform.shape[-1] / sample_rate:.1f}s "
          f"ref audio rides in every shot as <Audio 1>.", flush=True)
    return {"kind": "audio", "ref_audio_t": audio_latent.shape[-1],
            "audio_latent": audio_latent}


def _h3_build_reference_image_slots(mmh3, video_vae, reference_images,
                                    reference_image_size, width, height):
    items, blocks = [], []
    if reference_images is None:
        return items, blocks

    import math as _math_ri
    for idx in range(reference_images.shape[0]):
        img = reference_images[idx:idx + 1]
        img_h, img_w = img.shape[1], img.shape[2]
        if reference_image_size == "match":
            scale = min(1.0, _math_ri.sqrt((width * height) / (img_w * img_h)))
        else:
            scale = min(1.0, mmh3.REF_IMAGE_SHORT_EDGE / min(img_w, img_h))
        target_w = max(mmh3.CANVAS_MULTIPLE,
                       round(img_w * scale / mmh3.CANVAS_MULTIPLE)
                       * mmh3.CANVAS_MULTIPLE)
        target_h = max(mmh3.CANVAS_MULTIPLE,
                       round(img_h * scale / mmh3.CANVAS_MULTIPLE)
                       * mmh3.CANVAS_MULTIPLE)
        resized = mmh3._resize(img, target_w, target_h, "disabled")
        items.append({"type": "image", "data": resized})
        blocks.append({"kind": "image",
                       "latent_h": target_h // 16,
                       "latent_w": target_w // 16,
                       "latent": video_vae.encode(resized)})

    print(f"[H3Memory] {len(blocks)} reference image(s) ride in every shot "
          f"as <Picture 1..{len(blocks)}>.", flush=True)
    return items, blocks
