# -*- coding: utf-8 -*-
"""VRAM/model residency helpers for the H3 Multishot sampler."""


def h3_evict_dit_before_text_encoder(si):
    # Symmetric eviction: free the DiT before the encoder loads.
    # Without this the previous shot's DiT is still resident when a 15.69 GB
    # encoder is requested, and on a 24 GB card it cannot fit beside it -
    # ComfyUI streams the encoder off disk and the reader eventually fails
    # (hostbuf_file_reader_read). Costs nothing: the DiT is reloaded every shot
    # regardless.
    if si <= 0:
        return
    try:
        import comfy.model_management as model_management
        device = model_management.get_torch_device()
        before = model_management.get_free_memory(device) / (1024 ** 3)
        model_management.free_memory(
            model_management.get_total_memory(device) * 0.9, device)
        model_management.soft_empty_cache()
        after = model_management.get_free_memory(device) / (1024 ** 3)
        print("[H3Memory] DiT evicted before the text encoder; %.1f -> %.1f "
              "GB free" % (before, after), flush=True)
    except Exception as err:
        print("[H3Memory] could not evict before the encoder (%s) - if the "
              "encoder streams from disk this is why" % err, flush=True)


def h3_prepare_sampling_memory(si, clip, model, video_vae, audio_vae,
                               model_management):
    # issue #8: separate TE device -> nothing to reclaim, keep it hot
    te_dev = getattr(clip.patcher, "load_device", None)
    dit_dev = getattr(model, "load_device", None)
    if (te_dev is not None and dit_dev is not None
            and str(te_dev) != str(dit_dev)):
        if si == 0:
            print(f"[H3Memory] TE on {te_dev}, DiT on {dit_dev} - separate "
                  f"devices, TE stays resident.", flush=True)
        return

    try:
        clip.patcher.model.to(model_management.text_encoder_offload_device())
    except Exception:
        pass

    # The VAEs are dead weight during sampling - encode already happened,
    # decode has not. They were staying resident (5.5 GB between them) while
    # the DiT loaded, and on shot 2+ the larger conditioning payload raises the
    # activation reserve enough that the DiT then misses a FULL load by a few
    # hundred MB. Measured: shot 1 full load at 18.5 s/it, shot 2 with 399 MB
    # offloaded at 267 s/it - a 14x collapse for 2% of the weights, because
    # every offloaded layer streams over PCIe every step.
    vaes_freed = 0
    for vae in (video_vae, audio_vae):
        if getattr(vae, "patcher", None) is not None:
            vaes_freed += 1

    try:
        device = model_management.get_torch_device()
        # Unload through model management, NOT module.to(). On the DynamicVRAM
        # path (0.33+) weights live in a comfy_aimdo vbar arena; a bare
        # .to(cpu) moves the module and leaves the arena's pages resident
        # (~8 GB observed on a 24 GB card), after which free_memory reports
        # "0 models unloaded" because nothing LOOKS loaded any more. Proper
        # unload goes through the patcher and tears the arena down. The DiT is
        # not loaded yet, and TE/VAEs are re-requested every shot anyway, so
        # this costs nothing extra.
        try:
            model_management.unload_all_models()
        except Exception as unload_err:
            print("[H3Memory] unload_all_models failed (%s) - falling back to "
                  "module moves" % unload_err, flush=True)
            try:
                clip.patcher.model.to(
                    model_management.text_encoder_offload_device())
            except Exception:
                pass
            for vae in (video_vae, audio_vae):
                try:
                    vae.patcher.model.to(
                        model_management.vae_offload_device())
                except Exception:
                    pass

        model_management.free_memory(
            model_management.get_total_memory(device) * 0.9, device)
        model_management.soft_empty_cache()

        # Name whoever is still holding the card. This is the instrument that
        # settles the "mystery 8 GB": if the vbar theory is right this prints
        # nothing on 0.33.1 any more; if it is wrong, the culprit is named
        # instead of guessed.
        try:
            import torch
            free_bytes, total_bytes = torch.cuda.mem_get_info(
                device.index if hasattr(device, "index") else 0)
            if (total_bytes - free_bytes) > total_bytes * 0.25:
                print("[H3Memory] post-evict residents: driver %.1f GB held "
                      "| torch alloc %.1f reserved %.1f"
                      % ((total_bytes - free_bytes) / 2**30,
                         torch.cuda.memory_allocated(device) / 2**30,
                         torch.cuda.memory_reserved(device) / 2**30),
                      flush=True)
                for loaded_model in list(
                        getattr(model_management, "current_loaded_models",
                                [])):
                    try:
                        size = loaded_model.model.loaded_size()
                        if size > 256 * 1024**2:
                            print("[H3Memory]   resident: %s  %.2f GB"
                                  % (loaded_model.model.model.__class__.__name__,
                                     size / 2**30), flush=True)
                    except Exception:
                        pass
        except Exception:
            pass

        print("[H3Memory] TE%s evicted; %.1f GB free for the DiT"
              % (" + %d VAE(s)" % vaes_freed if vaes_freed else "",
                 model_management.get_free_memory(device) / (1024 ** 3)),
              flush=True)
    except Exception:
        pass
