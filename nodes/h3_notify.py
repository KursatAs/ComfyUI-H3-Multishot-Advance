"""Small ComfyUI browser notification bridge for H3 Advance nodes."""

EVENT_NAME = "h3multishotadvance.alert"
DEFAULT_TITLE = "H3 Multishot Advance"

_TOPIC_TITLES = {
    "checkpoint": "H3 mode/checkpoint notice",
    "context_pin": "H3 context_pin missing",
    "continuity": "H3 continuity risk",
    "end_anchor": "H3 end_anchor note",
    "experimental": "H3 experimental setting",
    "first_frame": "H3 first_frame risk",
    "fl2va_quality": "H3 fl2va quality path",
    "incompatible": "H3 incompatible settings",
    "memory_sampler": "H3 Memory Sampler",
    "ram": "H3 RAM risk",
    "seamless_tail": "H3 seamless_tail",
    "seamless_tail_conflict": "H3 seamless_tail conflict",
    "seamless_tail_unavailable": "H3 seamless_tail unavailable",
    "shot_cache": "H3 shot cache",
    "speed_booster": "H3 optional speed booster",
    "speed_quality": "H3 speed/quality risk",
    "start_image": "H3 start_image note",
}


def _h3_title(title=None, topic=None):
    if title:
        return str(title)
    key = str(topic or "").strip().lower()
    return _TOPIC_TITLES.get(key, DEFAULT_TITLE)


def h3_notify(message, severity="warning", title=DEFAULT_TITLE,
              tag=None, timeout_ms=None):
    """Send a best-effort toast event to the ComfyUI browser.

    KursatAs 2026-08-17 10:02: console-only warnings are easy to miss during
    long H3 renders, so critical warnings/errors also emit a frontend event.
    """
    try:
        msg = str(message or "").strip()
        if not msg:
            return False
        payload = {
            "severity": str(severity or "info").lower(),
            "title": str(title or DEFAULT_TITLE),
            "message": msg[:1800],
        }
        if tag is not None:
            payload["tag"] = str(tag)
        if timeout_ms is not None:
            payload["timeout_ms"] = int(timeout_ms)

        from server import PromptServer
        ps = getattr(PromptServer, "instance", None)
        if ps is None:
            return False
        ps.send_sync(EVENT_NAME, payload)
        return True
    except Exception:
        return False


def h3_notice(message, severity="info", title=None, topic=None, tag=None,
              timeout_ms=None):
    """Send a classified H3 toast.

    Keep severity/title policy in one place so call sites choose the intent
    (info/warning/error + topic), not presentation details.
    """
    return h3_notify(message, severity, _h3_title(title, topic),
                     tag=tag, timeout_ms=timeout_ms)


def h3_info(message, title=None, topic=None, tag=None, timeout_ms=None):
    return h3_notice(message, "info", title=title, topic=topic, tag=tag,
                     timeout_ms=timeout_ms)


def h3_warning(message, title=None, topic=None, tag=None, timeout_ms=None):
    return h3_notice(message, "warning", title=title, topic=topic, tag=tag,
                     timeout_ms=timeout_ms)


def h3_error(message, title=None, topic=None, tag=None, timeout_ms=None):
    return h3_notice(message, "error", title=title, topic=topic, tag=tag,
                     timeout_ms=timeout_ms)
