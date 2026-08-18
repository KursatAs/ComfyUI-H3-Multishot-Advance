"""Small ComfyUI browser notification bridge for H3 Advance nodes."""

EVENT_NAME = "h3multishotadvance.alert"


def h3_notify(message, severity="warning", title="H3 Multishot Advance",
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
            "title": str(title or "H3 Multishot Advance"),
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
