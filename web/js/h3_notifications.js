// H3 Multishot Advance toast notifications.
//
// KursatAs 2026-08-17 10:02: backend H3 warnings/errors emit a websocket event;
// this frontend listener shows them in the ComfyUI browser instead of leaving
// them buried in the console.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const EVENT = "h3multishotadvance.alert";
const STYLE_ID = "h3-multishot-advance-toast-style";
const CONTAINER_ID = "h3-multishot-advance-toast-container";
const recent = new Map();

function ensureStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
        #${CONTAINER_ID} {
            position: fixed;
            top: 68px;
            right: 18px;
            z-index: 100000;
            display: flex;
            flex-direction: column;
            gap: 10px;
            width: min(440px, calc(100vw - 36px));
            pointer-events: none;
        }
        .h3-toast {
            pointer-events: auto;
            border: 1px solid var(--h3-toast-border, rgba(255,255,255,0.14));
            border-left-width: 5px;
            border-radius: 9px;
            background: var(--h3-toast-bg, rgba(23, 23, 28, 0.96));
            color: #f4f4f5;
            box-shadow: 0 12px 34px var(--h3-toast-shadow, rgba(0,0,0,0.38));
            padding: 12px 40px 12px 13px;
            font: 13px/1.42 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            position: relative;
            overflow-wrap: anywhere;
            animation: h3ToastIn 140ms ease-out;
        }
        /* KursatAs 2026-08-18 05:47: make progress toasts visibly blue.
           KursatAs 2026-08-18: warning-level H3 notices are not hard failures,
           so show them as dark amber/orange; keep real errors red. */
        .h3-toast-info {
            --h3-toast-bg: linear-gradient(135deg, rgba(2, 132, 199, 0.98), rgba(12, 74, 110, 0.98));
            --h3-toast-border: rgba(125, 211, 252, 0.36);
            --h3-toast-shadow: rgba(2, 132, 199, 0.34);
            border-left-color: #7dd3fc;
        }
        .h3-toast-warning {
            --h3-toast-bg: linear-gradient(135deg, rgba(180, 83, 9, 0.98), rgba(120, 53, 15, 0.98));
            --h3-toast-border: rgba(251, 191, 36, 0.42);
            --h3-toast-shadow: rgba(180, 83, 9, 0.34);
            border-left-color: #fbbf24;
        }
        .h3-toast-error {
            --h3-toast-bg: linear-gradient(135deg, rgba(153, 27, 27, 0.98), rgba(69, 10, 10, 0.98));
            --h3-toast-border: rgba(254, 202, 202, 0.46);
            --h3-toast-shadow: rgba(153, 27, 27, 0.42);
            border-left-color: #fee2e2;
        }
        .h3-toast-title {
            font-weight: 700;
            margin-bottom: 4px;
        }
        .h3-toast-message {
            white-space: pre-wrap;
            opacity: 0.95;
        }
        .h3-toast-close {
            position: absolute;
            top: 6px;
            right: 8px;
            border: 0;
            background: transparent;
            color: inherit;
            opacity: 0.72;
            font-size: 20px;
            line-height: 1;
            cursor: pointer;
        }
        .h3-toast-close:hover { opacity: 1; }
        @keyframes h3ToastIn {
            from { opacity: 0; transform: translateY(-6px); }
            to { opacity: 1; transform: translateY(0); }
        }
    `;
    document.head.appendChild(style);
}

function ensureContainer() {
    ensureStyle();
    let el = document.getElementById(CONTAINER_ID);
    if (!el) {
        el = document.createElement("div");
        el.id = CONTAINER_ID;
        document.body.appendChild(el);
    }
    return el;
}

function normalizeSeverity(value) {
    const severity = String(value || "info").toLowerCase();
    return ["info", "warning", "error"].includes(severity) ? severity : "info";
}

function showToast(detail) {
    const message = String(detail?.message || "").trim();
    if (!message) return;

    const severity = normalizeSeverity(detail?.severity);
    const title = String(detail?.title || "H3 Multishot");
    const key = `${severity}|${message}`;
    const now = Date.now();
    if ((recent.get(key) || 0) > now - 1200) return;
    recent.set(key, now);

    const toast = document.createElement("div");
    toast.className = `h3-toast h3-toast-${severity}`;

    const close = document.createElement("button");
    close.className = "h3-toast-close";
    close.type = "button";
    close.textContent = "×";
    close.title = "Dismiss";

    const heading = document.createElement("div");
    heading.className = "h3-toast-title";
    heading.textContent = title;

    const body = document.createElement("div");
    body.className = "h3-toast-message";
    body.textContent = message;

    toast.append(close, heading, body);
    ensureContainer().appendChild(toast);

    let done = false;
    const dismiss = () => {
        if (done) return;
        done = true;
        toast.remove();
    };
    close.addEventListener("click", dismiss);
    const ttl = Number(detail?.timeout_ms)
        || (severity === "error" ? 16000 : severity === "warning" ? 11000 : 7000);
    window.setTimeout(dismiss, ttl);
}

function isH3ExecutionError(detail) {
    const nodeType = String(detail?.node_type || "");
    const message = String(detail?.exception_message || "");
    const exceptionType = String(detail?.exception_type || "");
    return nodeType.startsWith("H3")
        || message.includes("[H3")
        || message.includes("H3 ")
        || exceptionType.includes("H3");
}

function showExecutionError(detail) {
    if (!isH3ExecutionError(detail)) return;
    const nodeType = String(detail?.node_type || "H3 node");
    const message = String(detail?.exception_message
        || detail?.exception_type
        || "Unknown H3 execution error");
    showToast({
        severity: "error",
        title: `${nodeType} failed`,
        message,
        timeout_ms: 16000,
    });
}

app.registerExtension({
    name: "h3.multishot.advance.notifications",

    setup() {
        api.addEventListener(EVENT, (event) => showToast(event.detail || {}));
        // KursatAs 2026-08-18 05:47: catch native ComfyUI execution failures
        // from H3 nodes too, so hard errors still become red browser toasts
        // even when a backend path forgot to emit h3multishotadvance.alert first.
        api.addEventListener("execution_error",
            (event) => showExecutionError(event.detail || {}));
    },
});
