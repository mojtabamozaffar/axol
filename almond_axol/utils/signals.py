"""Signal handling for spawned worker subprocesses.

A terminal Ctrl+C delivers SIGINT to the whole foreground process group, so
every spawned child (the IK solver, the video relay, the dataset recorder)
receives it at the same instant as the parent. Left to the default handler each
child raises ``KeyboardInterrupt`` and tears itself down independently — racing
with, and usually preempting, the parent's orchestrated shutdown.

That is exactly what broke the graceful return-to-zero on Ctrl+C: the IK worker
exited on its own SIGINT, so by the time the parent asked it to plan the park
trajectory the pipe was already dead (``[Errno 104] Connection reset by peer``)
and the arms dropped instead of easing down.

The fix is the standard multiprocessing pattern: worker children ignore SIGINT
and are driven solely by the parent over their control pipe — each already stops
on a ``None`` / ``("shutdown",)`` sentinel the parent sends during teardown, with
a ``terminate()`` (SIGTERM, *not* ignored) fallback. Ctrl+C then interrupts only
the parent, which owns the user-facing lifecycle and shuts the children down in
order.
"""

from __future__ import annotations

import signal


def ignore_sigint() -> None:
    """Ignore SIGINT in this subprocess so a group-wide Ctrl+C can't tear it down.

    Call once, first thing, in a spawned worker's entry point (on its main
    thread). The parent keeps the default handler and orchestrates shutdown via
    the worker's control pipe. Best-effort: silently no-ops where the platform or
    calling thread doesn't permit installing the handler (``signal.signal`` only
    works on the main thread), which is never fatal here.
    """
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (ValueError, OSError):
        pass
