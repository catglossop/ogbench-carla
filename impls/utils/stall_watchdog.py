"""Detect a CARLA run that has stopped ticking the simulator, and take it down.

Ported from ``run_leaderboard.py``, which supervises each route from *outside* the worker:
it counts ``[RC-PID]`` tick lines on the worker's stdout, kills a worker whose log has gone
silent (``--route-timeout`` / ``leaderboard_runs/watchdog_b2d_6k.sh``), and re-queues the
route (``--retries``). ``main_carla.py`` has no supervisor around its env loop, and CARLA's
client timeout is 7200 s (``impls/configs/carla_config.yaml``), so a wedged ``world.tick()``
holds both GPUs for hours *without ever crashing* -- which is exactly what ``run_carla.sh``'s
crash-retry loop keys on (exit >= 128). Those runs just sit there.

Same idea, in-process:

* every ``env.step()`` / ``env.reset()`` that **returns** is a tick (:meth:`StallWatchdog.beat`);
* a daemon thread watches how long ago the last tick was;
* a stall is declared only when the process is *also* idle (``cpu_idle_frac``) -- so the
  17-minute Pi0-CoT JIT compile, which pins a core and emits no ticks, is never mistaken
  for a wedged simulator. This mirrors ``watchdog_b2d_6k.sh``'s deliberately conservative
  two-condition rule (log stale AND rpc port gone);
* on a stall it dumps every thread's traceback (so the next one is diagnosable), kills the
  CARLA server + Xvfb this run owns -- they are ``setsid``-ed children and would otherwise
  outlive us, holding ~7 GB of VRAM -- and ``os._exit``s with :data:`EXIT_STALLED`, which
  ``run_carla.sh`` relaunches under its own ``--max-stall-retries`` budget.

Known long no-tick sections (a blocking VLM window review, a checkpoint save) wrap
themselves in :func:`paused` so they never trip the check.

The thread can only run if the stalled thread released the GIL. CARLA's PythonAPI does drop
it around blocking RPCs, so this covers the wedged-``tick()`` case, but a stall that holds
the GIL would freeze the watchdog too. For that, each poll rewrites
``$OGBENCH_HEARTBEAT_FILE`` -- "the watchdog is alive and has not declared a stall" -- and a
detached shell nanny SIGKILLs the process if that file goes stale for twice the timeout.
Same "watch it from outside" structure as ``leaderboard_runs/watchdog_b2d_6k.sh``, and it
stays quiet through compiles and paused sections because the thread keeps writing through
them.
"""

from __future__ import annotations

import contextlib
import faulthandler
import functools
import os
import resource
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Iterator, Optional

# Exit code for "the simulator stopped ticking". Deliberately < 128 so that only launchers
# that know about it (run_carla.sh) retry: a bare `python impls/main_carla.py` exits instead
# of being restarted by something that thinks it saw a native crash.
EXIT_STALLED = 87

# Env vars run_carla.sh sets: the tick heartbeat the external nanny watches, and the file
# that tells the launcher a dead run died of a stall (so it uses the stall retry budget
# rather than the crash one -- a SIGKILL from the nanny is otherwise just exit 137).
HEARTBEAT_ENV = "OGBENCH_HEARTBEAT_FILE"
STALL_MARKER_ENV = "OGBENCH_STALL_MARKER"


_ACTIVE: Optional["StallWatchdog"] = None


def active() -> Optional["StallWatchdog"]:
    """The watchdog installed by :func:`install`, if any."""
    return _ACTIVE


def disarm() -> None:
    """Stop the installed watchdog. Call before teardown.

    Teardown -- ``env.close()``, then ``wandb.finish()`` uploading a run's rollout videos --
    is minutes of tick-free, network-bound waiting on a run that has already done its job.
    Killing there would relaunch a *finished* run from step 0, the same trap
    ``main_carla._mark_run_complete`` exists to avoid.
    """
    watchdog = _ACTIVE
    if watchdog is not None:
        watchdog.stop()


@contextlib.contextmanager
def paused(reason: str) -> Iterator[None]:
    """Hold the stall check across a section that legitimately produces no ticks.

    No-op when no watchdog is installed, so call sites don't need to care.
    """
    watchdog = _ACTIVE
    if watchdog is None:
        yield
        return
    with watchdog.paused(reason):
        yield


def _process_start_ticks() -> str:
    """This process's start time in clock ticks (``/proc/self/stat`` field 22), as a string.

    Identifies the process beyond its pid. The comm field can contain spaces and parens, so
    everything up to the final ``')'`` is dropped first -- the same split the nanny does.
    """
    try:
        with open("/proc/self/stat") as f:
            return f.read().rsplit(") ", 1)[1].split()[19]
    except (OSError, IndexError):
        return ""


def _process_cpu_seconds() -> float:
    """User+sys CPU seconds for this process (all threads). Stdlib only, no psutil."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return float(usage.ru_utime + usage.ru_stime)


class StallWatchdog:
    """Kill the process when the simulator has not ticked for ``timeout_s`` while idle."""

    def __init__(
        self,
        *,
        timeout_s: float,
        poll_s: float = 15.0,
        cpu_idle_frac: float = 0.05,
        rpc_port: Optional[int] = None,
        display_num: Optional[int] = None,
        resolve_ports: Optional[Callable[[], tuple[Optional[int], Optional[int]]]] = None,
        extra_cleanup: Optional[Callable[[], None]] = None,
        heartbeat_path: Optional[str] = None,
    ) -> None:
        self.timeout_s = float(timeout_s)
        self.poll_s = max(1.0, min(float(poll_s), self.timeout_s / 4.0))
        # Fraction of ONE core, averaged over a poll window, above which "no ticks" is read as
        # "busy doing something else" (XLA compile, OpenPI restore, buffer save) rather than a
        # stall. Negative disables the idle condition -- no-ticks alone then fires.
        self.cpu_idle_frac = float(cpu_idle_frac)
        self._rpc_port = rpc_port
        self._display_num = display_num
        self._resolve_ports = resolve_ports
        self._extra_cleanup = extra_cleanup
        self._heartbeat_path = heartbeat_path or os.environ.get(HEARTBEAT_ENV) or None
        self._marker_path = os.environ.get(STALL_MARKER_ENV) or None
        self._nanny: Optional[subprocess.Popen] = None

        self._lock = threading.Lock()
        self._last_beat = time.monotonic()
        self._ticks = 0
        self._armed = False
        self._pause_depth = 0
        self._pause_reason = ""
        self._busy_warned = False
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ #
    # Heartbeat                                                          #
    # ------------------------------------------------------------------ #

    def beat(self) -> None:
        """Record one simulator tick. Arms the watchdog on the first call."""
        now = time.monotonic()
        with self._lock:
            self._ticks += 1
            self._last_beat = now
            self._armed = True
            self._busy_warned = False

    def _touch_heartbeat(self, state: str) -> None:
        """Tell the external nanny this watchdog is alive and has *not* declared a stall.

        Written once per poll from the watcher thread rather than from :meth:`beat` (which
        runs at 20 Hz), so "heartbeat stale" means "the in-process watchdog is not running" --
        a wedged GIL, or a thread that died -- and never "a section that legitimately takes
        longer than the timeout", which the thread itself already excuses.
        """
        path = self._heartbeat_path
        if not path:
            return
        try:
            with open(path, "w") as f:
                f.write(f"{time.time():.3f} ticks={self._ticks} state={state}\n")
        except OSError:
            pass  # bookkeeping must never take down a healthy run

    @contextlib.contextmanager
    def paused(self, reason: str) -> Iterator[None]:
        """Suspend the check across a blocking, tick-free section (VLM call, checkpoint save)."""
        with self._lock:
            self._pause_depth += 1
            self._pause_reason = reason
        try:
            yield
        finally:
            with self._lock:
                self._pause_depth = max(0, self._pause_depth - 1)
                # Restart the window on resume: time spent inside the pause must not count
                # toward the next stall, or a 9-minute Gemini review leaves a 1-minute budget.
                self._last_beat = time.monotonic()

    # ------------------------------------------------------------------ #
    # Thread                                                             #
    # ------------------------------------------------------------------ #

    def start(self) -> "StallWatchdog":
        if self._thread is not None:
            return self
        # Write it once up front: the nanny ignores a missing heartbeat file (it has no way to
        # tell "not started yet" from "never will"), so a process that wedges before the first
        # poll would otherwise be invisible to it.
        self._touch_heartbeat("starting")
        self._thread = threading.Thread(target=self._run, name="stall-watchdog", daemon=True)
        self._thread.start()
        print(
            f"[stall-watchdog] armed: no CARLA tick for {self.timeout_s:.0f}s while idle "
            f"(<{self.cpu_idle_frac:.0%} of a core) -> exit {EXIT_STALLED} for the supervisor to retry.",
            flush=True,
        )
        self._spawn_nanny()
        return self

    def stop(self) -> None:
        self._stop_evt.set()
        nanny, self._nanny = self._nanny, None
        if nanny is not None and nanny.poll() is None:
            try:
                nanny.kill()
            except Exception:
                pass

    def _spawn_nanny(self) -> None:
        """Detached shell loop that SIGKILLs us if the tick heartbeat file goes stale.

        The in-process thread above cannot fire if whatever is stuck never releases the GIL.
        This is the outside view -- the same shape as ``leaderboard_runs/watchdog_b2d_6k.sh``,
        which kills a worker whose log has gone silent. It waits twice as long as the thread,
        so the thread gets first crack (better diagnostics, scoped CARLA cleanup); it does
        nothing until the first tick has been written, so model restore and JIT compile are
        never its business.
        """
        if not self._heartbeat_path:
            return
        deadline = int(self.timeout_s * 2)
        poll = max(15, int(self.poll_s))
        marker = self._marker_path or ""
        # Kill by (pid, start time), never by pid alone: pids are recycled, and the very next
        # thing to run after a stalled worker dies is run_carla.sh's retry of the same worker.
        # A nanny that outlived its process by one poll would SIGKILL the replacement.
        script = f"""
        pid={os.getpid()}
        start={_process_start_ticks()!r}
        hb={self._heartbeat_path!r}
        marker={marker!r}
        proc_start() {{ awk -F') ' '{{print $2}}' "/proc/$1/stat" 2>/dev/null | awk '{{print $20}}'; }}
        while :; do
          sleep {poll}
          [ "$(proc_start "$pid")" = "$start" ] || exit 0   # exited, or the pid was recycled
          mtime=$(stat -c %Y "$hb" 2>/dev/null) || continue
          age=$(( $(date +%s) - mtime ))
          [ "$age" -gt {deadline} ] || continue
          echo "[stall-watchdog/nanny] in-process watchdog silent for ${{age}}s (>{deadline}s): \
the run is wedged badly enough that it cannot kill itself. SIGKILLing $pid." >&2
          [ -n "$marker" ] && echo "nanny: in-process watchdog silent for ${{age}}s" > "$marker"
          kill -9 "$pid" 2>/dev/null
          exit 0
        done
        """
        try:
            self._nanny = subprocess.Popen(
                ["bash", "-c", script],
                start_new_session=True,  # outlives our process group; exits once we are gone
                stdin=subprocess.DEVNULL,
            )
            print(
                f"[stall-watchdog] external nanny pid={self._nanny.pid} watching "
                f"{self._heartbeat_path} (SIGKILL if this watchdog goes silent for {deadline}s).",
                flush=True,
            )
        except Exception as exc:  # a missing bash must not stop the run
            print(f"[stall-watchdog] could not start the external nanny: {exc}", flush=True)

    def _write_marker(self, detail: str) -> None:
        if not self._marker_path:
            return
        try:
            with open(self._marker_path, "w") as f:
                f.write(f"{detail}\n")
        except OSError:
            pass

    def _run(self) -> None:
        cpu_prev = _process_cpu_seconds()
        t_prev = time.monotonic()
        while not self._stop_evt.wait(self.poll_s):
            now = time.monotonic()
            cpu_now = _process_cpu_seconds()
            elapsed = max(now - t_prev, 1e-6)
            cpu_frac = (cpu_now - cpu_prev) / elapsed
            cpu_prev, t_prev = cpu_now, now

            with self._lock:
                armed = self._armed
                is_paused = self._pause_depth > 0
                reason = self._pause_reason
                age = now - self._last_beat
                ticks = self._ticks
            if not armed:
                self._touch_heartbeat("starting")
                continue
            if is_paused:
                self._touch_heartbeat(f"paused:{reason}")
                continue
            if age < self.timeout_s:
                self._touch_heartbeat("ticking")
                continue
            if self.cpu_idle_frac >= 0.0 and cpu_frac > self.cpu_idle_frac:
                self._touch_heartbeat(f"busy:{cpu_frac:.2f}cores")
                # No ticks, but the process is burning CPU: a JAX/XLA compile, an OpenPI
                # restore, a big buffer save. Not a wedged simulator -- say so once and wait.
                if not self._busy_warned:
                    self._busy_warned = True
                    print(
                        f"[stall-watchdog] no CARLA tick for {age:.0f}s but the process is busy "
                        f"({cpu_frac:.0%} of a core){f' [{reason}]' if reason else ''}; not a stall, still watching.",
                        flush=True,
                    )
                continue
            self._fire(age=age, ticks=ticks, cpu_frac=cpu_frac)

    # ------------------------------------------------------------------ #
    # Stall                                                              #
    # ------------------------------------------------------------------ #

    def _fire(self, *, age: float, ticks: int, cpu_frac: float) -> None:
        banner = (
            f"\n[stall-watchdog] STALLED: no CARLA tick in {age:.0f}s "
            f"(threshold {self.timeout_s:.0f}s, {ticks} ticks so far, "
            f"{cpu_frac:.1%} of a core -- the process is idle, not working).\n"
            f"[stall-watchdog] Dumping all thread stacks, killing this run's CARLA/Xvfb, "
            f"then exiting {EXIT_STALLED} so the launcher can retry.\n"
        )
        # Banner on stderr so it sits directly above the traceback dump; one line on stdout for
        # launchers (and tail -f) that only follow the run's stdout.
        print(f"[stall-watchdog] STALLED after {age:.0f}s without a tick; see stderr.", flush=True)
        sys.stdout.flush()
        sys.stderr.write(banner)
        sys.stderr.flush()
        try:
            # Where it actually hung. The single most useful artifact this produces -- without
            # it a stall leaves nothing behind but a silent log.
            faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
            sys.stderr.flush()
        except Exception:
            pass
        self._write_marker(f"watchdog age={age:.0f}s ticks={ticks} cpu_frac={cpu_frac:.3f}")
        self._kill_carla()
        self.stop()  # the nanny's job is done; don't leave it watching a recycled pid
        # os._exit, not sys.exit: the main thread is wedged, so no exception we raise here
        # would ever unwind it, and atexit/env.close() would block on the same dead RPC.
        os._exit(EXIT_STALLED)

    def _kill_carla(self) -> None:
        """Scoped kill of the CARLA server + Xvfb this run owns (``run_leaderboard.cleanup_slot_carla``).

        They are launched with ``setsid`` (``carla_utils._child_process_setup``), so they survive
        our exit and keep the GPU busy. Both patterns are port/display-specific, so a sibling
        ``carla_job.sh`` run on another port is never touched.
        """
        if self._extra_cleanup is not None:
            try:
                self._extra_cleanup()
            except Exception as exc:
                print(f"[stall-watchdog] extra cleanup failed: {exc}", flush=True)

        rpc_port, display_num = self._rpc_port, self._display_num
        if self._resolve_ports is not None:
            try:
                resolved_rpc, resolved_display = self._resolve_ports()
                rpc_port = resolved_rpc or rpc_port
                display_num = resolved_display or display_num
            except Exception:
                pass

        patterns = []
        if rpc_port:
            patterns.append(f"carla-rpc-port={int(rpc_port)}")
        if display_num:
            patterns.append(f"Xvfb :{int(display_num)} ")
        if not patterns:
            print(
                "[stall-watchdog] no resolved rpc port/display; leaving CARLA to the launcher's cleanup.",
                flush=True,
            )
            return
        for pattern in patterns:
            try:
                subprocess.run(
                    ["pkill", "-9", "-f", pattern],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                )
            except Exception:
                pass
        print(f"[stall-watchdog] killed CARLA/Xvfb matching {patterns}.", flush=True)


def _beating(fn: Callable[..., Any], watchdog: StallWatchdog) -> Callable[..., Any]:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        out = fn(*args, **kwargs)
        # Beat on RETURN, not on entry: "last tick that completed" is the quantity the stall
        # check needs, and a call that never returns is precisely what we are hunting.
        watchdog.beat()
        return out

    return wrapper


def install(
    env: Any,
    *,
    timeout_s: float,
    cpu_idle_frac: float = 0.05,
    rpc_port: Optional[int] = None,
    display_num: Optional[int] = None,
) -> Optional[StallWatchdog]:
    """Patch ``env.step``/``env.reset`` to beat, start the watcher thread, return it.

    Patching the env instance (rather than editing each loop) covers ``run_online_carla``,
    ``run_online_residual`` and ``run_online_grpo`` from the one choke point where every
    CARLA env is built, ``main_carla._make_carla_env``. ``timeout_s <= 0`` disables.
    """
    global _ACTIVE
    if timeout_s <= 0:
        return None
    if _ACTIVE is not None:
        return _ACTIVE

    def _resolve_ports() -> tuple[Optional[int], Optional[int]]:
        # Ports may be 0 in the yaml ("auto-assign a free one"); the real values only exist
        # after IsolatedLeaderboardEvaluator._setup_simulation has run.
        evaluator = getattr(env, "_evaluator", None)
        return (
            getattr(evaluator, "_launch_rpc_port", None),
            getattr(evaluator, "_launch_display_num", None),
        )

    def _kill_env_subprocess() -> None:
        # CARLA_ENV_SUBPROCESS_PYTHON path: the env (and its CARLA server) live in a
        # carla_env_server child that our os._exit would otherwise orphan on the GPU.
        # SIGTERM first -- the server has an atexit that shuts CARLA down properly.
        proc = getattr(env, "_proc", None)
        if proc is None or proc.poll() is not None:
            return
        print(f"[stall-watchdog] terminating carla_env_server child pid={proc.pid}.", flush=True)
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()

    watchdog = StallWatchdog(
        timeout_s=timeout_s,
        cpu_idle_frac=cpu_idle_frac,
        rpc_port=rpc_port,
        display_num=display_num,
        resolve_ports=_resolve_ports,
        extra_cleanup=_kill_env_subprocess,
    )
    for name in ("step", "reset"):
        fn = getattr(env, name, None)
        if callable(fn):
            setattr(env, name, _beating(fn, watchdog))
    _ACTIVE = watchdog
    return watchdog.start()
