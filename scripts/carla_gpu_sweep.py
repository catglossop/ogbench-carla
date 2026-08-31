#!/usr/bin/env python3
"""Boot CARLA once per ``-graphicsadapter`` index and report which GPUs work.

Mirrors the launch that ``IsolatedLeaderboardEvaluator._setup_simulation``
performs (``ogbench/carla/carla_utils.py``) -- same Xvfb, same UE4 flags, same
subprocess environment -- so a pass/fail here transfers directly to a real run.

For each adapter it:

1. starts a private Xvfb display and ``CarlaUE4.sh -RenderOffScreen``,
2. waits for the RPC port to accept connections,
3. connects the CARLA client, spawns a vehicle plus an RGB camera and counts
   frames actually delivered -- booting proves Vulkan init, rendered frames
   prove the graphics pipeline survives real work,
4. records which physical GPU the UE4 process landed on, and
5. tears its own processes down, scoped to its own port/display so parallel
   jobs on the box are untouched.

Each adapter gets its own RPC port and X display, so a wedged card cannot leak
into the next iteration.
"""

import argparse
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Client-side check, run in a separate interpreter so a hung/crashed CARLA
# client cannot take the sweep down with it.
CLIENT_PROBE = r'''
import json, sys, time
import carla

host, port, frames, timeout = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
out = {"ok": False, "stage": "connect", "frames": 0, "error": None}
world = None
prev = None
actors = []
try:
    client = carla.Client(host, port)
    client.set_timeout(timeout)
    out["server_version"] = client.get_server_version()
    out["client_version"] = client.get_client_version()

    out["stage"] = "get_world"
    world = client.get_world()
    out["map"] = world.get_map().name

    out["stage"] = "sync_mode"
    prev = world.get_settings()
    s = world.get_settings()
    s.synchronous_mode = True
    s.fixed_delta_seconds = 0.05
    world.apply_settings(s)

    out["stage"] = "spawn"
    bp = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    vehicle_bp = bp.filter("vehicle.*")[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
    actors.append(vehicle)

    cam_bp = bp.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "400")
    cam_bp.set_attribute("image_size_y", "300")
    cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=vehicle)
    actors.append(cam)

    got = []
    cam.listen(lambda img: got.append(img.frame))

    out["stage"] = "render"
    t0 = time.time()
    for _ in range(frames):
        world.tick()
    # Sensor callbacks land asynchronously; give them a moment to drain.
    deadline = time.time() + 10.0
    while len(got) < frames and time.time() < deadline:
        time.sleep(0.05)
    out["frames"] = len(got)
    out["render_seconds"] = round(time.time() - t0, 2)
    out["ok"] = len(got) > 0
    out["stage"] = "done"
except Exception as exc:
    out["error"] = f"{type(exc).__name__}: {exc}"
finally:
    for a in reversed(actors):
        try:
            if hasattr(a, "stop"):
                a.stop()
            a.destroy()
        except Exception:
            pass
    if world is not None and prev is not None:
        try:
            world.apply_settings(prev)
        except Exception:
            pass

print("PROBE_JSON " + json.dumps(out))
'''


def carla_subprocess_env(display_num, adapter):
    """Same minimal UE4 env the repo builds in ``_carla_subprocess_env``."""
    sys_path = '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
    env = {
        'HOME': os.environ.get('HOME', '/root'),
        'USER': os.environ.get('USER', 'root'),
        'LOGNAME': os.environ.get('LOGNAME', os.environ.get('USER', 'root')),
        'PATH': sys_path,
        'DISPLAY': f':{display_num}',
        'LANG': os.environ.get('LANG', 'C.UTF-8'),
        'NVIDIA_VISIBLE_DEVICES': str(adapter),
        'CUDA_VISIBLE_DEVICES': '0',
        '__GLX_VENDOR_LIBRARY_NAME': 'nvidia',
    }
    for candidate in (
        '/etc/vulkan/icd.d/nvidia_icd.json',
        '/etc/vulkan/icd.d/nvidia_icd.x86_64.json',
        '/usr/share/vulkan/icd.d/nvidia_icd.json',
        '/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json',
    ):
        if os.path.exists(candidate):
            env['VK_ICD_FILENAMES'] = candidate
            break
    return env


def gpu_table():
    """``nvidia-smi`` index -> {uuid, pci bus id, memory used}."""
    out = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id,memory.used', '--format=csv,noheader,nounits'],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    table = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 4:
            continue
        idx, uuid, bus, used = parts[0], parts[1], parts[2], parts[3]
        table[int(idx)] = {'uuid': uuid, 'pci_bus_id': bus, 'memory_used_mib': int(used)}
    return table


def gpu_index_for_pid(pid, table, baseline=None):
    """Which nvidia-smi GPU index a pid landed on, and how much VRAM it took.

    UE4 renders through Vulkan, so on some driver/kernel-module combinations it
    is a *graphics* client and never appears in ``--query-compute-apps``.  Three
    attempts, most direct first: the compute-app list, the full ``nvidia-smi``
    process table (which also lists ``G`` clients), and finally the per-GPU
    memory delta against a baseline taken just before launch.
    """
    uuid_to_index = {v['uuid']: k for k, v in table.items()}
    out = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 3 and parts[1].isdigit() and int(parts[1]) == pid:
            return uuid_to_index.get(parts[0]), int(parts[2]), 'compute-apps'

    # Full text table: "|    3   N/A  N/A   3172517      G   ...CarlaUE4  1234MiB |"
    full = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=False).stdout
    for line in full.splitlines():
        if str(pid) not in line:
            continue
        m = re.match(r'\|\s+(\d+)\s+\S+\s+\S+\s+(\d+)\s+\S+\s+.*?(\d+)MiB', line)
        if m and int(m.group(2)) == pid:
            return int(m.group(1)), int(m.group(3)), 'nvidia-smi table'

    if baseline:
        now = gpu_table()
        deltas = {i: now[i]['memory_used_mib'] - baseline.get(i, {}).get('memory_used_mib', 0) for i in now}
        idx = max(deltas, key=deltas.get)
        if deltas[idx] >= 200:
            return idx, deltas[idx], 'memory delta'
    return None, None, 'unattributed'


def unhealthy_gpu_indices():
    """Indices ``nvidia-smi`` reports as ``ERR!`` (a wedged card)."""
    out = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=False).stdout
    bad = []
    for idx in sorted(gpu_table()):
        probe = subprocess.run(['nvidia-smi', '-i', str(idx)], capture_output=True, text=True, check=False)
        if 'ERR!' in probe.stdout + probe.stderr:
            bad.append(idx)
    return bad, ('ERR!' in out)


def port_open(host, port, timeout=1.0):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def clear_display_lock(display_num):
    for path in (f'/tmp/.X{display_num}-lock', f'/tmp/.X11-unix/X{display_num}'):
        if not os.path.exists(path):
            continue
        try:
            os.unlink(path)
        except OSError:
            pass


def kill_scoped(rpc_port, display_num):
    """Kill only the CARLA/Xvfb belonging to this rpc port / display."""
    subprocess.run(['pkill', '-9', '-f', f'CarlaUE4.*-carla-rpc-port={rpc_port}'], capture_output=True)
    subprocess.run(['pkill', '-9', '-f', f'Xvfb :{display_num}'], capture_output=True)


def tail(path, n_bytes=3000):
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            data = f.read()
    except OSError as exc:
        return f'(log unreadable: {exc})'
    if not data.strip():
        return '(log empty -- UE4 died before writing anything: Vulkan/GPU init failure or a fatal signal)'
    return data[-n_bytes:]


def scan_log_for_gpu_errors(path):
    """Pull the lines that actually explain a GPU-side failure out of a UE4 log."""
    patterns = (
        r'VK_ERROR_DEVICE_LOST',
        r'device lost',
        r'Vulkan',
        r'GPU [Cc]rash',
        r'LogVulkanRHI',
        r'Failed to find all required Vulkan',
        r'no compatible.*device',
        r'Fatal error',
        r'out of memory',
    )
    rx = re.compile('|'.join(patterns), re.IGNORECASE)
    hits = []
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                if rx.search(line):
                    hits.append(line.rstrip())
    except OSError:
        return []
    # Keep the head and tail of the matches; the middle is repetitive banner noise.
    return hits[:8] + (['   ...'] + hits[-12:] if len(hits) > 20 else hits[8:])


def run_one(adapter, args, gpus, log_dir, attempt=0):
    rpc_port = args.base_rpc_port + 10 * adapter
    streaming_port = rpc_port + 1
    display_num = args.base_display + adapter
    suffix = f'_try{attempt}' if args.repeat > 1 else ''
    log_path = os.path.join(log_dir, f'carla_adapter{adapter}_rpc{rpc_port}{suffix}.log')

    result = {
        'adapter': adapter,
        'rpc_port': rpc_port,
        'display': display_num,
        'log': log_path,
        'boot_ok': False,
        'boot_seconds': None,
        'client_ok': False,
        'frames': 0,
        'gpu_index_observed': None,
        'gpu_mem_mib': None,
        'error': None,
        'log_gpu_errors': [],
    }

    print(f'\n=== adapter {adapter} -> rpc {rpc_port}, display :{display_num} ===', flush=True)

    # Start clean: nothing from a previous attempt may share this port/display.
    kill_scoped(rpc_port, display_num)
    time.sleep(1)
    clear_display_lock(display_num)

    xvfb = subprocess.Popen(
        ['Xvfb', f':{display_num}', '-screen', '0', '1280x1024x24', '-ac', '+extension', 'GLX', '+render', '-noreset'],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    time.sleep(2)
    if xvfb.poll() is not None:
        result['error'] = f'Xvfb :{display_num} exited immediately (rc={xvfb.returncode})'
        print(f'  FAIL: {result["error"]}', flush=True)
        return result

    # Baseline VRAM per GPU, so the UE4 process can be attributed by delta if the
    # driver does not list it as a client of any GPU.
    baseline = gpu_table()

    env = carla_subprocess_env(display_num, adapter)
    cmd = [
        os.path.join(args.carla_root, 'CarlaUE4.sh'),
        '-RenderOffScreen',
        '-nosound',
        '-g.TimeoutForBlockOnRenderFence=300000',
        f'-carla-rpc-port={rpc_port}',
        f'-graphicsadapter={adapter}',
        f'-carla-streaming-port={streaming_port}',
    ]
    print(f'  launch: {" ".join(cmd)}', flush=True)

    server = None
    try:
        with open(log_path, 'w', buffering=1) as logf:
            server = subprocess.Popen(
                cmd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=logf,
                stderr=logf,
                start_new_session=True,
            )
            result['pid'] = server.pid

            boot_start = time.time()
            while time.time() - boot_start < args.boot_timeout:
                if server.poll() is not None:
                    result['error'] = (
                        f'UE4 exited during boot after {int(time.time() - boot_start)}s (rc={server.returncode})'
                    )
                    break
                if port_open(args.host, rpc_port):
                    result['boot_ok'] = True
                    result['boot_seconds'] = round(time.time() - boot_start, 1)
                    break
                time.sleep(2)
            else:
                result['error'] = f'RPC port {rpc_port} never opened within {args.boot_timeout}s'

            if result['boot_ok']:
                print(f'  boot OK in {result["boot_seconds"]}s', flush=True)
                idx, mem, how = gpu_index_for_pid(server.pid, gpus, baseline)
                result['gpu_index_observed'], result['gpu_mem_mib'], result['gpu_attribution'] = idx, mem, how
                print(f'  UE4 pid {server.pid} on nvidia-smi GPU {idx} ({mem} MiB, via {how})', flush=True)

                probe = subprocess.run(
                    [args.python, '-c', CLIENT_PROBE, args.host, str(rpc_port), str(args.frames), str(args.client_timeout)],
                    capture_output=True,
                    text=True,
                    timeout=args.client_timeout * 4,
                    check=False,
                )
                for line in probe.stdout.splitlines():
                    if line.startswith('PROBE_JSON '):
                        data = json.loads(line[len('PROBE_JSON ') :])
                        result.update(
                            {
                                'client_ok': data.get('ok', False),
                                'frames': data.get('frames', 0),
                                'client_stage': data.get('stage'),
                                'map': data.get('map'),
                                'server_version': data.get('server_version'),
                                'render_seconds': data.get('render_seconds'),
                            }
                        )
                        if data.get('error'):
                            result['error'] = data['error']
                        break
                else:
                    result['error'] = f'client probe produced no result (rc={probe.returncode})'
                    result['client_stderr'] = probe.stderr[-1500:]

                status = 'OK' if result['client_ok'] else 'FAIL'
                print(f'  client {status}: {result["frames"]}/{args.frames} frames, map={result.get("map")}', flush=True)
                if result['error']:
                    print(f'  error: {result["error"]}', flush=True)
            else:
                print(f'  FAIL: {result["error"]}', flush=True)
    finally:
        if server is not None and server.poll() is None:
            try:
                os.killpg(os.getpgid(server.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                server.kill()
        if xvfb.poll() is None:
            try:
                os.killpg(os.getpgid(xvfb.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                xvfb.kill()
        kill_scoped(rpc_port, display_num)
        time.sleep(args.cooldown)
        clear_display_lock(display_num)

    result['log_gpu_errors'] = scan_log_for_gpu_errors(log_path)
    if not result['boot_ok']:
        result['log_tail'] = tail(log_path)
    return result


def parse_adapters(spec, available):
    if spec == 'all':
        return sorted(available)
    out = []
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            lo, hi = part.split('-', 1)
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--adapters', default='all', help='"all", or e.g. "0,3,5" / "0-7"')
    ap.add_argument('--carla-root', default=os.environ.get('CARLA_ROOT', ''), help='packaged CARLA install')
    ap.add_argument('--python', default=sys.executable, help='interpreter that can "import carla"')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--base-rpc-port', type=int, default=13400, help='adapter N uses base + 10*N')
    ap.add_argument('--base-display', type=int, default=60, help='adapter N uses base + N')
    ap.add_argument('--boot-timeout', type=float, default=180.0, help='seconds to wait for the RPC port')
    ap.add_argument('--client-timeout', type=float, default=60.0, help='CARLA client timeout')
    ap.add_argument('--frames', type=int, default=20, help='camera frames to demand per GPU')
    ap.add_argument('--cooldown', type=float, default=8.0, help='seconds between adapters, for teardown to settle')
    ap.add_argument(
        '--repeat',
        type=int,
        default=1,
        help='attempts per adapter; >1 separates a genuinely dead card from a flaky race',
    )
    ap.add_argument('--log-dir', default=os.path.join(REPO_ROOT, '.run_carla', 'gpu_sweep'))
    ap.add_argument('--json', help='write the full result table here')
    args = ap.parse_args()

    if not args.carla_root or not os.path.isfile(os.path.join(args.carla_root, 'CarlaUE4.sh')):
        raise SystemExit(
            f'no CarlaUE4.sh under --carla-root={args.carla_root!r}; pass --carla-root or export CARLA_ROOT'
        )
    if shutil.which('Xvfb') is None:
        raise SystemExit('Xvfb not found on PATH')

    gpus = gpu_table()
    bad, any_err = unhealthy_gpu_indices()
    print(f'CARLA_ROOT   = {args.carla_root}')
    print(f'client python= {args.python}')
    print(f'GPUs         = {len(gpus)} visible to nvidia-smi')
    for idx in sorted(gpus):
        print(f'  [{idx}] {gpus[idx]["pci_bus_id"]}  {gpus[idx]["memory_used_mib"]} MiB in use')
    if bad:
        print(f'WARNING: nvidia-smi reports ERR! on GPU(s) {bad} -- those are wedged until a reboot')

    adapters = parse_adapters(args.adapters, gpus)
    os.makedirs(args.log_dir, exist_ok=True)
    print(f'\nsweeping adapters {adapters}; logs -> {args.log_dir}')

    results = []
    for attempt in range(args.repeat):
        if args.repeat > 1:
            print(f'\n########## attempt {attempt + 1}/{args.repeat} ##########', flush=True)
        for adapter in adapters:
            try:
                r = run_one(adapter, args, gpus, args.log_dir, attempt)
                r['attempt'] = attempt
                results.append(r)
            except KeyboardInterrupt:
                print('\ninterrupted', flush=True)
                attempt = args.repeat
                break
            except Exception as exc:  # keep sweeping: one bad card must not end the run
                results.append(
                    {'adapter': adapter, 'attempt': attempt, 'boot_ok': False, 'client_ok': False, 'error': f'sweep error: {exc}'}
                )
                print(f'  sweep error on adapter {adapter}: {exc}', flush=True)

    print('\n' + '=' * 78)
    print('SUMMARY  (adapter = the -graphicsadapter / carla_config.yaml gpu_rank value)')
    print('=' * 78)
    header = f'{"adapter":>7} {"boot":>6} {"boot_s":>7} {"frames":>7} {"gpu@smi":>8} {"verdict":>8}  notes'
    print(header)
    print('-' * 78)
    for r in results:
        verdict = 'USABLE' if r.get('client_ok') else 'BROKEN'
        note = r.get('error') or ''
        if r.get('log_gpu_errors') and not r.get('client_ok'):
            note = (note + ' | ' + r['log_gpu_errors'][0]) if note else r['log_gpu_errors'][0]
        print(
            f'{r["adapter"]:>7} {"yes" if r.get("boot_ok") else "no":>6} '
            f'{str(r.get("boot_seconds") or "-"):>7} {r.get("frames", 0):>7} '
            f'{str(r.get("gpu_index_observed") if r.get("gpu_index_observed") is not None else "-"):>8} '
            f'{verdict:>8}  {note[:120]}'
        )

    tally = {}
    for r in results:
        a = r['adapter']
        tally.setdefault(a, [0, 0])
        tally[a][0] += 1 if r.get('client_ok') else 0
        tally[a][1] += 1

    if args.repeat > 1:
        print('\nper-adapter pass rate:')
        for a in sorted(tally):
            ok, total = tally[a]
            print(f'  adapter {a}: {ok}/{total} attempts usable')

    usable = sorted(a for a, (ok, _) in tally.items() if ok)
    always = sorted(a for a, (ok, total) in tally.items() if ok == total)
    never = sorted(a for a, (ok, _) in tally.items() if ok == 0)
    flaky = sorted(a for a, (ok, total) in tally.items() if 0 < ok < total)

    print('\nusable at least once:', usable if usable else 'NONE')
    if args.repeat > 1:
        print('usable every attempt :', always if always else 'NONE')
        print('flaky                :', flaky if flaky else 'none')
    print('never usable         :', never if never else 'none')
    if never or flaky:
        print('per-adapter UE4 logs are under', args.log_dir)

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2)
        print('wrote', args.json)

    return 0 if usable else 1


if __name__ == '__main__':
    sys.exit(main())
