#!/usr/bin/env python3
"""b70 Music Studio: a web UI and job queue for YuE2 on an Intel XPU.

GPU policy: when a song is queued and the LLM service holds the GPUs, the
studio pauses the LLM and starts a GPU child process that loads YuE2 and stays
loaded, so later songs start immediately. The LLM is not restarted
automatically: someone presses "Give GPUs back" (POST /api/gpu/release),
which ends the child process (the only way to fully release its GPU memory),
waits until the cards are free, then starts the LLM. STUDIO_RETURN_MINUTES > 0
re-enables an automatic return after that many idle minutes (default 0 = never).

Library layout: DATA/<job id>/job.json, plus DATA/<job id>/song/ (YuE2 artifacts)
and song.mp3. Stdlib HTTP server; run with the YuE2 venv's python.
"""
import dataclasses, json, multiprocessing as mp, os, queue, urllib.request, random, re, shutil, subprocess, sys, threading, time, traceback, uuid
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("STUDIO_DATA", "~/yue2/studio/library")).expanduser()
DEVICES = [d.strip() for d in os.environ.get("STUDIO_DEVICES", os.environ.get("STUDIO_DEVICE", "xpu:0,xpu:1")).split(",") if d.strip()]
DEVICE = DEVICES[0]
RETURN_MINUTES = float(os.environ.get("STUDIO_RETURN_MINUTES", "0"))  # 0 = never auto-return
LLM_UNIT = os.environ.get("STUDIO_LLM_UNIT", "vllm-qwen38.service")
PORT = int(os.environ.get("STUDIO_PORT", "7860"))
SKILL = Path(os.environ.get("STUDIO_INSTRUMENTAL", "~/yue2/YuE/skills/yue2-music/instrumental/scripts")).expanduser()
MAX_QUEUED = 24
# Lyrics / songwriter LLMs (OpenAI-compatible), tried in order: "base_url|model|label,...". The local LLM is
# paused while the studio holds the GPUs, so add a second server on another machine as the fallback.
LYRICS_LLMS = [tuple(x.split("|")) for x in os.environ.get(
    "STUDIO_LYRICS_LLMS", "http://localhost:8000/v1|qwen38|local LLM").split(",") if x.strip()]
TOKENS_PER_AUDIO_SECOND = 25          # YuE2 semantic tokens per second of audio
GPU_STATE = DATA.parent / "gpu.json"  # remembers whether we paused the LLM (survives restarts)
FILES = {"audio.flac": ("song/audio.flac", "audio/flac"), "song.mp3": ("song.mp3", "audio/mpeg"),
         "score.abc": ("song/score.abc", "text/plain; charset=utf-8"),
         "source.abc": ("source.abc", "text/plain; charset=utf-8")}
SAMPLING_KEYS = {"temperature": (float, 0.0, 5.0), "top_p": (float, 0.01, 1.0), "top_k": (int, 1, 1000),
                 "repetition_penalty": (float, 0.5, 3.0)}

lock = threading.RLock()
jobs: dict[str, dict] = {}
gpu = {"owner": "llm", "detail": "", "idle_since": None, "release_requested": False, "release_on_idle": False, "workers": []}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sh(*cmd, timeout=180):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception as e:
        return f"error: {e}"


def llm_active():
    return sh("systemctl", "is-active", LLM_UNIT) in ("active", "activating")


def save_job(job):
    d = DATA / job["id"]; d.mkdir(parents=True, exist_ok=True)
    tmp = d / "job.json.tmp"
    tmp.write_text(json.dumps({k: v for k, v in job.items() if not k.startswith("_")}, indent=1), encoding="utf-8")
    tmp.replace(d / "job.json")


def load_library():
    DATA.mkdir(parents=True, exist_ok=True)
    for f in DATA.glob("*/job.json"):
        try:
            job = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if job.get("status") in ("queued", "running"):
            job.update(status="failed", error="Interrupted: the studio restarted before this finished", stage="")
            save_job(job)
        jobs[job["id"]] = job


def read_gpu_state():
    try:
        return json.loads(GPU_STATE.read_text())
    except Exception:
        return {"paused_llm": False}


def write_gpu_state(paused):
    GPU_STATE.parent.mkdir(parents=True, exist_ok=True)
    GPU_STATE.write_text(json.dumps({"paused_llm": paused, "time": time.time()}))


# ----------------------------------------------------------------------------- validation

class BadRequest(ValueError):
    pass


def clean_text(v, name, limit, required=False):
    if v is None:
        v = ""
    if not isinstance(v, str):
        raise BadRequest(f"{name} must be text")
    v = v.replace("\r\n", "\n").strip()
    if required and not v:
        raise BadRequest(f"{name} is required")
    if len(v) > limit:
        raise BadRequest(f"{name} is too long (max {limit} characters)")
    return v


def clean_sampling(raw, name):
    out = {}
    for k, v in (raw or {}).items():
        if v in (None, ""):
            continue
        if k not in SAMPLING_KEYS:
            raise BadRequest(f"unknown {name} setting {k}")
        typ, lo, hi = SAMPLING_KEYS[k]
        try:
            v = typ(v)
        except Exception:
            raise BadRequest(f"{name} {k} must be a number")
        if not lo <= v <= hi:
            raise BadRequest(f"{name} {k} must be between {lo} and {hi}")
        out[k] = v
    return out


def validate(body):
    mode = body.get("mode", "song")
    if mode not in ("song", "instrumental"):
        raise BadRequest("mode must be song or instrumental")
    job = {"mode": mode,
           "title": clean_text(body.get("title"), "Title", 120),
           "author": clean_text(body.get("author"), "Name", 60),
           "style": clean_text(body.get("style"), "Style", 1500, required=True),
           "lyrics": clean_text(body.get("lyrics"), "Lyrics", 20000, required=(mode == "song")),
           "abc": clean_text(body.get("abc"), "Score", 65536) or None}
    cot = body.get("cot", "full")
    if cot not in ("full", "melody", "off"):
        raise BadRequest("Planning mode must be full, melody or off")
    if job["abc"] and cot == "off":
        raise BadRequest("A custom score needs planning mode full or melody")
    if mode == "instrumental" and cot == "off":
        raise BadRequest("Instrumental mode needs planning mode full or melody")
    job["cot"] = cot
    seed = body.get("seed")
    if seed in (None, "", "random"):
        seed = random.randrange(1, 2**31)
    try:
        seed = int(seed)
    except Exception:
        raise BadRequest("Seed must be a whole number")
    if not 0 <= seed < 2**63:
        raise BadRequest("Seed out of range")
    job["seed"] = seed
    cfg = body.get("cfg_scale")
    if cfg not in (None, ""):
        try:
            cfg = float(cfg)
        except Exception:
            raise BadRequest("Style adherence must be a number")
        if not 1.0 <= cfg <= 5.0:
            raise BadRequest("Style adherence must be between 1.0 and 5.0")
        job["cfg_scale"] = None if cfg == 1.0 else cfg
    else:
        job["cfg_scale"] = None
    try:
        job["max_seconds"] = int(body.get("max_seconds") or 360)
        job["ode_steps"] = int(body.get("ode_steps") or 32)
        takes = int(body.get("takes") or 1)
    except Exception:
        raise BadRequest("Length, quality steps and takes must be whole numbers")
    if not 20 <= job["max_seconds"] <= 360:
        raise BadRequest("Max length must be between 20 and 360 seconds")
    if not 8 <= job["ode_steps"] <= 64:
        raise BadRequest("Audio quality steps must be between 8 and 64")
    if not 1 <= takes <= 4:
        raise BadRequest("Takes must be 1 to 4")
    job["abc_sampling"] = clean_sampling(body.get("abc_sampling"), "Score sampling")
    job["semantic_sampling"] = clean_sampling(body.get("semantic_sampling"), "Song sampling")
    return job, takes


def new_jobs(body, extra=None):
    base, takes = validate(body)
    extra = extra or {}
    with lock:
        queued = sum(1 for j in jobs.values() if j["status"] in ("queued", "running") and j.get("source") != "radio")
        if not extra.get("source") and queued + takes > MAX_QUEUED:
            raise BadRequest(f"The queue is full ({queued} waiting); try again later")
        batch = uuid.uuid4().hex[:8]
        created = []
        for i in range(takes):
            jid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
            job = dict(base, id=jid, batch=batch, take=i + 1, takes=takes, seed=base["seed"] + i,
                       created=time.time(), status="queued", stage="Waiting in queue", progress={},
                       source="studio", priority=0)
            job.update(extra)
            if not job["title"]:
                first = next((l for l in job["lyrics"].splitlines() if l.strip() and not l.strip().startswith("[")), "")
                job["title"] = (first[:60] or job["style"].split(",")[0][:60] or "Untitled").strip()
            jobs[jid] = job
            save_job(job)
            created.append(jid)
    log("queued", created)
    return created


# ----------------------------------------------------------------------------- GPU worker

def gpu_free_mib(index):
    """Used VRAM in MiB on one card, via xpu-smi (the parent process never initialises the GPU)."""
    try:
        j = json.loads(sh("xpu-smi", "stats", "-d", str(index), "-j", timeout=20))
        return j["memory"]["used_mib"]["tile_0"]["current"]
    except Exception:
        return None


def soften_ending(path, fade=6.0):
    """A song that stops at full volume (usually the length cap cut it off) gets a radio-style fade-out.
    Runs on the lossless render before the MP3 is made. Returns True if the file was changed."""
    import numpy as np, soundfile as sf
    path = Path(path)
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    if len(y) < sr * 12:
        return False
    rms = lambda a: float(np.sqrt(np.mean(a ** 2)) + 1e-9)
    if 20 * np.log10(rms(y[-sr // 2:]) / rms(y[:-5 * sr])) < -12:      # already ends quietly on its own
        return False
    n = int(min(fade, len(y) / sr / 4) * sr)
    y[-n:] *= (0.5 * (1 + np.cos(np.linspace(0, np.pi, n, dtype=np.float32))))[:, None]
    tmp = path.with_name(".fade.flac")
    sf.write(tmp, y, sr, subtype=sf.info(str(path)).subtype)
    os.replace(tmp, path)                    # atomic: a broadcast already reading the old file keeps its copy
    return True


def run_job(pipe, base, job, update, cancelled):
    """Runs in the GPU child process. update(**fields) reports progress/results to the parent."""
    from yue2.protocol import SongRequest
    d = DATA / job["id"]
    t0 = time.time()
    counts, phase_start = {}, {}

    def on_token(phase, token):
        counts[phase] = counts.get(phase, 0) + 1
        now = time.time(); phase_start.setdefault(phase, now)
        n = counts[phase]
        if n % 4:                       # throttle IPC
            return
        tps = n / max(now - phase_start[phase], 1e-3)
        if phase == "abc":
            update(stage="Composing the score", progress={"phase": phase, "tokens": n, "tps": round(tps, 1)})
        else:
            update(stage="Generating the song", progress={"phase": phase, "tokens": n, "tps": round(tps, 1),
                   "music_seconds": round(n / TOKENS_PER_AUDIO_SECOND, 1), "max_music_seconds": job["max_seconds"]})

    abc_s = dataclasses.replace(base.abc, **job["abc_sampling"])
    sem_max = max(base.semantic.min_tokens + 1, min(base.semantic.max_tokens, job["max_seconds"] * TOKENS_PER_AUDIO_SECOND))
    sem_s = dataclasses.replace(base.semantic, max_tokens=int(sem_max), **job["semantic_sampling"])
    pipe.generation_config = dataclasses.replace(base, ode_steps=job["ode_steps"])
    orig_synth, orig_decode = pipe.synthesize, pipe.decode

    def synth(*a, **k):
        update(stage="Rendering audio", progress={"phase": "audio"}); return orig_synth(*a, **k)

    def decode(*a, **k):
        update(stage="Decoding audio", progress={"phase": "decode"}); return orig_decode(*a, **k)

    pipe.synthesize, pipe.decode = synth, decode
    update(status="running", started=t0, stage="Starting", progress={})
    log("start", job["id"], job["mode"], job["cot"], "seed", job["seed"])
    try:
        common = dict(style=job["style"], seed=job["seed"], cfg_scale=job["cfg_scale"],
                      abc_sampling=abc_s, semantic_sampling=sem_s, cancelled=cancelled, on_token=on_token)
        if job["mode"] == "instrumental":
            if str(SKILL) not in sys.path:
                sys.path.insert(0, str(SKILL))
            from instrumentalize import convert_score
            from instrumental import lyric_tags
            source = job["abc"]
            if not source:
                structure = job["lyrics"] or "[Intro]\n\n[Verse]\n\n[Chorus]\n\n[Verse]\n\n[Chorus]\n\n[Outro]"
                request = SongRequest(style=job["style"], lyrics=structure, cot=job["cot"], seed=job["seed"])
                plan = pipe.plan(request=request, abc_sampling=abc_s, cancelled=cancelled, on_token=on_token)
                if plan.truncated or not plan.abc:
                    raise RuntimeError("The model's score was empty or cut off; try another seed")
                source = plan.abc
            d.mkdir(parents=True, exist_ok=True)
            (d / "source.abc").write_text(source, encoding="utf-8")
            update(stage="Moving the melody to instruments")
            converted, check = convert_score(source, overlap="vocal", keep_chords=job["cot"] == "full")
            (d / "instrumental-check.json").write_text(json.dumps(check, indent=1), encoding="utf-8")
            counts.clear(); phase_start.clear()
            song = pipe(lyrics=lyric_tags(converted), cot=job["cot"], abc=converted, **common)
        else:
            song = pipe(lyrics=job["lyrics"], cot=job["cot"], abc=job["abc"], **common)
        out = d / "song"
        if out.exists():
            shutil.rmtree(out)
        song.save_artifacts(out)
        faded = False
        try:
            faded = soften_ending(out / "audio.flac")
        except Exception:
            log("fade-out failed", job["id"], traceback.format_exc())
        if faded:
            update(faded=True); log("faded out an abrupt ending", job["id"])
        update(stage="Encoding MP3")
        # the library keeps one 320 kbps MP3 per song; the 24-bit FLAC render is only an intermediate
        sh("ffmpeg", "-y", "-loglevel", "error", "-i", str(out / "audio.flac"), "-codec:a", "libmp3lame", "-b:a", "320k",
           "-metadata", f"title={job['title']}", "-metadata", "artist=YuE2 on b70", str(d / "song.mp3"))
        if (d / "song.mp3").stat().st_size > 10_000:
            (out / "audio.flac").unlink()
        result = json.loads((out / "result.json").read_text())
        gen = round(time.time() - t0, 1)
        update(status="done", stage="", finished=time.time(), seconds=result.get("audio_seconds"),
               truncated=result.get("truncated"), timing={k: v for k, v in result.get("timing", {}).items() if k != "load"},
               gen_seconds=gen, progress={})
        log("done", job["id"], f"{result.get('audio_seconds', 0):.1f}s audio in {gen}s")
    except InterruptedError:
        update(status="cancelled", stage="", finished=time.time(), progress={})
        log("cancelled", job["id"])
    except Exception as e:
        update(status="failed", stage="", finished=time.time(), error=f"{type(e).__name__}: {e}"[:500], progress={})
        log("failed", job["id"], traceback.format_exc())
    finally:
        pipe.synthesize, pipe.decode = orig_synth, orig_decode
        pipe.generation_config = base


def gpu_main(device, task_q, event_q, cancel_ev):
    """GPU child process for one card: loads YuE2 once, renders jobs until told to stop, then exits (freeing all VRAM)."""
    try:
        from yue2 import YuE2Pipeline
        pipe = YuE2Pipeline.from_pretrained("m-a-p/YuE2-3B", device=device, progress=False)
        base = pipe.generation_config
        event_q.put(("ready", None, {}))
    except Exception as e:
        event_q.put(("fatal", None, {"error": f"{type(e).__name__}: {e}"}))
        return
    while True:
        job = task_q.get()
        if job is None:
            break
        cancel_ev.clear()
        run_job(pipe, base, job, lambda **kw: event_q.put(("update", job["id"], kw)), cancel_ev.is_set)
        event_q.put(("finished", job["id"], {}))
    try:
        pipe.close()
    except Exception:
        pass


class Slot:
    def __init__(self, device):
        self.device, self.index = device, int(device.split(":")[1]) if ":" in device else 0
        self.proc = self.job = None
        self.ready = False
        self.fails = 0              # consecutive failed jobs
        self.restarts = []          # restart timestamps (rate limit)
        self.disabled = False

    def alive(self):
        return self.proc is not None and self.proc.is_alive()

    def start(self):
        ctx = mp.get_context("spawn")
        self.task_q, self.event_q, self.cancel_ev = ctx.Queue(), ctx.Queue(), ctx.Event()
        self.proc = ctx.Process(target=gpu_main, args=(self.device, self.task_q, self.event_q, self.cancel_ev),
                                daemon=True, name=f"yue2-{self.device}")
        self.proc.start(); self.ready = False; self.job = None

    def stop(self):
        if self.proc is None:
            return
        try:
            self.task_q.put(None); self.proc.join(60)
        except Exception:
            pass
        if self.proc.is_alive():
            self.proc.kill(); self.proc.join(10)
        self.proc, self.ready, self.job = None, False, None


class Worker(threading.Thread):
    """Parent-side scheduler for one YuE2 process per GPU. Never touches the GPU itself.
    Studio jobs (priority 0) always go before radio jobs (priority 1)."""
    def __init__(self):
        super().__init__(daemon=True)
        self.slots = [Slot(d) for d in DEVICES]
        self.paused_llm = read_gpu_state().get("paused_llm", False)
        self.last_activity = time.time()

    def set_gpu(self, owner, detail=""):
        with lock:
            gpu["owner"], gpu["detail"] = owner, detail
            gpu["workers"] = [{"device": sl.device, "ready": sl.ready, "job": sl.job["title"] if sl.job else None,
                               "disabled": sl.disabled} for sl in self.slots]

    def holding(self):
        return any(sl.alive() for sl in self.slots)

    def pending(self):
        import radio
        with lock:
            for j in jobs.values():           # radio work never starts (or re-takes the GPUs) while the radio is off
                if j["status"] == "queued" and j.get("source") == "radio" and not radio.generating():
                    j.update(status="cancelled", stage="", finished=time.time(), radio_status="cancelled"); save_job(j)
            return sorted((j for j in jobs.values() if j["status"] == "queued"), key=lambda j: (j.get("priority", 0), j["created"]))

    def run(self):
        import radio
        if self.paused_llm and not llm_active() and radio.generating():
            log("radio resumed after a quick restart: leaving the LLM paused")
        elif self.paused_llm and not llm_active():
            log("recovering: restarting the LLM that a previous studio run paused")
            sh("sudo", "systemctl", "start", LLM_UNIT)
            self.paused_llm = False; write_gpu_state(False)
        while True:
            try:
                self.step()
            except Exception:
                log("worker error:", traceback.format_exc())
                time.sleep(3)

    def step(self):
        pend = self.pending()
        busy = any(sl.job for sl in self.slots)
        if pend and not self.holding():
            self.acquire()
        if pend and self.holding():
            for sl in self.slots:                 # bring back crashed workers while a session is active
                if not sl.alive() and not sl.disabled and (not sl.restarts or time.time() - sl.restarts[-1] > 20):
                    self.heal(sl, "worker process is not running")
        # hand queued jobs to idle, loaded slots
        for sl in self.slots:
            if sl.alive() and sl.ready and sl.job is None and pend:
                job = pend.pop(0)
                with lock:
                    job.update(status="running", stage="Starting", device=sl.device)
                sl.job = job
                sl.task_q.put({k: v for k, v in job.items() if not k.startswith("_")})
                self.set_gpu("studio", f"{sum(1 for x in self.slots if x.job)} of {len(self.slots)} GPUs rendering")
        self.pump()
        busy = any(sl.job for sl in self.slots)
        with lock:
            if busy or pend:
                gpu["idle_since"] = None
            elif self.holding() and gpu["idle_since"] is None:
                gpu["idle_since"] = time.time()
            want_release = self.holding() and not busy and not pend and (
                gpu["release_requested"] or gpu["release_on_idle"] or
                (RETURN_MINUTES > 0 and gpu["idle_since"] and time.time() - gpu["idle_since"] > RETURN_MINUTES * 60))
        if want_release:
            self.release()
        elif not self.holding():
            self.set_gpu("llm" if llm_active() else "free")
            with lock:
                gpu["release_requested"] = gpu["release_on_idle"] = False
            time.sleep(1)
        elif not busy:
            self.set_gpu("studio", "YuE2 loaded — idle")
            time.sleep(0.5)

    def pump(self):
        """Drain progress events from every slot (and notice crashed slots)."""
        deadline = time.time() + 0.5
        while time.time() < deadline:
            got = False
            for sl in self.slots:
                if sl.proc is None:
                    continue
                if sl.job and sl.job.get("_cancel"):
                    sl.cancel_ev.set()
                try:
                    kind, jid, data = sl.event_q.get_nowait()
                except queue.Empty:
                    if not sl.alive() and sl.proc is not None:
                        if sl.job:
                            with lock:
                                sl.job.update(status="failed", stage="", finished=time.time(), progress={},
                                              error="The GPU process crashed while rendering this song")
                            save_job(sl.job); self.finished(sl.job)
                        log("GPU process on", sl.device, "exited unexpectedly"); sl.proc, sl.ready, sl.job = None, False, None
                    continue
                got = True
                if kind == "ready":
                    sl.ready = True; log("YuE2 ready on", sl.device)
                elif kind == "fatal":
                    log("YuE2 failed to load on", sl.device, data.get("error")); sl.stop()
                elif sl.job and jid == sl.job["id"]:
                    if kind == "update":
                        with lock:
                            sl.job.update(data)
                        if "status" in data:
                            save_job(sl.job)
                    elif kind == "finished":
                        save_job(sl.job); done = sl.job; sl.job = None
                        self.last_activity = time.time(); self.finished(done)
                        err = done.get("error") or ""
                        sl.fails = sl.fails + 1 if done["status"] == "failed" else 0
                        if "DEVICE_LOST" in err or "level_zero backend failed" in err or sl.fails >= 3:
                            self.heal(sl, err or f"{sl.fails} failures in a row")
            if not got:
                time.sleep(0.05)

    def heal(self, sl, why):
        """A GPU context that hit a device fault stays dead; replace the worker process (a fresh one works after the
        driver's engine reset). Too many restarts -> retire that card for this session."""
        now = time.time()
        sl.restarts = [t for t in sl.restarts if now - t < 600] + [now]
        log(f"GPU worker {sl.device} unhealthy ({why[:120]}); restart #{len(sl.restarts)} in 10 min")
        sl.stop(); sl.fails = 0
        if len(sl.restarts) > 3:
            sl.disabled = True
            log(f"GPU worker {sl.device} retired for this session after repeated faults")
        else:
            sl.start()

    def finished(self, job):
        if job.get("source") == "radio":
            import radio
            threading.Thread(target=radio.judge, args=(job,), daemon=True).start()

    def wait_vram(self, index, below_mib, timeout, what):
        for _ in range(timeout):
            used = gpu_free_mib(index)
            if used is not None and used < below_mib:
                return True
            time.sleep(1)
        log(f"timeout waiting for {what} on xpu:{index}; VRAM used {gpu_free_mib(index)} MiB")
        return False

    def acquire(self):
        if llm_active():
            self.set_gpu("switching", "Pausing the LLM to free the GPUs…")
            log("stopping", LLM_UNIT)
            sh("sudo", "systemctl", "stop", LLM_UNIT)
            self.paused_llm = True; write_gpu_state(True)
        self.set_gpu("switching", "Waiting for GPU memory…")
        for sl in self.slots:
            self.wait_vram(sl.index, 16 * 1024, 120, "the LLM to free GPU memory")
        self.set_gpu("switching", "Loading YuE2…")
        if all(sl.disabled for sl in self.slots):
            for sl in self.slots:
                sl.disabled = False; sl.restarts = []
        for sl in self.slots:
            if not sl.alive() and not sl.disabled:
                sl.start()
        t0 = time.time()
        while not any(sl.ready for sl in self.slots) and time.time() - t0 < 300:
            self.pump()
            if not self.holding():
                raise RuntimeError("Could not start YuE2 on any GPU")
        self.set_gpu("studio", "YuE2 loaded")

    def release(self):
        self.set_gpu("switching", "Returning the GPUs to the LLM…")
        log("releasing GPUs: stopping the YuE2 processes")
        for sl in self.slots:
            sl.stop(); sl.disabled = False; sl.restarts = []; sl.fails = 0
        for sl in self.slots:
            self.wait_vram(sl.index, 1024, 60, "the YuE2 process to free GPU memory")
        with lock:
            gpu["release_requested"] = gpu["release_on_idle"] = False; gpu["idle_since"] = None
        if not llm_active():
            log("starting", LLM_UNIT)
            sh("sudo", "systemctl", "start", LLM_UNIT)
        self.paused_llm = False; write_gpu_state(False)
        self.set_gpu("llm" if llm_active() else "free")


# ----------------------------------------------------------------------------- lyrics (via an LLM)

LYRICS_SYSTEM = """You are a professional songwriter writing lyrics that a singing AI model will perform.
Output format (strict):
- First line: Title: <short song title>
- Then a blank line, then the lyrics.
- Every section starts with a tag on its own line, chosen only from: [Intro] [Verse] [Pre-Chorus] [Chorus] [Bridge] [Outro]. Do not number tags.
- One sung phrase per line, 5-12 syllables (or a similar length in other languages), consistent rhythm within a section. A blank line between sections.
- When the chorus returns, repeat it word for word.
- Only words to be sung. No markdown, no bold, no bullet points, no parenthetical notes, no stage directions, no chords, no explanations before or after.
- Write in the requested language (if none is given, the language named in the style; default English). Match the genre, mood and era of the style."""
LENGTHS = {"short": "about 1 minute: one verse and one chorus", "medium": "about 2 minutes",
           "long": "about 3 to 4 minutes, with fuller verses"}
RHYMES = {"rhyming": "Use clear end rhymes.", "loose": "Use loose, natural rhymes and near-rhymes.", "free": "Free verse: no rhyme scheme needed."}
POVS = {"i": "Write in the first person (I).", "you": "Address the listener directly (you).", "we": "Write as 'we'.",
        "story": "Tell a story in the third person."}
SECTION_RE = re.compile(r"^\W*(intro|verse|pre[- ]?chorus|chorus|bridge|outro|hook|refrain)\s*\d*\W*$", re.I)
ZH_TAGS = {"前奏": "Intro", "主歌": "Verse", "预副歌": "Pre-Chorus", "导歌": "Pre-Chorus", "副歌": "Chorus",
           "桥段": "Bridge", "间奏": "Bridge", "尾奏": "Outro", "结尾": "Outro", "尾声": "Outro"}
TAG_NAMES = {"intro": "Intro", "verse": "Verse", "prechorus": "Pre-Chorus", "chorus": "Chorus", "bridge": "Bridge",
             "outro": "Outro", "hook": "Chorus", "refrain": "Chorus"}


def lyrics_prompt(b):
    action = b.get("action", "write")
    style = clean_text(b.get("style"), "Style", 1500)
    lang = clean_text(b.get("language"), "Language", 40)
    parts = [f"Style: {style or 'any'}"]
    if lang:
        parts.append(f"Language: {lang}")
    if action == "write":
        idea = clean_text(b.get("idea"), "Idea", 2000)
        title = clean_text(b.get("title"), "Title", 120)
        parts.append(f"What the song is about: {idea or 'choose a fitting theme for this style'}")
        if title:
            parts.append(f"Use this title: {title}")
        parts.append(f"Structure: {clean_text(b.get('structure'), 'Structure', 300) or 'Verse, Chorus, Verse, Chorus, Bridge, Chorus'}")
        parts.append(f"Length: {LENGTHS.get(b.get('length'), LENGTHS['medium'])}")
        parts.append(RHYMES.get(b.get("rhyme"), RHYMES["rhyming"]))
        if b.get("pov") in POVS:
            parts.append(POVS[b["pov"]])
        if b.get("extra"):
            parts.append("Also: " + clean_text(b.get("extra"), "Extra instructions", 1000))
        parts.append("Write the song.")
    else:
        current = clean_text(b.get("lyrics"), "Lyrics", 20000, required=True)
        if action == "rewrite":
            parts.append("Revise these lyrics according to this feedback: " + (clean_text(b.get("feedback"), "Feedback", 1000) or "make them stronger and more singable"))
            parts.append("Keep what already works; return the complete revised song.")
        elif action == "polish":
            parts.append("Format these user-written lyrics for singing. Keep the user's words and meaning as much as possible: "
                         "add or fix section tags, split overly long lines into singable phrases, make repeated choruses identical, "
                         "and only lightly smooth the rhythm. Keep the existing title if there is one.")
        else:
            raise BadRequest("action must be write, rewrite or polish")
        parts.append("Lyrics:\n" + current)
    return "\n".join(parts)


def normalize_lyrics(text):
    """Pull out 'Title:' and coerce the model's text into YuE2's tag/line format."""
    title, out = "", []
    for raw in text.replace("\r\n", "\n").split("\n"):
        line = raw.strip().strip("`").replace("**", "").replace("__", "").strip()
        if not line:
            if out and out[-1] != "":
                out.append("")
            continue
        m = re.match(r"^(?:#+\s*)?(?:title|标题|歌名|曲名)\s*[:：]\s*(.+)$", line, re.I)
        if m and not title and not any(l for l in out if l):
            title = m.group(1).strip().strip('"“”')
            continue
        if line.startswith("#"):
            line = line.lstrip("#").strip()
        bare = line.strip("[]()（）【】: ").strip()
        sm = SECTION_RE.match(bare)
        zh = ZH_TAGS.get(re.sub(r"[\d一二三四五六七八九十\s]+$", "", bare))
        if sm or zh:
            name = zh or TAG_NAMES[re.sub(r"[- ]", "", sm.group(1).lower())]
            if out and out[-1] != "":
                out.append("")
            out.append(f"[{name}]")
            continue
        line = re.sub(r"^\s*[-*•]\s+", "", line)
        line = re.sub(r"\s*\((?:x\d+|repeat|×\d+)\)\s*$", "", line, flags=re.I)
        out.append(line)
    while out and out[-1] == "":
        out.pop()
    lyrics = "\n".join(out)
    lyrics = re.sub(r"\]\n\n", "]\n", lyrics)
    return title, lyrics


def pick_llm():
    for base, model, label in LYRICS_LLMS:
        try:
            with urllib.request.urlopen(base.rstrip("/") + "/models", timeout=2) as r:
                if r.status == 200:
                    return base.rstrip("/"), model, label
        except Exception:
            continue
    return None


def stream_lyrics(handler, body):
    prompt = lyrics_prompt(body)            # validates before we commit to a stream
    llm = pick_llm()
    if llm is None:
        return handler.send(503, {"error": "No lyrics writer is reachable right now. b70's LLM is paused while the studio "
                                           "holds the GPUs, and the backup server did not answer."})
    base, model, label = llm
    try:
        temperature = min(1.5, max(0.2, float(body.get("temperature", 0.85))))
    except Exception:
        temperature = 0.85
    req = urllib.request.Request(base + "/chat/completions", json.dumps({
        "model": model, "stream": True, "temperature": temperature, "top_p": 0.95, "max_tokens": 1600,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "system", "content": LYRICS_SYSTEM}, {"role": "user", "content": prompt}]}).encode(),
        {"Content-Type": "application/json"})
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream"); handler.send_header("Cache-Control", "no-store")
    handler.end_headers()

    def emit(obj):
        handler.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n"); handler.wfile.flush()

    t0, text = time.time(), []
    try:
        emit({"source": label})
        with urllib.request.urlopen(req, timeout=300) as r:
            for line in r:
                line = line.strip()
                if not line.startswith(b"data: ") or line == b"data: [DONE]":
                    continue
                d = json.loads(line[6:])
                piece = (d.get("choices") or [{}])[0].get("delta", {}).get("content") or ""
                if piece:
                    text.append(piece); emit({"delta": piece})
        title, lyrics = normalize_lyrics("".join(text))
        emit({"done": True, "title": title, "lyrics": lyrics, "source": label, "seconds": round(time.time() - t0, 1)})
        log("lyrics", body.get("action", "write"), f"via {label}", f"{time.time() - t0:.1f}s")
    except (BrokenPipeError, ConnectionResetError):
        pass
    except Exception as e:
        try:
            emit({"error": f"The lyrics writer failed: {type(e).__name__}: {e}"[:300]})
        except Exception:
            pass


# ----------------------------------------------------------------------------- HTTP

def summary(job):
    keys = ("id", "batch", "take", "takes", "title", "author", "mode", "style", "cot", "seed", "status", "stage",
            "progress", "created", "started", "finished", "seconds", "gen_seconds", "error", "cfg_scale", "max_seconds")
    s = {k: job.get(k) for k in keys}
    d = DATA / job["id"]
    s["files"] = [name for name, (rel, _) in FILES.items() if (d / rel).is_file()]
    return s


def state():
    with lock:
        allj = sorted((j for j in jobs.values() if j.get("source") != "radio"), key=lambda j: -j["created"])
        queue = sorted((j for j in allj if j["status"] in ("queued", "running")), key=lambda j: j["created"])
        library = [j for j in allj if j["status"] in ("done", "failed", "cancelled")]
        idle_left = None
        if RETURN_MINUTES > 0 and gpu["owner"] == "studio" and gpu["idle_since"] and not queue:
            idle_left = max(0, RETURN_MINUTES * 60 - (time.time() - gpu["idle_since"]))
        import radio
        return {"gpu": dict(owner=gpu["owner"], detail=gpu["detail"], llm_unit=LLM_UNIT, return_minutes=RETURN_MINUTES,
                            idle_seconds_left=idle_left, release_requested=gpu["release_requested"], workers=gpu["workers"],
                            radio_on=radio.state["on"], radio_generating=radio.generating()),
                "queue": [summary(j) for j in queue], "library": [summary(j) for j in library[:300]],
                "limits": {"max_queued": MAX_QUEUED}, "lyrics_writers": [label for _, _, label in LYRICS_LLMS]}


class Handler(BaseHTTPRequestHandler):
    server_version = "b70-music-studio"

    def send(self, code, body, ctype="application/json", extra=None):
        b = (json.dumps(body) if ctype == "application/json" else body)
        b = b.encode() if isinstance(b, str) else b
        self.send_response(code)
        self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers(); self.wfile.write(b)

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > 200_000:
            raise BadRequest("Request too large")
        return json.loads(self.rfile.read(n) or b"{}")

    def job_or_404(self, jid):
        with lock:
            job = jobs.get(jid)
        if not job:
            self.send(404, {"error": "No such song"})
        return job

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            return self.send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path in ("/radio", "/radio.html"):
            return self.send(200, (HERE / "radio.html").read_bytes(), "text/html; charset=utf-8")
        if path in ("/radio/stream.mp3", "/radio/stream", "/stream.mp3"):
            import radio
            return radio.stream(self)
        if path.startswith("/api/radio/"):
            import radio
            if radio.handle(self, "GET", path):
                return
        if path == "/api/state":
            return self.send(200, state())
        if path == "/api/health":
            st = state()
            return self.send(200, {"ok": True, "gpu_owner": st["gpu"]["owner"], "queued": len(st["queue"]),
                                   "running": next((j["title"] for j in st["queue"] if j["status"] == "running"), None),
                                   "songs": sum(1 for j in st["library"] if j["status"] == "done"), "radio_on": st["gpu"]["radio_on"]})
        m = re.fullmatch(r"/api/jobs/([\w-]+)", path)
        if m:
            job = self.job_or_404(m.group(1))
            if job:
                full = {k: v for k, v in job.items() if not k.startswith("_")}
                full["files"] = summary(job)["files"]
                for name in ("score.abc", "source.abc"):
                    f = DATA / job["id"] / FILES[name][0]
                    full[name.replace(".", "_")] = f.read_text(encoding="utf-8") if f.is_file() else None
                self.send(200, full)
            return
        m = re.fullmatch(r"/files/([\w-]+)/([\w.]+)", path)
        if m and m.group(2) in FILES:
            return self.file(m.group(1), m.group(2))
        self.send(404, {"error": "not found"})

    def file(self, jid, name):
        job = self.job_or_404(jid)
        if not job:
            return
        rel, ctype = FILES[name]
        root = DATA.resolve()
        f = (DATA / jid / rel).resolve()
        if not str(f).startswith(str(root) + os.sep) or not f.is_file():
            return self.send(404, {"error": "file not found"})
        size = f.stat().st_size
        safe = re.sub(r"[^\w.-]+", "-", job.get("title") or jid).strip("-")[:60] or jid
        ext = name.rsplit(".", 1)[1]
        extra = {"Accept-Ranges": "bytes"}
        if "download=1" in self.path:
            extra["Content-Disposition"] = f'attachment; filename="{safe}-take{job.get("take", 1)}.{ext}"'
        rng = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        with open(f, "rb") as fh:
            if rng and (rng.group(1) or rng.group(2)):
                a = int(rng.group(1)) if rng.group(1) else max(0, size - int(rng.group(2)))
                b = min(int(rng.group(2)), size - 1) if rng.group(1) and rng.group(2) else size - 1
                fh.seek(a); data = fh.read(b - a + 1)
                extra["Content-Range"] = f"bytes {a}-{b}/{size}"
                return self.send(206, data, ctype, extra)
            return self.send(200, fh.read(), ctype, extra)

    def do_PUT(self):
        path = self.path.split("?")[0]
        import radio
        if not (path.startswith("/api/radio/") and radio.handle(self, "PUT", path)):
            self.send(404, {"error": "not found"})

    def do_POST(self):
        path = self.path.split("?")[0]
        if path.startswith("/api/radio/"):
            import radio
            if radio.handle(self, "POST", path):
                return
        try:
            if path == "/api/jobs":
                return self.send(200, {"ids": new_jobs(self.body())})
            if path == "/api/lyrics":
                return stream_lyrics(self, self.body())
            m = re.fullmatch(r"/api/jobs/([\w-]+)/cancel", path)
            if m:
                job = self.job_or_404(m.group(1))
                if job:
                    with lock:
                        if job["status"] == "queued":
                            job.update(status="cancelled", stage="", finished=time.time()); save_job(job)
                        elif job["status"] == "running":
                            job["_cancel"] = True; job["stage"] = "Cancelling…"
                    self.send(200, {"ok": True})
                return
            if path == "/api/gpu/release":
                import radio
                if radio.generating():
                    radio.set_power(False)          # otherwise the radio would take the GPUs straight back (reruns can stay on)
                with lock:
                    busy = any(j["status"] in ("queued", "running") and j.get("source") != "radio" for j in jobs.values())
                    if not busy:
                        gpu["release_requested"] = True
                return self.send(200, {"ok": not busy, "message": "Songs are still queued" if busy else "Returning the GPUs"})
        except BadRequest as e:
            return self.send(400, {"error": str(e)})
        except json.JSONDecodeError:
            return self.send(400, {"error": "Invalid JSON"})
        self.send(404, {"error": "not found"})

    def do_DELETE(self):
        if self.path.startswith("/api/radio/"):
            import radio
            if radio.handle(self, "DELETE", self.path.split("?")[0]):
                return
        m = re.fullmatch(r"/api/jobs/([\w-]+)", self.path.split("?")[0])
        if not m:
            return self.send(404, {"error": "not found"})
        job = self.job_or_404(m.group(1))
        if not job:
            return
        if job["status"] in ("queued", "running"):
            return self.send(400, {"error": "Cancel it first"})
        with lock:
            jobs.pop(job["id"], None)
        shutil.rmtree(DATA / job["id"], ignore_errors=True)
        log("deleted", job["id"])
        self.send(200, {"ok": True})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    load_library()
    sys.modules.setdefault("app", sys.modules["__main__"])   # radio.py imports the studio as `app`
    import radio
    radio.init(sys.modules["__main__"])
    Worker().start()
    log(f"Music Studio on :{PORT}, data {DATA}, devices {DEVICES}, returns GPUs after {RETURN_MINUTES} min idle")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
