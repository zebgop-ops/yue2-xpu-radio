"""b70 Radio: infinite AI radio stations for the Music Studio (inspired by CharlesMod/infinite-tapedeck).

Stations are text-defined ("late-night lo-fi") or learned from a folder of songs.
While the radio is on, a spool manager keeps a few approved songs ready per
station: an LLM writes each song's spec (title, style, lyrics) from the station
description, its learned profile and listener feedback; YuE2 renders it through
the studio's GPU pool; a CLAP critic scores the take against the station (its
songs, or its description) and drops the weakest. Keep / skip / dislike feedback
steers both the prompts and the critic.

Everything runs on the CPU here (CLAP, librosa); the GPUs stay with YuE2.
"""
import json, os, queue, random, re, shutil, subprocess, threading, time, traceback, urllib.request, uuid
from pathlib import Path

import numpy as np

app = None                          # the studio module, set by init()
RADIO = None
lock = threading.RLock()
stations: dict[str, dict] = {}
state = {"on": False, "station": None, "target_ahead": 4, "max_inflight": 2}
writing = {}                        # station id -> number of specs being written right now
backoff = {}                        # station id -> {"until": t, "delay": s, "reason": str}
AUDIO_EXT = {".mp3", ".flac", ".wav", ".ogg", ".m4a", ".aac", ".opus"}
KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# CLAP tag vocabulary: turns a folder of songs into words for the style prompt.
TAGS = {
    "genre": ["pop", "rock", "indie rock", "alternative rock", "punk", "metal", "folk", "country", "blues", "jazz", "soul",
              "R&B", "funk", "disco", "hip hop", "rap", "trap", "lo-fi hip hop", "electronic", "house", "techno", "trance",
              "drum and bass", "dubstep", "synthwave", "ambient", "downtempo", "trip hop", "classical", "orchestral",
              "cinematic soundtrack", "piano ballad", "reggae", "latin", "bossa nova", "k-pop", "j-pop", "gospel", "chiptune"],
    "mood": ["happy", "uplifting", "energetic", "aggressive", "dark", "melancholic", "sad", "romantic", "calm", "relaxing",
             "dreamy", "nostalgic", "epic", "playful", "mysterious", "hopeful", "angry", "peaceful"],
    "instrument": ["acoustic guitar", "electric guitar", "distorted guitar", "piano", "electric piano", "synthesizer",
                   "synth pads", "strings", "violin", "cello", "brass", "saxophone", "trumpet", "flute", "bass guitar",
                   "808 bass", "drum machine", "live drums", "organ", "harp", "ukulele", "banjo"],
}
TEMPLATES = {"genre": ["This is a {} song.", "{} music", "a {} track"],
             "mood": ["{} music", "a {} song", "music that feels {}"],
             "instrument": ["music featuring {}", "a song with {}", "{} playing"]}
VOCAL = ["a song with a singer singing lyrics", "vocals singing a melody", "a pop song with lead vocals", "a person singing"]
INSTRUMENTAL = ["an instrumental track with no vocals", "instrumental music without singing", "background music with no voice", "an instrumental piece"]
FEMALE, MALE = ["a female singer", "a woman singing"], ["a male singer", "a man singing"]


def init(studio_module):
    global app, RADIO
    app = studio_module
    RADIO = app.DATA.parent / "radio"
    (RADIO / "stations").mkdir(parents=True, exist_ok=True)
    try:
        state.update(json.loads((RADIO / "state.json").read_text()))
    except Exception:
        pass
    # resume only after a quick service restart (studio was alive < 2 min ago); after a reboot the radio starts off
    state["on"] = bool(state.get("on")) and time.time() - (state.get("heartbeat") or 0) < 120 and state.get("station") is not None
    if state["on"]:
        app.log("radio: resuming the broadcast after a quick restart")
    for f in (RADIO / "stations").glob("*/station.json"):
        try:
            s = json.loads(f.read_text())
            if s.get("analysis", {}).get("status") == "running":
                s["analysis"]["status"] = "interrupted"
            stations[s["id"]] = s
        except Exception:
            pass
    if not stations:
        for name, desc, vocals in [
                ("Late Night Lo-fi", "Mellow late-night lo-fi hip hop and chillhop: dusty piano, warm Rhodes, soft boom-bap drums, rain on the window, mostly instrumental with the occasional soft vocal.", "mix"),
                ("Indie Road Trip", "Sunny indie rock and indie folk for driving with the windows down: jangly guitars, big singalong choruses, male and female vocals, nostalgic and hopeful.", "vocals"),
                ("Neon Nights", "80s-inspired synthwave and synth-pop: analog synths, gated drums, arpeggiated bass, dreamy vocals, neon city at night.", "mix")]:
            create_station({"name": name, "kind": "text", "description": desc, "vocals": vocals})
    save_state()
    threading.Thread(target=spool_loop, daemon=True, name="radio-spool").start()
    threading.Thread(target=playout_loop, daemon=True, name="radio-playout").start()


def save_state():
    state["heartbeat"] = time.time()
    (RADIO / "state.json").write_text(json.dumps({k: state.get(k) for k in ("station", "target_ahead", "on", "heartbeat")}))


def sdir(sid):
    return RADIO / "stations" / sid


def save_station(s):
    d = sdir(s["id"]); d.mkdir(parents=True, exist_ok=True)
    tmp = d / "station.json.tmp"; tmp.write_text(json.dumps(s, indent=1)); tmp.replace(d / "station.json")


def create_station(body):
    name = app.clean_text(body.get("name"), "Station name", 60, required=True)
    kind = body.get("kind", "text")
    if kind not in ("text", "folder"):
        raise app.BadRequest("kind must be text or folder")
    desc = app.clean_text(body.get("description"), "Description", 2000, required=(kind == "text"))
    vocals = body.get("vocals", "mix")
    if vocals not in ("vocals", "instrumental", "mix"):
        raise app.BadRequest("vocals must be vocals, instrumental or mix")
    s = {"id": uuid.uuid4().hex[:10], "name": name, "kind": kind, "description": desc, "vocals": vocals,
         "language": app.clean_text(body.get("language"), "Language", 40), "created": time.time(),
         "stats": {"generated": 0, "approved": 0, "rejected": 0, "played": 0, "kept": 0, "skipped": 0, "disliked": 0},
         "feedback": [], "scores": [], "analysis": {"status": "none" if kind == "folder" else "n/a"}, "profile": None}
    with lock:
        stations[s["id"]] = s
        save_station(s)
    if kind == "folder":
        (sdir(s["id"]) / "music").mkdir(parents=True, exist_ok=True)
    return s


def delete_station(sid):
    with lock:
        s = stations.pop(sid, None)
        if state["station"] == sid:
            state["station"] = None; state["on"] = False
    if s:
        with app.lock:
            for j in list(app.jobs.values()):
                if j.get("station") == sid:
                    if j["status"] == "queued":
                        j.update(status="cancelled", finished=time.time())
                    elif j["status"] == "running":
                        j["_cancel"] = True
        shutil.rmtree(sdir(sid), ignore_errors=True)
    save_state()


# ----------------------------------------------------------------------------- CLAP + listener (CPU)

_clap = {"model": None, "proc": None, "lock": threading.Lock()}


def clap():
    with _clap["lock"]:
        if _clap["model"] is None:
            import torch
            from transformers import ClapModel, ClapProcessor
            torch.set_num_threads(8)
            _clap["model"] = ClapModel.from_pretrained("laion/clap-htsat-unfused").eval()
            _clap["proc"] = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        return _clap["model"], _clap["proc"]


def _norm(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def audio_embedding(path, max_windows=6):
    """Mean of CLAP embeddings over up to 6 x 10 s windows from the middle of the track; also returns the mono audio."""
    import librosa, torch
    y, sr = librosa.load(str(path), sr=48000, mono=True)
    n = 480000
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    starts = list(range(0, len(y) - n + 1, n))
    if len(starts) > max_windows:
        mid = len(starts) // 2; starts = starts[max(0, mid - max_windows // 2):][:max_windows]
    model, proc = clap()
    with _clap["lock"], torch.inference_mode():
        a = model.get_audio_features(**proc(audio=[y[s:s + n] for s in starts], sampling_rate=48000, return_tensors="pt"))
    return _norm(_norm(a.numpy()).mean(0)), y


_text_cache = {}


def text_embeddings(texts):
    import torch
    missing = [t for t in texts if t not in _text_cache]
    if missing:
        model, proc = clap()
        with _clap["lock"], torch.inference_mode():
            e = model.get_text_features(**proc(text=missing, return_tensors="pt", padding=True)).numpy()
        for t, v in zip(missing, _norm(e)):
            _text_cache[t] = v
    return np.stack([_text_cache[t] for t in texts])


def features(y):
    import librosa
    y22 = librosa.resample(y, orig_sr=48000, target_sr=22050)
    tempo, _ = librosa.beat.beat_track(y=y22, sr=22050)
    tempo = float(np.atleast_1d(tempo)[0])
    while tempo > 150: tempo /= 2          # librosa often reports double time
    while tempo and tempo < 60: tempo *= 2
    chroma = librosa.feature.chroma_cqt(y=y22, sr=22050).mean(1)
    rms = librosa.feature.rms(y=y22)[0]
    cent = librosa.feature.spectral_centroid(y=y22, sr=22050)[0]
    return {"bpm": round(tempo), "key": KEYS[int(chroma.argmax())], "energy": float(np.sqrt(np.mean(rms ** 2))),
            "brightness": float(np.median(cent)), "seconds": round(len(y) / 48000, 1)}


def tag_scores(emb):
    """Prompt-ensembled CLAP similarities, centred within each category (removes per-category text bias)."""
    out = {}
    for cat, words in TAGS.items():
        sims = np.mean([text_embeddings([t.format(w) for w in words]) @ emb for t in TEMPLATES[cat]], axis=0)
        out[cat] = dict(zip(words, (sims - sims.mean()).tolist()))
    vocal_margin = float((text_embeddings(VOCAL) @ emb).mean() - (text_embeddings(INSTRUMENTAL) @ emb).mean())
    out["p_vocal"] = float(1 / (1 + np.exp(-(vocal_margin + 0.04) * 60)))   # calibrated on known vocal/instrumental takes
    out["female_margin"] = float((text_embeddings(FEMALE) @ emb).mean() - (text_embeddings(MALE) @ emb).mean())
    return out


def analyze_station(sid):
    s = stations[sid]
    music = sdir(sid) / "music"
    files = sorted(p for p in music.rglob("*") if p.suffix.lower() in AUDIO_EXT)
    s["analysis"] = {"status": "running", "done": 0, "total": len(files), "started": time.time(), "errors": 0}
    save_station(s)
    tracks, embs = [], []
    for i, f in enumerate(files):
        try:
            emb, y = audio_embedding(f)
            t = dict(features(y), file=str(f.relative_to(music)), tags=tag_scores(emb))
            tracks.append(t); embs.append(emb)
        except Exception as e:
            s["analysis"]["errors"] += 1
            app.log("radio: analysis failed for", f, e)
        s["analysis"]["done"] = i + 1
    if not embs:
        s["analysis"].update(status="failed", error="No readable audio files"); save_station(s)
        return
    E = np.stack(embs)
    np.save(sdir(sid) / "corpus.npy", E)
    sims = E @ E.T; np.fill_diagonal(sims, np.nan)
    nn = np.nanmax(sims, axis=1) if len(E) > 1 else np.array([1.0])
    agg = {cat: {w: float(np.mean([t["tags"][cat][w] for t in tracks])) for w in TAGS[cat]} for cat in TAGS}
    top = {cat: [w for w, v in sorted(agg[cat].items(), key=lambda x: -x[1])[:5] if v > 0] for cat in TAGS}
    bpms = [t["bpm"] for t in tracks if t["bpm"]]
    vocal = float(np.mean([t["tags"]["p_vocal"] for t in tracks]))
    sung = [t["tags"]["female_margin"] for t in tracks if t["tags"]["p_vocal"] > 0.5]
    fem = float(np.mean([m > 0 for m in sung])) if sung else None
    top["voice"] = ([] if vocal < 0.3 else ["female vocals"] if fem is not None and fem > 0.65 else
                    ["male vocals"] if fem is not None and fem < 0.35 else ["female and male vocals"]) or ["instrumental"]
    keyc = {}
    for t in tracks:
        keyc[t["key"]] = keyc.get(t["key"], 0) + 1
    s["profile"] = {"tracks": len(tracks), "top": top, "weights": agg, "bpm_median": int(np.median(bpms)) if bpms else None,
                    "bpm_range": [int(np.percentile(bpms, 20)), int(np.percentile(bpms, 80))] if bpms else None,
                    "keys": sorted(keyc.items(), key=lambda x: -x[1])[:4], "vocal_share": round(vocal, 2),
                    "corpus_nn_p10": float(np.percentile(nn, 10)), "corpus_nn_median": float(np.median(nn)),
                    "track_list": [{k: t[k] for k in ("file", "bpm", "key", "seconds")} for t in tracks]}
    if s.get("vocals") == "mix" and vocal < 0.25:
        s["vocals"] = "instrumental"
    s["analysis"].update(status="done", finished=time.time())
    save_station(s)
    app.log("radio: analyzed", s["name"], len(tracks), "tracks", top)


def profile_text(s):
    p = s.get("profile")
    if not p:
        return ""
    t = p["top"]
    parts = [f"Genres heard in the station's songs: {', '.join(t['genre'])}.", f"Moods: {', '.join(t['mood'])}.",
             f"Instruments: {', '.join(t['instrument'])}."]
    parts.append("Vocals: mostly instrumental." if p["vocal_share"] < 0.3 else f"Vocals: {', '.join(t['voice'])} on about {round(p['vocal_share'] * 100)}% of songs.")
    if p.get("bpm_range"):
        parts.append(f"Tempo mostly {p['bpm_range'][0]}-{p['bpm_range'][1]} BPM.")
    return " ".join(parts)


# ----------------------------------------------------------------------------- song specs (LLM)

SPEC_SYSTEM = """You are the program director and songwriter of an AI radio station. Each time you are asked, invent ONE new song for the station.
Reply with a single JSON object and nothing else:
{"title": "...", "instrumental": true|false, "style": "...", "lyrics": "..."}
- style: one line for a music model: language, genre/subgenre, vocal type (or 'instrumental, no vocals'), 3-5 instruments, mood, and tempo as 'NN BPM'. Example: "English, dreamy synth-pop, airy female vocal, analog synth pads, arpeggiated bass, gated drums, nostalgic, 108 BPM"
- lyrics (when not instrumental): sections tagged [Verse] [Pre-Chorus] [Chorus] [Bridge] [Outro] on their own lines, one sung phrase per line, chorus repeated word for word. Only sung words. Size it for about 2.5-3 minutes: 24-34 sung lines in total (e.g. Verse 6, Chorus 4, Verse 6, Chorus 4, Bridge 4, Chorus 4), and always finish with a short [Outro] of 2-3 lines so the song comes to a real ending.
- lyrics (instrumental): only section tags, e.g. "[Intro]\\n\\n[Verse]\\n\\n[Chorus]\\n\\n[Verse]\\n\\n[Chorus]\\n\\n[Outro]".
- Stay inside the station's sound, but vary subgenre, tempo, key feel, instrumentation, topic and title from song to song so the station never sounds repetitive."""


VARIATIONS = [
    "pick a noticeably different tempo from the recent songs", "explore a different subgenre within the station's sound",
    "change the vocal approach (different voice type, delivery or harmonies)", "write about a completely different topic",
    "feature a different lead instrument", "make it more upbeat than the last few", "make it more laid-back than the last few",
    "use a different song structure (e.g. start with the chorus, or add a bridge)", "shift the decade or regional flavour slightly",
    "give it a storytelling lyric about a specific character or place"]


def parse_json_reply(text):
    """Pull a JSON object out of a chat reply (no JSON mode: xgrammar + MTP speculative decoding makes some vLLM builds return 500s)."""
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        candidates = [m.group(0)]
    elif "{" in text:                     # the model sometimes stops mid-lyrics without closing the string/object
        tail = text[text.index("{"):].rstrip()
        candidates = [tail + "}", tail + '"}', tail[:tail.rindex('"') + 1] + "}"]
    else:
        raise ValueError(f"no JSON object in reply: {text[:80]!r}")
    for raw in candidates:
        for fix in (raw, re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", raw)):        # escape stray backslashes
            try:
                return json.loads(fix, strict=False)              # strict=False tolerates raw newlines in strings
            except json.JSONDecodeError:
                pass
    raise ValueError(f"unparseable JSON in reply: {text[:80]!r}")


TITLE_STOP = {"the", "a", "an", "of", "and", "in", "on", "my", "your", "to", "for", "&", "it", "is", "me", "got", "made",
              "at", "with", "from", "by", "we", "you", "i", "i'm", "our", "all", "no", "so", "up", "down", "out"}


def title_words(t):
    return {w.rstrip("s") for w in re.findall(r"[a-z0-9']+", t.lower()) if w not in TITLE_STOP and len(w) > 2}


def overused_words(titles, min_count=2):
    """Title words the songwriter keeps reaching for (e.g. 'Paper' six times in a row)."""
    counts = {}
    for t in titles:
        for w in title_words(t):
            counts[w] = counts.get(w, 0) + 1
    return sorted(w for w, n in counts.items() if n >= min_count)


def similar_title(title, recent_titles, last=()):
    """True when a title remixes a recent one: shares 2+ content words with any recent title (or all of a short one),
    or shares any content word with the last few titles, or uses a word that is already overused."""
    w = title_words(title)
    for t in recent_titles:
        o = title_words(t)
        if o and w and len(w & o) >= min(2, len(w), len(o)):
            return True
    if any(w & title_words(t) for t in last):
        return True
    return bool(w & set(overused_words(recent_titles)))


def reference_tracks(s, limit=15):
    """Song names from a learned station's files - the LLM usually knows these artists far better than CLAP's tags."""
    d = sdir(s["id"]) / "music"
    if s["kind"] != "folder" or not d.exists():
        return []
    out = []
    for f in sorted(d.rglob("*")):
        if f.suffix.lower() not in AUDIO_EXT:
            continue
        n = re.sub(r"^[\d\s.\-_]+", "", f.stem)                      # leading track numbers
        n = re.sub(r"[-_ ](fb|hq|official|audio|lyrics|320|128)$", "", n, flags=re.I)
        n = re.sub(r"\s+", " ", n.replace("_", " ")).strip(" -")
        if "-" in n and " - " not in n:
            n = n.replace("-", " - ", 1)
        if n and n.lower() not in (x.lower() for x in out):
            out.append(n.title() if n.islower() else n)
        if len(out) >= limit:
            break
    return out


def llm_candidates():
    """Reachable LLMs in order. While the radio is on, b70's own LLM is (or is about to be) paused for the
    GPUs, so the other servers go first."""
    cands = list(app.LYRICS_LLMS)
    if state["on"] or app.gpu["owner"] != "llm":
        cands.sort(key=lambda c: "localhost" in c[0] or "127.0.0.1" in c[0])
    out = []
    for base, model, label in cands:
        try:
            with urllib.request.urlopen(base.rstrip("/") + "/models", timeout=2) as r:
                if r.status == 200:
                    out.append((base.rstrip("/"), model, label))
        except Exception:
            pass
    return out


def write_spec(s, short=False):
    want_inst = s["vocals"] == "instrumental" or (s["vocals"] == "mix" and random.random() < 0.4)
    recent = [j for j in app.jobs.values() if j.get("station") == s["id"] and j.get("source") == "radio"
              and j.get("radio_status") in ("approved", "played", "rendering", "judging")]
    recent = sorted(recent, key=lambda j: -j["created"])[:14]
    attempts = sorted((j for j in app.jobs.values() if j.get("station") == s["id"] and j.get("source") == "radio"
                       and j.get("spec_writer") != "fallback"), key=lambda j: -j["created"])[:20]
    attempt_titles = [j["title"] for j in attempts]
    likes = [f for f in s["feedback"] if f["kind"] == "keep"][-6:]
    dislikes = [f for f in s["feedback"] if f["kind"] == "dislike"][-6:]
    skips = [f for f in s["feedback"] if f["kind"] == "skip"][-4:]
    u = [f"Station: {s['name']}", f"Station sound: {s['description'] or '(learned from its songs; see below)'}"]
    if s.get("profile"):
        u.append("Learned from the station's own songs (automatic tags, can be wrong): " + profile_text(s))
    refs = reference_tracks(s)
    if refs:
        u.append("The station's own songs are: " + "; ".join(refs) + ". Match the sound of these artists and songs "
                 "(genre, vocal type and gender, instrumentation, era, lyrical themes); where your knowledge of them "
                 "disagrees with the automatic tags, trust your knowledge.")
    if s.get("language"):
        u.append(f"Language for vocals: {s['language']}")
    u.append("This song must be INSTRUMENTAL." if want_inst else "This song has vocals.")
    if short:
        u.append("Make this one SHORT - it opens the broadcast: [Verse] of 4 lines, [Chorus] of 4 lines, [Outro] of 2 lines (10 sung lines total).")
    if recent:
        u.append("Recently played (do not repeat these titles or exact styles): " + "; ".join(f"{j['title']} [{j['style'][:70]}]" for j in recent))
    if likes:
        u.append("The listener liked these. Keep the general vibe that made them work, but this song must be clearly "
                 "DIFFERENT from them: new title (reuse none of their title words), new topic, different groove/tempo or "
                 "instrumentation: " + "; ".join(f"{f['title']} [{f['style'][:80]}]" for f in likes))
    u.append("For variety, this time: " + random.choice(VARIATIONS) + ".")
    banned = overused_words(attempt_titles + [f["title"] for f in likes])
    if attempt_titles:
        u.append("Titles of the latest songs (yours must not reuse any of their words): " + "; ".join(attempt_titles[:10]))
    if banned:
        u.append("These title words are overused - do NOT use them anywhere in the title: " + ", ".join(banned) + ".")
    if dislikes:
        u.append("The listener DISLIKED these, steer away from them: " + "; ".join(f"{f['title']} [{f['style'][:80]}]" for f in dislikes))
    if skips:
        u.append("Skipped (mildly less of this): " + "; ".join(f"[{f['style'][:60]}]" for f in skips))
    for base, model, label in [c for c in llm_candidates() for _ in range(3)]:     # each server gets two retries
        if not state["on"]:
            break
        try:
            body = {"model": model, "temperature": 1.0, "top_p": 0.95, "max_tokens": 2200,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "messages": [{"role": "system", "content": SPEC_SYSTEM}, {"role": "user", "content": "\n".join(u)}]}
            r = json.loads(urllib.request.urlopen(urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(),
                           {"Content-Type": "application/json"}), timeout=180).read())
            text = r["choices"][0]["message"]["content"]
            spec = parse_json_reply(text)
            style = str(spec.get("style", "")).strip()[:600]
            inst = bool(spec.get("instrumental", want_inst))
            ltitle, lyrics = app.normalize_lyrics(str(spec.get("lyrics", "")))
            if not style:
                raise ValueError("spec without a style")
            title = str(spec.get("title") or spec.get("name") or ltitle or "").strip()
            if not title:
                first = next((l for l in lyrics.splitlines() if l.strip() and not l.startswith("[")), "")
                genre = style.split(",")[1].strip().title()[:40] if "," in style else ""
                title = first[:40] or genre or "Untitled"
            recent_titles = attempt_titles + [j["title"] for j in recent[:8]] + [f["title"] for f in likes]
            if similar_title(title, recent_titles, last=attempt_titles[:5]) and not u[-1].startswith("IMPORTANT"):
                u.append(f"IMPORTANT: your title '{title}' repeats words from recent titles. Write a NEW song with a "
                         "completely different title and topic.")
                raise ValueError(f"title '{title}' repeats recent titles; asking again")
            return {"title": title[:80], "style": style, "instrumental": inst, "lyrics": lyrics if not inst else "", "writer": label}
        except Exception as e:
            app.log(f"radio: spec writer {label} failed ({e}); trying the next one")
    words = s["description"] or profile_text(s)
    return {"title": f"{s['name']} #{s['stats']['generated'] + 1}", "style": "Instrumental, " + words[:300] + ", no vocals",
            "instrumental": True, "lyrics": "", "writer": "fallback"}


# ----------------------------------------------------------------------------- spool

def station_tracks(sid, statuses=("approved",)):
    with app.lock:
        return [j for j in app.jobs.values() if j.get("source") == "radio" and j.get("station") == sid and j.get("radio_status") in statuses]


def inflight(sid):
    with app.lock:
        n = sum(1 for j in app.jobs.values() if j.get("source") == "radio" and j.get("station") == sid and
                (j["status"] in ("queued", "running") or j.get("radio_status") == "judging"))
    return n + writing.get(sid, 0)


def spool_loop():
    while True:
        try:
            time.sleep(2)
            if time.time() - state.get("heartbeat", 0) > 20:
                save_state()                               # heartbeat: lets a quick restart resume the broadcast
            sid = state["station"]
            if not state["on"] or sid not in stations:
                continue
            s = stations[sid]
            if s["kind"] == "folder" and s["analysis"].get("status") != "done":
                continue
            b = backoff.get(sid)
            if b and time.time() < b["until"]:
                continue
            with app.lock:
                last = sorted((j for j in app.jobs.values() if j.get("source") == "radio" and j.get("station") == sid
                               and j["status"] in ("done", "failed") and not (j.get("error") or "").startswith("Interrupted")),
                              key=lambda j: -j.get("finished", 0))[:3]
            if len(last) == 3 and all(j["status"] == "failed" for j in last) and not (b and b.get("seen") == last[0]["id"]):
                delay = min(600, (b["delay"] * 2) if b else 60)
                backoff[sid] = {"until": time.time() + delay, "delay": delay, "seen": last[0]["id"],
                                "reason": (last[0].get("error") or "songs keep failing")[:160]}
                app.log(f"radio: 3 failed takes in a row on {s['name']}; pausing generation {delay}s")
                continue
            if last and last[0]["status"] == "done":
                backoff.pop(sid, None)
            ready = len(station_tracks(sid, ("approved",)))
            fly = inflight(sid)
            if ready + fly < state["target_ahead"] and fly < state["max_inflight"] + 1:
                with lock:
                    writing[sid] = writing.get(sid, 0) + 1
                threading.Thread(target=produce, args=(sid,), daemon=True).start()
        except Exception:
            app.log("radio spool error:", traceback.format_exc())


def produce(sid):
    try:
        s = stations.get(sid)
        if not s:
            return
        with app.lock:   # nothing ready and nothing being made for this station -> open with a short song
            busy = any(j.get("station") == sid and j.get("source") == "radio" and j["status"] in ("queued", "running")
                       for j in app.jobs.values())
        short = not busy and not station_tracks(sid, ("approved",)) and writing.get(sid, 0) <= 1
        spec = write_spec(s, short=short)
        if not state["on"] or state["station"] != sid:
            app.log("radio: discarding a spec written after the radio was switched off / retuned")
            return
        body = {"mode": "instrumental" if spec["instrumental"] else "song", "title": spec["title"], "author": "radio",
                "style": spec["style"], "lyrics": spec["lyrics"], "cot": "full", "max_seconds": 150 if short else 300}   # a safety cap only: songs should end on their own
        if not spec["instrumental"] and not spec["lyrics"].strip():
            body["mode"] = "instrumental"
        app.new_jobs(body, extra={"source": "radio", "station": sid, "priority": 1, "radio_status": "rendering",
                                  "spec_writer": spec["writer"]})
        with lock:
            s["stats"]["generated"] += 1; save_station(s)
    except Exception:
        app.log("radio produce error:", traceback.format_exc())
    finally:
        with lock:
            writing[sid] = max(0, writing.get(sid, 1) - 1)


def judge(job):
    """Called (in a thread) when a radio job finishes rendering: CLAP critic decides approve/reject."""
    sid = job.get("station"); s = stations.get(sid)
    if not s:
        return
    if job["status"] != "done":
        job["radio_status"] = "failed"; app.save_job(job)
        return
    job["radio_status"] = "judging"
    try:
        path = app.DATA / job["id"] / "song.mp3"
        if not path.is_file():
            path = app.DATA / job["id"] / "song" / "audio.flac"
        emb, y = audio_embedding(path)
        np.save(app.DATA / job["id"] / "clap.npy", emb)
        rms = float(np.sqrt(np.mean(y ** 2)))
        sane = (job.get("seconds") or 0) >= 30 and rms > 0.01
        if s["kind"] == "folder" and (sdir(sid) / "corpus.npy").is_file():
            ref = np.load(sdir(sid) / "corpus.npy")
        else:
            ref = text_embeddings([s["description"][:300] or s["name"]])
        base = float(np.sort(ref @ emb)[::-1][:5].mean())
        likes, dislikes, recent = [], [], []
        for f in s["feedback"]:
            p = app.DATA / f["track"] / "clap.npy"
            if p.is_file() and f["kind"] in ("keep", "dislike"):
                (likes if f["kind"] == "keep" else dislikes).append(np.load(p))
        with app.lock:
            aired = sorted((j for j in app.jobs.values() if j.get("station") == sid and j.get("source") == "radio" and
                            j["id"] != job["id"] and j.get("radio_status") in ("approved", "played")),
                           key=lambda j: -j.get("approved_at", 0))[:12]
        for j in aired:
            p = app.DATA / j["id"] / "clap.npy"
            if p.is_file():
                recent.append(np.load(p))
        like_b = 0.15 * (float(np.max(np.stack(likes) @ emb)) - 0.5) if likes else 0.0       # gentle nudge only
        dis_p = 0.5 * max(0.0, float(np.max(np.stack(dislikes) @ emb)) - 0.6) if dislikes else 0.0
        dup = float(np.max(np.stack(recent) @ emb)) if recent else 0.0
        score = base + like_b - dis_p - 1.0 * max(0.0, dup - 0.93)    # CLAP saturates within a genre: soft novelty penalty
        if dup > 0.975:
            sane = False                                                            # near-clone of something that aired
        hist = s["scores"][-20:]
        learned = s["kind"] == "folder"
        keep_frac, warmup = (0.7, 4) if learned else (0.85, 8)    # text stations only drop clear misses
        bar = float(np.quantile(hist, 1 - keep_frac)) if len(hist) >= warmup else -1.0
        verdict = "approved" if sane and score >= bar else "rejected"
        with lock:
            if job.get("spec_writer") != "fallback":
                s["scores"] = (s["scores"] + [score])[-50:]
            s["stats"]["approved" if verdict == "approved" else "rejected"] += 1
            save_station(s)
        job.update(radio_status=verdict, critic={"score": round(score, 4), "bar": round(bar, 4), "sane": sane, "dup": round(dup, 3)},
                   approved_at=time.time())
        app.save_job(job)
        app.log("radio critic", job["title"], verdict, round(score, 3), "bar", round(bar, 3))
    except Exception:
        app.log("radio critic error:", traceback.format_exc())
        job["radio_status"] = "approved"; job["approved_at"] = time.time(); app.save_job(job)


# ----------------------------------------------------------------------------- broadcast
# One shared live MP3 stream: a single playout thread picks the next approved track and every listener
# (browser deck, VLC, phone) hears the same audio. 192 kbit/s CBR at 48 kHz -> exactly 24000 bytes per second,
# so a byte offset in the stream is also a time offset; clients use that to know which song is audible.
BPS = 24000
FRAME = 576                         # 144 * 192000 / 48000: every frame is exactly this size (no padding)
BURST = 4 * BPS // FRAME * FRAME   # ~4 s, whole frames: new listeners get it at once so playback starts immediately
ENCODE = ["-vn", "-ac", "2", "-ar", "48000", "-c:a", "libmp3lame", "-b:a", "192k", "-id3v2_version", "0",
          "-write_xing", "0", "-map_metadata", "-1", "-f", "mp3", "pipe:1"]
bc = {"listeners": {}, "buf": bytearray(), "bytes": 0, "epoch": 0.0, "now": None, "aired": [],
      "skip": threading.Event(), "wake": threading.Event(), "silence": b"", "last": 0.0}
bc_lock = threading.Lock()


def _broadcast(chunk):
    """Pace the stream at real time (bytes sent == elapsed seconds * BPS), then fan the chunk out to listeners."""
    ahead = (bc["bytes"] + len(chunk)) / BPS - (time.time() - bc["epoch"])
    if ahead > 0:
        time.sleep(ahead)
    with bc_lock:
        if ahead < -1.0:                # we were idle (no listeners): restart the clock instead of bursting to catch up
            bc["epoch"] = time.time() - (bc["bytes"] + len(chunk)) / BPS
            bc["buf"].clear()
        bc["bytes"] += len(chunk)
        bc["last"] = time.time()
        bc["buf"] += chunk
        del bc["buf"][:-BURST]                  # BURST is a whole number of frames, so the burst starts on a frame
        for lid, li in list(bc["listeners"].items()):
            try:
                li["q"].put_nowait(chunk)
            except queue.Full:          # a listener that cannot keep up gets dropped (it reconnects)
                bc["listeners"].pop(lid, None)


def _air(entry):
    with bc_lock:
        entry = dict(entry or {"id": None}, start_pos=round(bc["bytes"] / BPS, 3))
        bc["now"] = entry if entry.get("id") else None
        bc["aired"] = (bc["aired"] + [entry])[-8:]


def _play_track(t):
    d = app.DATA / t["id"]
    src = d / "song.mp3"
    if not src.is_file():
        src = d / "song" / "audio.flac"
    proc = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(src), *ENCODE],
                            stdout=subprocess.PIPE, stdin=subprocess.DEVNULL)
    bc["skip"].clear()
    _air(t | {"duration": t.get("seconds")})
    app.log("radio on air:", t["title"], f"({len(bc['listeners'])} listening)")
    checked, start, pending = 0.0, bc["bytes"], b""
    try:
        while True:
            chunk = os.read(proc.stdout.fileno(), 8192)
            if not chunk:
                break
            pending += chunk                       # only send whole MP3 frames so a skip never leaves a torn frame
            n = len(pending) // FRAME * FRAME
            if not n:
                continue
            _broadcast(pending[:n]); pending = pending[n:]
            if bc["skip"].is_set() or not state["on"]:
                break
            if not bc["listeners"]:                # everyone left: stop airing to nobody
                if (bc["bytes"] - start) / BPS < 30:     # barely started -> put it back on the spool for later
                    with app.lock:
                        j = app.jobs.get(t["id"])
                        if j:
                            j.update(radio_status="approved", played_at=None); app.save_job(j)
                break
            if time.time() - checked > 1:          # retuned to another station: cut over once it has a song ready
                checked = time.time()
                if t.get("station") != state["station"] and station_tracks(state["station"]):
                    break
    finally:
        proc.kill(); proc.wait()


def _play_silence():
    """Dead air while the radio is off or the next song is still being made (keeps listener connections open)."""
    _air(None)
    while bc["listeners"]:
        for i in range(0, len(bc["silence"]), 8 * FRAME):
            _broadcast(bc["silence"][i:i + 8 * FRAME])
        if state["on"] and state["station"] and station_tracks(state["station"]):
            return


def playout_loop():
    try:
        bc["silence"] = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                                        "anullsrc=r=48000:cl=stereo", "-t", "1.152", *ENCODE], capture_output=True, timeout=30).stdout
    except Exception:
        app.log("radio: could not make the silence frames:", traceback.format_exc())
    while True:
        try:
            if not bc["listeners"]:                 # nobody listening: don't burn songs into the void
                if bc["now"] is not None:
                    _air(None)
                bc["wake"].wait(2); bc["wake"].clear()
                continue
            t = next_track(state["station"]) if state["on"] and state["station"] else None
            if t:
                _play_track(t)
            else:
                _play_silence()
        except Exception:
            app.log("radio playout error:", traceback.format_exc()); time.sleep(1)


def stream(h):
    """GET /radio/stream.mp3 - the live broadcast. ?l=<id> lets the deck match up what it hears with the playlist."""
    lid = re.sub(r"\W", "", (h.path.split("l=", 1)[1] if "l=" in h.path else "").split("&")[0])[:40] or uuid.uuid4().hex[:12]
    q = queue.Queue(maxsize=400)                    # ~2 minutes of slack before a stuck client is dropped
    with bc_lock:
        if time.time() - bc["last"] > 1.5:          # stream was idle: the buffer holds stale audio
            bc["buf"].clear()
        burst = bytes(bc["buf"])
        bc["listeners"][lid] = {"q": q, "p0": (bc["bytes"] - len(burst)) / BPS, "since": time.time(),
                                "agent": (h.headers.get("User-Agent") or "")[:80]}
    bc["wake"].set()
    try:
        h.send_response(200)
        for k, v in (("Content-Type", "audio/mpeg"), ("Cache-Control", "no-cache, no-store"), ("Connection", "close"),
                     ("Access-Control-Allow-Origin", "*"), ("icy-name", "b70 radio"), ("X-Content-Type-Options", "nosniff")):
            h.send_header(k, v)
        h.end_headers()
        if burst:
            h.wfile.write(burst)
        while lid in bc["listeners"] and bc["listeners"][lid]["q"] is q:
            try:
                chunk = q.get(timeout=5)
            except queue.Empty:
                continue
            h.wfile.write(chunk)
    except (BrokenPipeError, ConnectionResetError, OSError):
        pass
    finally:
        with bc_lock:
            if bc["listeners"].get(lid, {}).get("q") is q:
                bc["listeners"].pop(lid, None)
    h.close_connection = True


def skip(track_id, kind="skip"):
    """Skip (or NOPE) the song on air - for everyone, it's one broadcast."""
    if track_id:
        feedback(track_id, kind)
    now = bc["now"]
    if now and (not track_id or now["id"] == track_id):
        bc["skip"].set()
    return {"ok": True}


def stream_view(listener=None):
    with bc_lock:
        li = bc["listeners"].get(listener) if listener else None
        return {"pos": round(bc["bytes"] / BPS, 3), "listeners": len(bc["listeners"]), "aired": bc["aired"],
                "now": bc["now"], "p0": li["p0"] if li else None, "url": "/radio/stream.mp3"}


# ----------------------------------------------------------------------------- API

def track_view(j):
    return {"id": j["id"], "title": j["title"], "style": j["style"], "lyrics": j.get("lyrics", ""), "mode": j["mode"],
            "seconds": j.get("seconds"), "station": j.get("station"), "critic": j.get("critic"), "feedback": j.get("radio_feedback"),
            "played_at": j.get("played_at"), "audio": f"/files/{j['id']}/song.mp3"}


def api_state(listener=None):
    with lock:
        st = []
        for s in sorted(stations.values(), key=lambda s: s["created"]):
            st.append({k: s.get(k) for k in ("id", "name", "kind", "description", "vocals", "language", "stats", "analysis")} |
                      {"ready": len(station_tracks(s["id"])), "inflight": inflight(s["id"]),
                       "problem": (backoff[s["id"]]["reason"] + f" — retrying in {int(backoff[s['id']]['until'] - time.time())}s")
                                  if s["id"] in backoff and time.time() < backoff[s["id"]]["until"] else None,
                       "profile": {k: v for k, v in (s.get("profile") or {}).items() if k in ("tracks", "top", "bpm_median", "bpm_range", "keys", "vocal_share")} or None,
                       "files": len([p for p in (sdir(s["id"]) / "music").rglob("*") if p.suffix.lower() in AUDIO_EXT]) if s["kind"] == "folder" and (sdir(s["id"]) / "music").exists() else None})
    with app.lock:
        making = [{"title": j["title"], "status": j["status"], "stage": j.get("stage"), "progress": j.get("progress"), "station": j.get("station")}
                  for j in app.jobs.values() if j.get("source") == "radio" and j["status"] in ("queued", "running")]
        judging = sum(1 for j in app.jobs.values() if j.get("radio_status") == "judging")
    return {"on": state["on"], "station": state["station"], "target_ahead": state["target_ahead"], "stations": st,
            "making": making, "judging": judging, "gpu": app.state()["gpu"], "stream": stream_view(listener)}


def next_track(sid, mark=True):
    with app.lock:
        ready = sorted(station_tracks(sid, ("approved",)), key=lambda j: j.get("approved_at", 0))
        if not ready:
            return None
        j = ready[0]
        if mark:
            j.update(radio_status="played", played_at=time.time()); app.save_job(j)
    if mark and sid in stations:
        with lock:
            stations[sid]["stats"]["played"] += 1; save_station(stations[sid])
    return track_view(j)


def history(sid, limit=40):
    with app.lock:
        played = sorted((j for j in app.jobs.values() if j.get("source") == "radio" and j.get("station") == sid and j.get("played_at")),
                        key=lambda j: -j["played_at"])[:limit]
    return [track_view(j) for j in played]


def feedback(track_id, kind):
    if kind not in ("keep", "skip", "dislike", "clear"):
        raise app.BadRequest("feedback must be keep, skip, dislike or clear")
    with app.lock:
        j = app.jobs.get(track_id)
    if not j or j.get("source") != "radio":
        raise app.BadRequest("No such radio track")
    s = stations.get(j.get("station"))
    with lock:
        if s:
            s["feedback"] = [f for f in s["feedback"] if f["track"] != track_id]
            if kind != "clear":
                s["feedback"].append({"track": track_id, "kind": kind, "title": j["title"], "style": j["style"], "t": time.time()})
                s["feedback"] = s["feedback"][-200:]
                s["stats"][{"keep": "kept", "skip": "skipped", "dislike": "disliked"}[kind]] += 1
            save_station(s)
    j["radio_feedback"] = None if kind == "clear" else kind
    app.save_job(j)
    return {"ok": True}


def set_power(on, sid=None):
    with lock:
        if sid:
            if sid not in stations:
                raise app.BadRequest("No such station")
            state["station"] = sid
        if on and not state["station"]:
            raise app.BadRequest("Pick a station first")
        state["on"] = bool(on)
        save_state()
    if not on:
        with app.lock:                         # drop queued and in-progress radio work so the GPUs free up fast
            for j in app.jobs.values():
                if j.get("source") == "radio" and j["status"] == "queued":
                    j.update(status="cancelled", stage="", finished=time.time(), radio_status="cancelled"); app.save_job(j)
                elif j.get("source") == "radio" and j["status"] == "running":
                    j["_cancel"] = True
            studio_busy = any(j["status"] in ("queued", "running") and j.get("source") != "radio" for j in app.jobs.values())
            if not studio_busy:
                app.gpu["release_on_idle"] = True   # radio off -> give the GPUs back once running takes finish
    else:
        app.gpu["release_on_idle"] = False
    return {"ok": True, "on": state["on"], "station": state["station"]}


def upload(sid, name, data):
    s = stations.get(sid)
    if not s or s["kind"] != "folder":
        raise app.BadRequest("Uploads go to a folder station")
    name = re.sub(r"[^\w .()\[\]-]+", "_", Path(name).name).strip()[:120]
    if Path(name).suffix.lower() not in AUDIO_EXT:
        raise app.BadRequest("Unsupported file type")
    d = sdir(sid) / "music"; d.mkdir(parents=True, exist_ok=True)
    stem, ext, n = Path(name).stem, Path(name).suffix, 2
    while (d / name).exists():
        name = f"{stem} ({n}){ext}"; n += 1
    (d / name).write_bytes(data)
    return {"ok": True, "file": name}


def start_analysis(sid):
    s = stations.get(sid)
    if not s or s["kind"] != "folder":
        raise app.BadRequest("Only folder stations are analyzed")
    if s["analysis"].get("status") == "running":
        return {"ok": True}
    threading.Thread(target=analyze_station, args=(sid,), daemon=True).start()
    return {"ok": True}


def handle(h, method, path):
    """Route /api/radio/* requests. Returns True if handled."""
    q = {}
    if "?" in h.path:
        q = dict(x.split("=", 1) for x in h.path.split("?", 1)[1].split("&") if "=" in x)
    try:
        if method == "GET" and path == "/api/radio/state":
            return h.send(200, api_state(q.get("listener"))) or True
        if method == "POST" and path == "/api/radio/skip":
            b = h.body(); return h.send(200, skip(b.get("track"), "dislike" if b.get("kind") == "dislike" else "skip")) or True
        if method == "GET" and path == "/api/radio/next":
            t = next_track(q.get("station") or state["station"], mark=q.get("peek") != "1")
            return h.send(200, {"track": t}) or True
        m = re.fullmatch(r"/api/radio/stations/([\w]+)/history", path)
        if method == "GET" and m:
            return h.send(200, {"tracks": history(m.group(1))}) or True
        if method == "POST" and path == "/api/radio/stations":
            return h.send(200, {"station": create_station(h.body())}) or True
        m = re.fullmatch(r"/api/radio/stations/([\w]+)", path)
        if method == "DELETE" and m:
            delete_station(m.group(1)); return h.send(200, {"ok": True}) or True
        if method == "POST" and path == "/api/radio/power":
            b = h.body(); return h.send(200, set_power(b.get("on"), b.get("station"))) or True
        if method == "POST" and path == "/api/radio/feedback":
            b = h.body(); return h.send(200, feedback(b.get("track", ""), b.get("kind", ""))) or True
        m = re.fullmatch(r"/api/radio/stations/([\w]+)/analyze", path)
        if method == "POST" and m:
            return h.send(200, start_analysis(m.group(1))) or True
        m = re.fullmatch(r"/api/radio/stations/([\w]+)/files/(.+)", path)
        if method == "PUT" and m:
            n = int(h.headers.get("Content-Length") or 0)
            if n > 200 * 2**20:
                raise app.BadRequest("File too large (200 MB max)")
            from urllib.parse import unquote
            return h.send(200, upload(m.group(1), unquote(m.group(2)), h.rfile.read(n))) or True
    except app.BadRequest as e:
        return h.send(400, {"error": str(e)}) or True
    except json.JSONDecodeError:
        return h.send(400, {"error": "Invalid JSON"}) or True
    return False
