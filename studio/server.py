"""Hush Studio: open any recording, see what's in it, and pull sounds out into layers you can edit.

  .venv/bin/python studio/server.py [file]      then open http://127.0.0.1:8740

How it works:
  - soundmap (Apple's on-device sound classifier) finds which sounds are where. Fast and light.
  - SAM Audio (hush.py) pulls a named sound out into its own layer. Slow, so only when you ask.
  - Plain DSP does the instant splits: lows/mids/highs, hits/sustained, loud/quiet, a dragged box.
Layers always add back up to the recording: whatever you pull out comes out of "Everything else".
"""

import json
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import hush  # noqa: E402

PORT = 8740
STUDIO = Path(__file__).resolve().parent
SOUNDMAP = STUDIO.parent / ".build/soundmap"
PROJECTS = Path.home() / "Movies/Hush/Studio"
SR = hush.SR

# Apple's 300 sound types, folded into layers people recognise. (name, SAM Audio prompt, classes)
INSTRUMENTS = ("plucked_string_instrument guitar electric_guitar bass_guitar acoustic_guitar steel_guitar_slide_guitar "
               "guitar_tapping guitar_strum banjo sitar mandolin zither ukulele keyboard_musical piano electric_piano organ "
               "electronic_organ hammond_organ synthesizer harpsichord percussion drum_kit drum snare_drum bass_drum timpani "
               "tabla cymbal hi_hat tambourine rattle_instrument gong mallet_percussion marimba_xylophone glockenspiel "
               "vibraphone steelpan orchestra brass_instrument french_horn trumpet trombone bowed_string_instrument "
               "violin_fiddle cello double_bass wind_instrument flute saxophone clarinet oboe bassoon harp harmonica "
               "accordion bagpipes didgeridoo shofar theremin singing_bowl disc_scratching")
GROUPS = [
    ("Voice", "a person speaking", "speech shout yell battle_cry children_shouting screaming whispering laughter "
     "baby_laughter giggling snicker belly_laugh chuckle_chortle crying_sobbing baby_crying sigh rapping humming"),
    ("Horns", "car horn honking", "car_horn air_horn train_horn foghorn reverse_beeps bicycle_bell"),
    ("Sirens", "siren", "emergency_vehicle police_siren ambulance_siren fire_engine_siren siren civil_defense_siren"),
    ("Traffic", "traffic noise", "traffic_noise car_passing_by truck bus motorcycle race_car vehicle_skidding "
     "power_windows engine engine_knocking engine_starting engine_idling engine_accelerating_revving"),
    ("Trains & planes", "train and aircraft noise", "rail_transport train train_whistle railroad_car "
     "train_wheels_squealing subway_metro aircraft helicopter airplane"),
    ("Crowd", "crowd chatter", "crowd chatter babble cheering applause booing clapping"),
    ("Music", "music", "music singing choir_singing yodeling whistling " + INSTRUMENTS),
    ("Wind", "wind noise", "wind wind_rustling_leaves wind_noise_microphone"),
    ("Water & rain", "water and rain", "thunderstorm thunder water rain raindrop stream_burbling waterfall ocean "
     "sea_waves gurgling boat_water_vehicle"),
    ("Birds", "birds chirping", "bird bird_vocalization bird_chirp_tweet bird_squawk pigeon_dove_coo crow_caw owl_hoot "
     "bird_flapping fowl chicken chicken_cluck rooster_crow turkey_gobble duck_quack goose_honk"),
    ("Animals", "animal sounds", "dog dog_bark dog_howl dog_bow_wow dog_growl dog_whimper cat cat_purr cat_meow "
     "horse_clip_clop horse_neigh cow_moo pig_oink sheep_bleat lion_roar frog frog_croak coyote_howl"),
    ("Insects", "insects buzzing", "insect cricket_chirp mosquito_buzz fly_buzz bee_buzz"),
    ("Fan & AC hum", "fan and air conditioner hum", "mechanical_fan air_conditioner hair_dryer vacuum_cleaner "
     "blender microwave_oven"),
    ("Footsteps", "footsteps", "person_running person_shuffling person_walking"),
    ("Knocks & bangs", "knocking and banging", "door door_slam knock tap thump_thud slap_smack hammer "
     "bowling_impact basketball_bounce glass_clink"),
    ("Phones & beeps", "phone ringing and beeps", "telephone telephone_bell_ringing ringtone alarm_clock beep "
     "smoke_detector door_bell"),
    ("Coughs & breath", "coughing and breathing", "breathing snoring gasp cough sneeze nose_blowing"),
    ("Typing & clicks", "typing and clicking", "typing typewriter typing_computer_keyboard click writing"),
    ("Tools", "power tool noise", "power_tool drill saw chainsaw lawn_mower hedge_trimmer sewing_machine"),
]
GROUP_OF = {c: g[0] for g in GROUPS for c in g[2].split()}
PROMPT_OF = {g[0]: g[1] for g in GROUPS}


# --- audio helpers -------------------------------------------------------------------------------

N_FFT, HOP = 2048, 512
WINDOW = np.hanning(N_FFT + 1)[:-1].astype(np.float32)


def stft(x):
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(x, (N_FFT // 2, N_FFT)), N_FFT)[::HOP]
    return np.fft.rfft(frames * WINDOW, axis=1).astype(np.complex64)


def istft(spec, length):
    frames = np.fft.irfft(spec, n=N_FFT, axis=1).astype(np.float32) * WINDOW
    out = np.zeros(len(frames) * HOP + N_FFT, np.float32)
    norm = np.zeros_like(out)
    for i, f in enumerate(frames):
        out[i * HOP:i * HOP + N_FFT] += f
        norm[i * HOP:i * HOP + N_FFT] += WINDOW ** 2
    return (out / np.maximum(norm, 1e-6))[N_FFT // 2:N_FFT // 2 + length]


def apply_masks(audio, masks):
    """Split stereo audio with STFT masks that add up to 1, so the parts add back up exactly."""
    parts = [np.zeros_like(audio) for _ in masks]
    for ch in range(audio.shape[1]):
        spec = stft(audio[:, ch])
        for part, mask in zip(parts, masks):
            part[:, ch] = istft(spec * mask, len(audio))
    return parts


def median_filter(a, k, axis):
    """Median over k neighbours along one axis, in blocks so it stays light on memory."""
    a = np.moveaxis(a, axis, -1)
    out = np.empty_like(a)
    padded = np.pad(a, [(0, 0)] * (a.ndim - 1) + [(k // 2, k // 2)], mode="edge")
    for i in range(0, len(a), 256):
        out[i:i + 256] = np.median(np.lib.stride_tricks.sliding_window_view(padded[i:i + 256], k, axis=-1), axis=-1)
    return np.moveaxis(out, -1, axis)


def split_masks(audio, kind, box=None):
    """Masks for the instant DSP splits. Returns [(name, mask)]; the masks add up to 1."""
    freqs = np.fft.rfftfreq(N_FFT, 1 / SR)[None, :]
    mag = np.abs(stft(audio.mean(axis=1)))
    if kind == "bands":
        low = 1 / (1 + (freqs / 250) ** 4)
        high = 1 - 1 / (1 + (freqs / 4000) ** 4)
        return [("Lows", low), ("Mids", np.clip(1 - low - high, 0, 1)), ("Highs", high)]
    if kind == "hits":  # harmonic/percussive split: sustained sounds are smooth in time, hits are smooth in pitch
        sustained = median_filter(mag, 17, axis=0) ** 2
        hits = median_filter(mag, 17, axis=1) ** 2
        share = hits / (hits + sustained + 1e-9)
        return [("Hits", share), ("Sustained", 1 - share)]
    if kind == "loud":  # frames well above the recording's typical level
        level = 10 * np.log10((mag ** 2).sum(axis=1) + 1e-9)
        loud = (level > np.median(level) + 8).astype(np.float32)
        loud = np.convolve(loud, np.ones(5) / 5, mode="same")[:, None] * np.ones_like(freqs)
        return [("Loud moments", loud), ("Quiet bed", 1 - loud)]
    if kind == "box":  # a time x frequency rectangle with soft edges
        t = np.arange(len(mag))[:, None] * HOP / SR
        soft = lambda v, a, b, w: np.clip(np.minimum(v - a, b - v) / w + 0.5, 0, 1)  # noqa: E731
        inside = soft(t, box["t0"], box["t1"], 0.03) * soft(np.log2(freqs + 1), np.log2(box["f0"] + 1),
                                                            np.log2(box["f1"] + 1), 0.15)
        return [("Selection", inside), (None, 1 - inside)]
    raise ValueError(kind)


def spectrogram_image(audio, width=1600, height=220):
    """Log-frequency spectrogram as uint8 rows (high pitch at the top), for the canvas."""
    mag = np.abs(stft(audio.mean(axis=1)))
    freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
    rows = np.geomspace(40, 18000, height)
    idx = np.clip(np.searchsorted(freqs, rows), 1, len(freqs) - 1)
    img = mag[:, idx]
    cols = min(width, len(img))
    edges = np.linspace(0, len(img), cols + 1).astype(int)
    img = np.stack([img[a:max(b, a + 1)].max(axis=0) for a, b in zip(edges[:-1], edges[1:])], axis=1)
    db = 20 * np.log10(img + 1e-6)
    db = np.clip((db - (db.max() - 80)) / 80, 0, 1)
    return (db[::-1] * 255).astype(np.uint8), cols, height


def sound_map(wav):
    """Lanes of detected sounds: per lane a confidence curve (one value per 0.5 s) and merged segments."""
    raw = json.loads(subprocess.run([str(SOUNDMAP), str(wav)], capture_output=True, check=True).stdout)
    frames = raw["frames"]
    curves = {}
    for i, (_, scores) in enumerate(frames):
        for cls, conf in scores.items():
            if cls == "silence":
                continue
            name = GROUP_OF.get(cls) or ("Knocks & bangs" if cls.startswith(("playing_", "rope_")) else
                                         cls.replace("_", " ").capitalize())
            curve = curves.setdefault(name, np.zeros(len(frames)))
            curve[i] = max(curve[i], conf)
    lanes = []
    for name, curve in curves.items():
        on = curve > 0.3
        if on.sum() < 2 or curve.max() < 0.5:
            continue
        segments, start = [], None
        for i, v in enumerate(list(on) + [False]):
            if v and start is None:
                start = i
            if not v and start is not None:
                a, b = start * raw["hop"], (i - 1) * raw["hop"] + 1.0
                if segments and a - segments[-1][1] < 0.5:
                    segments[-1][1] = b
                else:
                    segments.append([a, b])
                start = None
        lanes.append({"name": name, "prompt": PROMPT_OF.get(name, name.lower()), "segments": segments,
                      "curve": [round(float(v), 2) for v in curve], "seconds": float(on.sum() * raw["hop"])})
    lanes.sort(key=lambda l: -l["seconds"])
    return {"hop": raw["hop"], "lanes": lanes[:10]}


# --- the open recording ----------------------------------------------------------------------------

class Project:
    def __init__(self, source):
        source = Path(source)
        self.name = source.name
        self.dir = PROJECTS / f"{source.stem}-{time.strftime('%Y%m%d-%H%M%S')}"
        (self.dir / "layers").mkdir(parents=True, exist_ok=True)
        probe = subprocess.run([hush.FFMPEG, "-nostdin", "-i", str(source)], capture_output=True, text=True).stderr
        dur = next((l.split("Duration:")[1].split(",")[0].strip() for l in probe.splitlines() if "Duration:" in l), None)
        if not dur or dur == "N/A":
            raise ValueError("That file has no readable audio.")
        h, m, s = dur.split(":")
        audio = hush.read_audio(str(source), 0, int(h) * 3600 + int(m) * 60 + float(s))
        self.duration = len(audio) / SR
        hush.write_wav(self.dir / "original.wav", audio)
        self.map = sound_map(self.dir / "original.wav")
        self.spectrogram = spectrogram_image(audio)
        self.layers = []  # dicts: id, name, kind, version; audio kept in self.audio
        self.audio = {}
        self.next_id = 0
        self.add_layer("Everything else", audio, kind="rest")

    def add_layer(self, name, audio, kind="layer", after=None):
        lid = f"L{self.next_id}"
        self.next_id += 1
        layer = {"id": lid, "name": name, "kind": kind, "version": 0}
        pos = len(self.layers) if after is None else next(i for i, l in enumerate(self.layers) if l["id"] == after)
        self.layers.insert(pos, layer)
        self.set_audio(lid, audio)
        return layer

    def set_audio(self, lid, audio):
        self.audio[lid] = audio.astype(np.float32)
        layer = self.layer(lid)
        layer["version"] += 1
        hush.write_wav(self.dir / "layers" / f"{lid}.wav", self.audio[lid])

    def layer(self, lid):
        return next(l for l in self.layers if l["id"] == lid)

    def remove_layer(self, lid):
        """Put a layer's sound back into "Everything else"."""
        rest = self.layers[-1]["id"] if self.layers[-1]["kind"] == "rest" else None
        if lid == rest:
            return
        self.set_audio(rest, self.audio[rest] + self.audio.pop(lid))
        self.layers = [l for l in self.layers if l["id"] != lid]

    def info(self):
        return {"name": self.name, "duration": self.duration, "folder": str(self.dir), "map": self.map,
                "layers": self.layers, "spectrogram": {"width": self.spectrogram[1], "height": self.spectrogram[2]}}


project = None
lock = threading.Lock()
job = {"busy": False, "progress": 0, "message": "", "error": None}
model = {"sam": None, "used": 0.0}


def extract_ai(source, name, prompt, ranges):
    """Pull `prompt` out of layer `source` within the time ranges, into a new layer, with SAM Audio."""
    try:
        job.update(busy=True, progress=0, error=None, message=f"Getting ready to extract {name}…")
        text = hush.encode_prompt(prompt)
        if model["sam"] is None:
            job["message"] = "Loading SAM Audio…"
            model["sam"] = hush.load_model()
        audio = project.audio[source]
        sound, rest = np.zeros_like(audio), audio.copy()
        merged = []
        for a, b in sorted(ranges):  # pad a little and merge overlaps
            a, b = max(0.0, a - 0.3), min(project.duration, b + 0.3)
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        total = sum(b - a for a, b in merged) or 1
        done = 0.0
        for a, b in merged:
            i, j = int(a * SR), int(b * SR)
            span = b - a
            report = lambda k, n: job.update(progress=(done + span * k / n) / total,  # noqa: E731
                                             message=f"Extracting {name}… {(done + span * k / n):.0f} of {total:.0f} s")
            target, residual = hush.separate(model["sam"], audio[i:j].mean(axis=1), text, on_chunk=report)
            fade = np.ones(j - i, np.float32)  # 30 ms fades into the untouched audio around each range
            ramp = min(int(0.03 * SR), (j - i) // 2)
            fade[:ramp], fade[len(fade) - ramp:] = np.linspace(0, 1, ramp), np.linspace(1, 0, ramp)
            sound[i:j] = target[:, None] * fade[:, None]
            rest[i:j] = audio[i:j] * (1 - fade[:, None]) + residual[:, None] * fade[:, None]
            done += span
        with lock:
            project.add_layer(name, sound, kind="ai", after=source)
            project.set_audio(source, rest)
        job.update(busy=False, progress=1, message=f"{name} extracted")
    except Exception as e:  # noqa: BLE001
        job.update(busy=False, error=str(e), message="")
    finally:
        model["used"] = time.time()


def unload_idle_model():
    """Give SAM Audio's ~2.5 GB back to the Mac after 3 idle minutes."""
    while True:
        time.sleep(30)
        if model["sam"] is not None and not job["busy"] and time.time() - model["used"] > 180:
            import gc

            import mlx.core as mx

            model["sam"] = None
            gc.collect()
            mx.clear_cache()


# --- web server ----------------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, body, kind="application/json", status=200, headers=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            return self.send((STUDIO / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/api/project":
            return self.send(project.info() if project else {})
        if path == "/api/job":
            return self.send(job)
        if path == "/api/spectrogram" and project:
            img, w, h = project.spectrogram
            return self.send(img.tobytes(), "application/octet-stream", headers={"X-Width": w, "X-Height": h})
        if path.startswith("/audio/") and project:
            f = project.dir / "layers" / Path(path).name
            if f.exists():
                return self.send(f.read_bytes(), "audio/wav")
        self.send({"error": "not found"}, status=404)

    def do_POST(self):
        global project
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        path = self.path.split("?")[0]
        try:
            if path == "/api/upload":  # body = the file itself
                name = Path(self.path.split("name=", 1)[1]).name if "name=" in self.path else "recording"
                from urllib.parse import unquote

                tmp = PROJECTS / ".incoming" / unquote(name)
                tmp.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_bytes(body)
                project = Project(tmp)
                shutil.copy(tmp, project.dir / tmp.name)
                tmp.unlink()
                return self.send(project.info())
            args = json.loads(body or b"{}")
            if path == "/api/open":
                project = Project(args["path"])
                return self.send(project.info())
            if job["busy"]:
                return self.send({"error": "Still working on the last extract."}, status=409)
            if path == "/api/extract":
                threading.Thread(target=extract_ai, daemon=True,
                                 args=(args["source"], args["name"], args["prompt"], args["ranges"])).start()
                return self.send({"started": True})
            with lock:
                if path == "/api/split":
                    audio = project.audio[args["source"]]
                    masks = split_masks(audio, args["kind"], args.get("box"))
                    parts = apply_masks(audio, [m for _, m in masks])
                    named = [(n, p) for (n, _), p in zip(masks, parts)]
                    keep = next((p for n, p in named if n is None), None)  # a box leaves the rest in place
                    source = project.layer(args["source"])
                    for n, p in named:
                        if n:
                            label = n if source["kind"] == "rest" or keep is not None else f"{source['name']} · {n}"
                            project.add_layer(label, p, kind="dsp", after=args["source"])
                    if keep is not None:
                        project.set_audio(args["source"], keep)
                    elif project.layer(args["source"])["kind"] == "rest":
                        project.set_audio(args["source"], np.zeros_like(audio))
                    else:
                        project.layers = [l for l in project.layers if l["id"] != args["source"]]
                        project.audio.pop(args["source"])
                elif path == "/api/remove":
                    project.remove_layer(args["id"])
                elif path == "/api/rename":
                    project.layer(args["id"])["name"] = args["name"]
                elif path == "/api/export":
                    out = project.dir / "export"
                    out.mkdir(exist_ok=True)
                    gains = args.get("gains", {})
                    mix = np.zeros_like(next(iter(project.audio.values())))
                    for old in out.glob("*.wav"):
                        old.unlink()
                    for i, l in enumerate(project.layers, 1):
                        safe = "".join(c if c.isalnum() or c in " -_&·" else "-" for c in l["name"]).strip()
                        hush.write_wav(out / f"{i:02d} {safe}.wav", project.audio[l["id"]])
                        mix += project.audio[l["id"]] * gains.get(l["id"], 1.0)
                    hush.write_wav(out / "00 Mix.wav", mix)
                    subprocess.run(["open", str(out)])
                    return self.send({"folder": str(out)})
                else:
                    return self.send({"error": "unknown"}, status=404)
            return self.send(project.info())
        except Exception as e:  # noqa: BLE001
            return self.send({"error": str(e)}, status=500)


def main():
    global project
    PROJECTS.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 1:
        project = Project(sys.argv[1])
    threading.Thread(target=unload_idle_model, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Hush Studio on http://127.0.0.1:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
