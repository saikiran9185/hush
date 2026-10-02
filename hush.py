"""Hush: split a recording into layers, one per sound (horns, traffic, wind, music…), on this Mac.

How it fits together:
  1. In Resolve you select a clip and run Workspace > Scripts > Hush - Extract Layers. That script
     drops a request file per clip into ~/Library/Application Support/Hush/inbox.
  2. launchd sees the inbox change and runs this file. It finds which sounds are in the clip
     (Apple's on-device sound classifier), pulls each one out with SAM Audio, and writes one WAV
     per layer to ~/Movies/Hush.
  3. Status files tell the waiting Resolve script to put each layer on its own track.

Without Resolve:
  python hush.py recording.m4a                      # writes the layers next to the recording
  python hush.py clip.mov --remove "car horn" --out clean.wav
"""

import argparse
import contextlib
import csv
import ctypes
import gc
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # install.sh downloads the models; after that, never touch the network
HERE = Path(__file__).resolve().parent
SUPPORT = Path.home() / "Library/Application Support/Hush"
INBOX, STATUS, MEDIA = SUPPORT / "inbox", SUPPORT / "status", Path.home() / "Movies/Hush"
ASK = HERE / ".build/Hush Ask.app/Contents/MacOS/ask"
SOUNDMAP = HERE / ".build/soundmap"
MODEL = "mlx-community/sam-audio-small-fp16"
SR = 48000
FFMPEG = next((p for p in ("/opt/homebrew/bin/ffmpeg", str(Path.home() / ".local/bin/ffmpeg"), "/usr/local/bin/ffmpeg")
               if os.path.exists(p)), "ffmpeg")

REQUEST = re.compile(r"^(\d+)-(\d+)of(\d+)_s([\d.]+)_e([\d.]+)\.csv$")  # <id>-<k>of<n>_s<start>_e<end>, source seconds
MESSAGE = re.compile(r"^(\d+)-msg_(\w+)\.edl$")
MESSAGES = {
    "noclip": "Select a clip in Resolve (or park the playhead on one), then run Hush again.",
    "retimed": "Hush can't split clips with speed changes. Reset the clip speed and try again.",
}

# Apple's 300 sound types, folded into layers people recognise: (layer, SAM Audio prompt, classes)
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


# --- audio -----------------------------------------------------------------------------------

def read_audio(path, start=0.0, end=None):
    """Decode [start, end) seconds of any audio/video file to stereo float32 at 48 kHz."""
    cmd = [FFMPEG, "-nostdin", "-v", "error", "-ss", f"{start:.6f}"] + (["-to", f"{end:.6f}"] if end else [])
    out = subprocess.run(cmd + ["-i", str(path), "-vn", "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, stdin=subprocess.DEVNULL)
    if out.returncode:
        raise RuntimeError(f"Couldn't read the audio: {out.stderr.decode(errors='ignore').strip()[-200:]}")
    audio = np.frombuffer(out.stdout, np.float32).reshape(-1, 2)
    if end is None:
        return audio.copy()
    want = round((end - start) * SR)  # decoders are off by a few ms; Resolve needs the exact length
    return np.pad(audio, ((0, max(0, want - len(audio))), (0, 0)))[:want]


def write_wav(path, audio):
    import soundfile as sf

    tmp = path.with_name(f".{path.name}.part")  # the Resolve script must never see a half-written file
    sf.write(tmp, np.clip(audio, -1, 1), SR, subtype="PCM_24", format="WAV")
    os.replace(tmp, path)


# --- what's in it: the sound map ------------------------------------------------------------------

def sound_map(audio):
    """Which sounds are where. Returns {"hop", "lanes": [{name, prompt, segments, curve, seconds}]}."""
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        write_wav(wav, audio)
        raw = json.loads(subprocess.run([str(SOUNDMAP), str(wav)], capture_output=True, check=True).stdout)
    hop, frames = raw["hop"], raw["frames"]
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
        segments = []
        for i in np.flatnonzero(on):  # each 1 s window that heard it, merged when they touch
            a, b = i * hop, i * hop + 1.0
            if segments and a - segments[-1][1] < 0.5:
                segments[-1][1] = b
            else:
                segments.append([a, b])
        lanes.append({"name": name, "prompt": PROMPT_OF.get(name, name.lower()), "segments": segments,
                      "curve": [round(float(v), 2) for v in curve], "seconds": float(on.sum() * hop)})
    lanes.sort(key=lambda l: -l["seconds"])
    return {"hop": hop, "lanes": lanes[:10]}


# --- pulling a sound out: SAM Audio --------------------------------------------------------------

def free_memory_percent():
    """macOS's own "memory free percentage" (what `memory_pressure` prints)."""
    value, size = ctypes.c_int(100), ctypes.c_size_t(4)
    ctypes.CDLL("libc.dylib").sysctlbyname(b"kern.memorystatus_level", ctypes.byref(value), ctypes.byref(size), None, 0)
    return value.value


def limit_memory():
    import mlx.core as mx
    import mlx_audio.sts.models.sam_audio.model as sam

    # SAM Audio wires up to the GPU's whole working set (5.7 GB on an 8 GB Mac). Wired memory can't be
    # paged out, and with Resolve open that froze the display. Keep it pageable and cap MLX at 40% of RAM.
    sam.wired_limit = lambda model: contextlib.nullcontext()
    mx.set_memory_limit(int(mx.device_info()["memory_size"] * 0.4))
    mx.set_cache_limit(256 << 20)


def encode_prompts(prompts):
    """Sound names -> SAM Audio text features. Done before the main model loads, so the ~900 MB
    T5 encoder and SAM Audio never share memory."""
    import mlx.core as mx
    from mlx_audio.sts.models.sam_audio.config import T5EncoderConfig
    from mlx_audio.sts.models.sam_audio.text_encoder import T5TextEncoder

    limit_memory()
    encoder, out = T5TextEncoder(T5EncoderConfig()), []
    for p in prompts:
        text, mask = encoder([p])
        mx.eval(text, mask)
        out.append((text, mask))
    del encoder
    gc.collect()
    mx.clear_cache()
    return out


def load_model():
    from mlx_audio.sts import SAMAudio

    limit_memory()
    return SAMAudio.from_pretrained(MODEL)


def separate(model, mono, text, chunk=2.0, overlap=0.5, on_chunk=None):
    """Split mono audio into (the encoded sound, everything else). 2 s chunks: memory grows with length."""
    import mlx.core as mx

    text, mask = text
    n, size, fade = len(mono), int(chunk * SR), int(overlap * SR)
    starts = range(0, max(n - fade, 1), size - fade)
    target, rest, weight = (np.zeros(n, np.float32) for _ in range(3))
    for i, a in enumerate(starts):
        while free_memory_percent() < 12:  # never push the Mac into heavy swapping
            gc.collect()
            mx.clear_cache()
            time.sleep(0.5)
        b = min(a + size, n)
        mx.random.seed(42 + i)
        result = model.separate(mx.array(mono[a:b])[None, None], [""], ode_opt={"method": "midpoint", "step_size": 1 / 16},
                                ode_decode_chunk_size=10, _text_features=text, _text_mask=mask)
        w = np.ones(b - a, np.float32)  # crossfade the overlaps
        if a > 0:
            w[:fade] = np.linspace(0, 1, fade)
        if b < n:
            w[-fade:] = np.linspace(1, 0, fade)
        for out, part in ((target, result.target), (rest, result.residual)):
            out[a:b] += np.array(part[0]).reshape(-1)[: b - a] * w
        weight[a:b] += w
        mx.clear_cache()
        if on_chunk:
            on_chunk(i + 1, len(starts))
    return target / np.maximum(weight, 1e-6), rest / np.maximum(weight, 1e-6)


def extract(model, audio, text, ranges, on_progress=None):
    """Pull the encoded sound out of stereo `audio`, only inside the time ranges (seconds).

    Returns (sound, rest), both full length; outside the ranges `rest` is untouched audio.
    """
    merged = []
    for a, b in sorted(ranges):  # pad a little, merge overlaps
        a, b = max(0.0, a - 0.3), min(len(audio) / SR, b + 0.3)
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    total, done = sum(b - a for a, b in merged) or 1, 0.0
    sound, rest = np.zeros_like(audio), audio.copy()
    for a, b in merged:
        i, j = int(a * SR), int(b * SR)
        report = (lambda k, n: on_progress((done + (b - a) * k / n) / total)) if on_progress else None
        target, residual = separate(model, audio[i:j].mean(axis=1), text, on_chunk=report)
        fade = np.ones(j - i, np.float32)  # 30 ms fades into the untouched audio around each range
        ramp = min(int(0.03 * SR), (j - i) // 2)
        fade[:ramp], fade[len(fade) - ramp:] = np.linspace(0, 1, ramp), np.linspace(1, 0, ramp)
        sound[i:j] = target[:, None] * fade[:, None]
        rest[i:j] = audio[i:j] * (1 - fade[:, None]) + residual[:, None] * fade[:, None]
        done += b - a
    return sound, rest


def split_into_layers(audio, max_layers=5):
    """Find the sounds in `audio` and pull each into its own layer. Yields (name, audio) as each is
    ready; the last one is whatever's left (the voice, if there is one)."""
    lanes = sound_map(audio)["lanes"]
    picks = [l for l in lanes if l["name"] != "Voice" and l["seconds"] >= 1.5][:max_layers]
    rest = audio
    if picks:
        texts = encode_prompts([l["prompt"] for l in picks])
        model = load_model()
        for lane, text in zip(picks, texts):
            sound, separated = extract(model, rest, text, lane["segments"])
            if np.sqrt((sound ** 2).mean()) < 0.001:  # about -60 dB: nothing really came out, no track for it
                continue
            rest = separated
            yield lane["name"], sound
    yield ("Voice & rest" if any(l["name"] == "Voice" for l in lanes) else "Everything else"), rest


# --- talking to the Resolve script -------------------------------------------------------------

def ask(*args):
    """Show the Hush panel (only used for messages now)."""
    subprocess.run([str(ASK), *args], capture_output=True)


def media_path(csv_file):
    """Resolve writes clip metadata as UTF-16 CSV; "Clip Directory" + "File Name" is the media file."""
    with open(csv_file, encoding="utf-16", newline="") as f:
        row = next(csv.DictReader(f))
    return os.path.join(row["Clip Directory"], row["File Name"])


def read_inbox(pending):
    """Collect request files into `pending` (id -> request); return the requests that are complete."""
    for f in sorted(INBOX.iterdir()):
        age = time.time() - f.stat().st_mtime
        if age > 60:  # the Resolve script gave up long ago
            f.unlink()
            continue
        if age < 0.2:  # let Resolve finish writing
            continue
        if m := REQUEST.match(f.name):
            req = pending.setdefault(m[1], {"clips": [], "messages": [], "seen": time.time()})
            req["total"] = int(m[3])
            req["clips"].append({"k": int(m[2]), "path": media_path(f), "start": float(m[4]), "end": float(m[5])})
        elif m := MESSAGE.match(f.name):
            pending.setdefault(m[1], {"clips": [], "messages": [], "seen": time.time()})["messages"].append(m[2])
        f.unlink()
    ready = [rid for rid, r in pending.items() if len(r["clips"]) == r.get("total", -1) or time.time() - r["seen"] > 2]
    return [(rid, pending.pop(rid)) for rid in ready]


def handle(rid, req):
    """One request from Resolve. The .ack file tells the script we're on it; removing it means we're finished."""
    ack = STATUS / f"{rid}.ack"
    ack.touch()
    try:
        if not req["clips"]:
            return ask("--message", "\n".join(MESSAGES.get(code, code) for code in req["messages"]))
        for c in sorted(req["clips"], key=lambda c: c["k"]):
            tag = f"{rid}-{c['k']}"
            try:
                audio = read_audio(c["path"], c["start"], c["end"])
                for j, (name, layer) in enumerate(split_into_layers(audio), 1):
                    write_wav(MEDIA / f"{tag}-{j}__{name}.wav", layer)
                    (STATUS / f"{tag}-{j}.layer").touch()  # the script puts it on a track named `name`
                (STATUS / f"{tag}.done").touch()
            except Exception as e:  # noqa: BLE001
                print(f"{tag}: {e}", file=sys.stderr)
                (STATUS / f"{tag}.failed").touch()
                text = "Not enough free memory. Close some apps and try again." if "memory" in str(e).lower() else str(e)
                ask("--message", f"Hush couldn't split {Path(c['path']).name}:\n{text}")
    finally:
        time.sleep(1)  # let the script pick up the last status file first
        ack.unlink(missing_ok=True)


def serve():
    """Run by launchd when the inbox changes: handle requests, then exit after 20 s of quiet."""
    os.nice(10)  # Resolve first
    for d in (INBOX, STATUS, MEDIA):
        d.mkdir(parents=True, exist_ok=True)
    for f in STATUS.iterdir():  # leftovers from a run that was killed
        if f.suffix == ".ack" or time.time() - f.stat().st_mtime > 86400:
            f.unlink()
    pending, quiet_since = {}, time.time()
    while time.time() - quiet_since < 20:
        ready = read_inbox(pending)
        for rid, req in ready:
            handle(rid, req)
        if ready or pending:
            quiet_since = time.time()
        time.sleep(0.3)


def main():
    if len(sys.argv) == 1:
        return serve()
    p = argparse.ArgumentParser(description="Split a recording into sound layers, or remove one sound.")
    p.add_argument("input")
    p.add_argument("--remove", help="remove just this sound, e.g. 'car horn' (needs --out)")
    p.add_argument("--out", help="output WAV for --remove")
    args = p.parse_args()
    audio = read_audio(args.input)
    if args.remove:
        text = encode_prompts([args.remove])[0]
        _, rest = extract(load_model(), audio, text, [[0, len(audio) / SR]])
        return write_wav(Path(args.out or "clean.wav").resolve(), rest)
    src = Path(args.input).resolve()
    for j, (name, layer) in enumerate(split_into_layers(audio), 1):
        out = src.with_name(f"{src.stem} - {j} {name}.wav")
        write_wav(out, layer)
        print(out)


if __name__ == "__main__":
    main()
