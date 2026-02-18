#!/usr/bin/env python3
"""Transcription audio via faster-whisper.

Deux modes disponibles :
  --mode=wyoming  (défaut) Protocole Wyoming TCP sur le port 10300
  --mode=api      API REST OpenAI-compatible (nécessite le container
                  fedirz/faster-whisper-server sur le port 8080)

Exemples :
  ./audio2texte.py                          # Wyoming, test.wav
  ./audio2texte.py mon_fichier.wav          # Wyoming, fichier spécifié
  ./audio2texte.py --mode=api recording.wav # API HTTP
  ./audio2texte.py --mode=api --ip=10.1.5.57 --port=8080 recording.wav

Conversion préalable si nécessaire :
  ffmpeg -i input.flac -ar 16000 -ac 1 -sample_fmt s16 output.wav
"""

import argparse
import asyncio
import io
import subprocess
import sys
import time
import wave
from pathlib import Path

# ---------------------------------------------------------------------------
# Paramètres audio attendus par faster-whisper
# ---------------------------------------------------------------------------
RATE = 16000
WIDTH = 2       # 16 bits
CHANNELS = 1
# Envoyer 4 s par chunk (moins de round-trips réseau que 1 s)
CHUNK_SECONDS = 4
CHUNK_SIZE = RATE * WIDTH * CHANNELS * CHUNK_SECONDS


# ===================================================================
#  Chargement audio
# ===================================================================

def load_audio_as_pcm(audio_file: str) -> bytes:
    """Retourne du PCM 16-bit mono 16 kHz. Accepte WAV, FLAC, OGG, MP3…"""
    path = Path(audio_file)
    if path.suffix.lower() == ".wav":
        try:
            return _read_wav(audio_file)
        except Exception:
            pass
    return _convert_with_ffmpeg(audio_file)


def _read_wav(audio_file: str) -> bytes:
    with wave.open(audio_file, "rb") as w:
        assert w.getsampwidth() == WIDTH
        assert w.getnchannels() == CHANNELS
        assert w.getframerate() == RATE
        pcm = w.readframes(w.getnframes())
        dur = w.getnframes() / w.getframerate()
        print(f"   WAV : {w.getframerate()} Hz, mono, {dur:.1f}s, {len(pcm)} octets")
        return pcm


def _convert_with_ffmpeg(audio_file: str) -> bytes:
    print("   Conversion via ffmpeg…")
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", audio_file,
         "-ar", str(RATE), "-ac", str(CHANNELS),
         "-sample_fmt", "s16", "-f", "wav", "pipe:1"],
        capture_output=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg erreur :\n{r.stderr.decode(errors='replace')}")
    with wave.open(io.BytesIO(r.stdout), "rb") as w:
        pcm = w.readframes(w.getnframes())
        dur = w.getnframes() / w.getframerate()
        print(f"   Converti : {w.getframerate()} Hz, mono, {dur:.1f}s, {len(pcm)} octets")
        return pcm


# ===================================================================
#  Mode Wyoming (TCP port 10300)
# ===================================================================

async def transcribe_wyoming(ip: str, port: int, audio_file: str) -> str | None:
    from wyoming.asr import Transcribe, Transcript
    from wyoming.audio import AudioChunk, AudioStart, AudioStop
    from wyoming.event import async_read_event, async_write_event

    print(f"📡 Wyoming → {ip}:{port}")
    reader, writer = await asyncio.open_connection(ip, port)

    try:
        pcm_data = load_audio_as_pcm(audio_file)

        # Transcribe
        await async_write_event(Transcribe(language="fr").event(), writer)
        # AudioStart
        await async_write_event(
            AudioStart(rate=RATE, width=WIDTH, channels=CHANNELS).event(), writer
        )
        # AudioChunks — gros morceaux pour réduire les round-trips
        offset, n = 0, 0
        while offset < len(pcm_data):
            chunk = pcm_data[offset:offset + CHUNK_SIZE]
            await async_write_event(
                AudioChunk(rate=RATE, width=WIDTH, channels=CHANNELS, audio=chunk).event(),
                writer,
            )
            n += 1
            offset += CHUNK_SIZE
        print(f"→ {n} chunk(s) envoyé(s)")
        # AudioStop
        await async_write_event(AudioStop().event(), writer)

        # Attente résultat
        print("⏳ Transcription…")
        t0 = time.perf_counter()
        while True:
            event = await asyncio.wait_for(async_read_event(reader), timeout=120)
            if event is None:
                print("❌ Connexion fermée")
                return None
            if Transcript.is_type(event.type):
                transcript = Transcript.from_event(event)
                dt = time.perf_counter() - t0
                print(f"✅ Inférence : {dt:.1f}s")
                return transcript.text

    except asyncio.TimeoutError:
        print("❌ Timeout (120s)")
        return None
    except Exception as e:
        print(f"❌ {type(e).__name__}: {e}")
        return None
    finally:
        writer.close()
        await writer.wait_closed()


# ===================================================================
#  Mode API HTTP (OpenAI-compatible, port 8080)
# ===================================================================


#TODO : Ne fonctionne pas car non compatible sur ARM du GB10


def transcribe_api(ip: str, port: int, audio_file: str) -> str | None:
    """Appel REST POST /v1/audio/transcriptions (OpenAI-compatible)."""
    import requests

    url = f"http://{ip}:{port}/v1/audio/transcriptions"
    print(f"📡 API HTTP → {url}")

    pcm_data = load_audio_as_pcm(audio_file)

    # Construire un WAV en mémoire à envoyer
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(WIDTH)
        w.setframerate(RATE)
        w.writeframes(pcm_data)
    buf.seek(0)

    print("⏳ Transcription…")
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            url,
            files={"file": ("audio.wav", buf, "audio/wav")},
            data={
                "model": "large-v3",
                "language": "fr",
                "response_format": "json",
            },
            timeout=120,
        )
        if resp.status_code != 200:
            print(f"❌ HTTP {resp.status_code}")
            print(f"   Réponse : {resp.text[:1000]}")
            return None
        dt = time.perf_counter() - t0
        print(f"✅ Inférence : {dt:.1f}s")
        result = resp.json()
        return result.get("text", "")
    except Exception as e:
        print(f"❌ {type(e).__name__}: {e}")
        return None


# ===================================================================
#  Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Transcription audio faster-whisper")
    parser.add_argument("audio", nargs="?", default="test.wav", help="Fichier audio")
    parser.add_argument("--mode", choices=["wyoming", "api"], default="wyoming",
                        help="wyoming (TCP 10300) ou api (HTTP 8080)")
    parser.add_argument("--ip", default="10.1.5.57", help="IP du serveur")
    parser.add_argument("--port", type=int, default=None,
                        help="Port (défaut: 10300 pour wyoming, 8080 pour api)")
    args = parser.parse_args()

    if args.port is None:
        args.port = 10300 if args.mode == "wyoming" else 8080

    print(f"📁 Fichier : {args.audio}")
    t_total = time.perf_counter()

    if args.mode == "wyoming":
        text = asyncio.run(transcribe_wyoming(args.ip, args.port, args.audio))
    else:
        text = transcribe_api(args.ip, args.port, args.audio)

    dt = time.perf_counter() - t_total
    if text:
        print(f"\n🎤 TRANSCRIPTION: {text}")
    print(f"\n⏱️  Temps total : {dt:.1f}s")


if __name__ == "__main__":
    main()
