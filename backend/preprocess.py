'EOF'
import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel


def muunna(lahde: Path, kohde: Path):
    """Muuntaa äänen 16 kHz mono WAV -muotoon."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(lahde), "-ar", "16000", "-ac", "1", str(kohde)],
        check=True,
        capture_output=True,
    )


def lue_wav(polku: Path):
    """Lukee 16 kHz mono WAV -tiedoston lukutaulukoksi."""
    with wave.open(str(polku), "rb") as w:
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def main():
    if len(sys.argv) < 2:
        print("Käyttö: python preprocess.py aanitiedosto.mp3")
        sys.exit(1)

    lahde = Path(sys.argv[1])
    data = Path("data")
    data.mkdir(exist_ok=True)

    wav = data / (lahde.stem + ".wav")
    ulos = data / (lahde.stem + ".jsonl")

    print("Muunnetaan ääni...")
    muunna(lahde, wav)

    print("Litteroidaan...")
    malli = WhisperModel("small")
    segmentit, info = malli.transcribe(lue_wav(wav), language="fi")

    with open(ulos, "w", encoding="utf-8") as f:
        for s in segmentit:
            rivi = {
                "audio": str(wav),
                "start": round(s.start, 2),
                "end": round(s.end, 2),
                "text": s.text.strip(),
                "avg_logprob": round(s.avg_logprob, 3),
                "status": "pending",
            }
            f.write(json.dumps(rivi, ensure_ascii=False) + "\n")
            print(f"[{rivi['start']}-{rivi['end']}] {rivi['text']}")

    print(f"Valmis: {ulos}")


if __name__ == "__main__":
    main()
EOF
