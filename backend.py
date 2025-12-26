from flask import Flask, jsonify
import librosa

app = Flask(__name__)


def analyze_audio(file_path):
    audio, sr = librosa.load(file_path, sr=44100, mono=True)

    frame_length = 2048
    hop_length = 256
    fmin = librosa.note_to_hz("E2")
    fmax = librosa.note_to_hz("E6")

    result = librosa.pyin(
        audio,
        fmin=fmin,
        fmax=fmax,
        frame_length=frame_length,
        hop_length=hop_length
    )

    f0 = result[0]
    voiced_flag = result[1]
    voiced_probs = result[2]

    note_names = []
    for hz in f0:
        if hz is None:
            note_names.append(None)
        else:
            midi = librosa.hz_to_midi(hz)
            note = librosa.midi_to_note(midi)
            note_names.append(note)


    frames = []
    for i in range(len(f0)):
        frames.append({
            "frame": i,
            "f0": float(f0[i]) if f0[i] is not None else None,
            "note": note_names[i],
            "voiced": bool(voiced_flag[i]) if voiced_flag[i] is not None else None,
            "prob": float(voiced_probs[i]) if voiced_probs[i] is not None else None
        })

    return frames


@app.route("/analyze")
def analyze():
    frames = analyze_audio("440hz.mp3")
    return jsonify({"frames": frames})


if __name__ == "__main__":
    app.run()
