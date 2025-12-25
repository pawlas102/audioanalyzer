import librosa
import librosa.display
import matplotlib.pyplot as plt
import librosa
audio, sr = librosa.load("440hz.mp3", sr=44100, mono=True)

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

for i in range(20):
    print(f"Frame {i}: F0 = {f0[i]}, Note = {note_names[i]}, Voiced = {voiced_flag[i]}, Prob = {voiced_probs[i]}")

