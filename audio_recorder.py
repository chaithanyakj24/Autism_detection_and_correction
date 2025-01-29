import sounddevice as sd
from scipy.io.wavfile import write

def record_audio(file_name, duration=5, fs=44100):
    """Record audio and save to a WAV file."""
    print("Recording audio... Speak now!")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait for the recording to finish
    write(file_name, fs, audio_data)
    print(f"Audio saved to {file_name}")
