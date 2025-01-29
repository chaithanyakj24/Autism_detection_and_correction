import os
import librosa
import numpy as np
import speech_recognition as sr
from syllapy import count
import matplotlib.pyplot as plt
import io
import base64
import librosa.display

class SpeechAnalyzer:
    def __init__(self):
        self.thresholds = {
            "pitch": (80, 350),
            "pauses_duration": 1.5,
            "speech_rate": (1.5, 5.5),
            "tone_spectral_centroid": (800, 3000)
        }

    def analyze_audio(self, audio_path):
        try:
            y, sr = librosa.load(audio_path, sr=16000)

            # Calculate pitch
            pitches, voiced_flags, _ = librosa.pyin(y, fmin=50, fmax=350, sr=sr)
            avg_pitch = np.nanmean(pitches[voiced_flags]) if np.any(voiced_flags) else 0

            # Calculate pauses
            rms_energy = librosa.feature.rms(y=y)[0]
            silence_threshold = 0.02
            silence_frames = rms_energy < silence_threshold
            pause_duration = np.sum(silence_frames) * (1 / sr)

            # Calculate speech rate
            duration = librosa.get_duration(y=y, sr=sr)
            transcription = self.transcribe_audio(audio_path)
            num_syllables = sum(count(word) for word in transcription.split())
            speech_rate = num_syllables / duration if duration > 0 else 0

            # Calculate tone (spectral centroid)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            avg_centroid = np.mean(spectral_centroid)

            return {
                "pitch": avg_pitch,
                "pauses_duration": pause_duration,
                "speech_rate": speech_rate,
                "tone_spectral_centroid": avg_centroid,
                "raw_data": {
                    "y": y,
                    "sr": sr,
                    "pitches": pitches,
                    "rms_energy": rms_energy,
                    "spectral_centroid": spectral_centroid
                }
            }

        except Exception as e:
            raise RuntimeError(f"Error analyzing audio: {e}")

    def transcribe_audio(self, audio_path):
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio)
        except (sr.UnknownValueError, sr.RequestError):
            return ""

    def classify_audio(self, features):
        """Classify audio as Dysarthria or Non-Dysarthria."""
        results = {
            "pitch": self.thresholds["pitch"][0] <= features["pitch"] <= self.thresholds["pitch"][1],
            "pauses_duration": features["pauses_duration"] <= self.thresholds["pauses_duration"],
            "speech_rate": self.thresholds["speech_rate"][0] <= features["speech_rate"] <= self.thresholds["speech_rate"][1],
            "tone_spectral_centroid": self.thresholds["tone_spectral_centroid"][0] <= features["tone_spectral_centroid"] <= self.thresholds["tone_spectral_centroid"][1],
        }
        classification = "Non-Dysarthria" if all(results.values()) else "Dysarthria"
        return classification, results

    def plot_to_base64(self, plt_figure):
        buf = io.BytesIO()
        plt_figure.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(plt_figure)
        return img_str

    def generate_single_audio_analysis(self, audio_path):
 
        features = self.analyze_audio(audio_path)

    # Ensure the temp directory exists
        output_dir = "temp_single_analysis"
        os.makedirs(output_dir, exist_ok=True)

        plots = {}

    # Waveform Plot
        waveform_path = os.path.join(output_dir, "waveform.png")
        fig, ax = plt.subplots(figsize=(12, 6))
        librosa.display.waveshow(features['raw_data']['y'], sr=features['raw_data']['sr'], ax=ax)
        ax.set_title("Waveform")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        plt.savefig(waveform_path)
        plt.close(fig)
        plots['waveform'] = waveform_path

    # Pitch Contour Plot
        pitch_path = os.path.join(output_dir, "pitch_contour.png")
        pitches = features['raw_data']['pitches']
        times = librosa.times_like(pitches)
        fig, ax = plt.subplots(figsize=(12, 6))
        pitches = pitches[:len(times)]  # Ensure alignment with times
        ax.plot(times, pitches)
        ax.set_title("Pitch Contour")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        plt.savefig(pitch_path)
        plt.close(fig)
        plots['pitch'] = pitch_path

    # Spectrogram Plot
        spectrogram_path = os.path.join(output_dir, "spectrogram.png")
        fig, ax = plt.subplots(figsize=(12, 6))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(features['raw_data']['y'])), ref=np.max)
        librosa.display.specshow(D, y_axis='log', x_axis='time', ax=ax)
        ax.set_title("Spectrogram")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        plt.colorbar(format='%+2.0f dB', ax=ax)
        plt.savefig(spectrogram_path)
        plt.close(fig)
        plots['spectrogram'] = spectrogram_path

    # Energy (RMS) Plot
        energy_path = os.path.join(output_dir, "energy.png")
        rms_energy = features['raw_data']['rms_energy']
        times = librosa.times_like(rms_energy)
        fig, ax = plt.subplots(figsize=(12, 6))
        rms_energy = rms_energy[:len(times)]  # Ensure alignment with times
        ax.plot(times, rms_energy)
        ax.set_title("Energy (RMS)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Energy")
        plt.savefig(energy_path)
        plt.close(fig)
        plots['energy'] = energy_path

    # Summary of Features
        summary_path = os.path.join(output_dir, "summary.txt")
        with open(summary_path, "w") as f:
            f.write("Audio Analysis Summary\n")
            f.write("----------------------\n")
            f.write(f"Pitch (Avg): {features['pitch']:.2f} Hz\n")
            f.write(f"Pauses Duration: {features['pauses_duration']:.2f} s\n")
            f.write(f"Speech Rate: {features['speech_rate']:.2f} syllables/s\n")
            f.write(f"Tone (Spectral Centroid Avg): {features['tone_spectral_centroid']:.2f} Hz\n")
        plots['summary'] = summary_path

        return plots


    def generate_comparison_plots(self, doctor_audio, patient_audio):
        doctor_features = self.analyze_audio(doctor_audio)
        patient_features = self.analyze_audio(patient_audio)
        
        # Ensure the temp directory exists
        output_dir = "temp_plots"
        os.makedirs(output_dir, exist_ok=True)

        plots = {}

        # Waveform Comparison
        waveform_path = os.path.join(output_dir, "waveform_comparison.png")
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        librosa.display.waveshow(doctor_features['raw_data']['y'], sr=doctor_features['raw_data']['sr'], ax=axes[0])
        axes[0].set_title("Doctor's Waveform")
        librosa.display.waveshow(patient_features['raw_data']['y'], sr=patient_features['raw_data']['sr'], ax=axes[1])
        axes[1].set_title("Patient's Waveform")
        plt.savefig(waveform_path)
        plt.close(fig)
        plots['waveform'] = waveform_path

        # Pitch Contour
        pitch_path = os.path.join(output_dir, "pitch_comparison.png")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        doctor_pitches = doctor_features['raw_data']['pitches']
        patient_pitches = patient_features['raw_data']['pitches']
        doctor_times = librosa.times_like(doctor_pitches)
        patient_times = librosa.times_like(patient_pitches)
        doctor_pitches = doctor_pitches[:len(doctor_times)]
        patient_pitches = patient_pitches[:len(patient_times)]
        ax1.plot(doctor_times, doctor_pitches)
        ax1.set_title("Doctor's Pitch Contour")
        ax2.plot(patient_times, patient_pitches)
        ax2.set_title("Patient's Pitch Contour")
        plt.savefig(pitch_path)
        plt.close(fig)
        plots['pitch'] = pitch_path

        # Spectrogram Comparison
        spectrogram_path = os.path.join(output_dir, "spectrogram_comparison.png")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        D_doctor = librosa.amplitude_to_db(np.abs(librosa.stft(doctor_features['raw_data']['y'])), ref=np.max)
        D_patient = librosa.amplitude_to_db(np.abs(librosa.stft(patient_features['raw_data']['y'])), ref=np.max)
        librosa.display.specshow(D_doctor, y_axis='log', x_axis='time', ax=ax1)
        ax1.set_title("Doctor's Spectrogram")
        librosa.display.specshow(D_patient, y_axis='log', x_axis='time', ax=ax2)
        ax2.set_title("Patient's Spectrogram")
        plt.savefig(spectrogram_path)
        plt.close(fig)
        plots['spectrogram'] = spectrogram_path

        # Energy/RMS Comparison
        energy_path = os.path.join(output_dir, "energy_comparison.png")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        doctor_energy = doctor_features['raw_data']['rms_energy']
        patient_energy = patient_features['raw_data']['rms_energy']
        doctor_times = librosa.times_like(doctor_energy)
        patient_times = librosa.times_like(patient_energy)
        doctor_energy = doctor_energy[:len(doctor_times)]
        patient_energy = patient_energy[:len(patient_times)]
        ax1.plot(doctor_times, doctor_energy)
        ax1.set_title("Doctor's Energy Level")
        ax2.plot(patient_times, patient_energy)
        ax2.set_title("Patient's Energy Level")
        plt.savefig(energy_path)
        plt.close(fig)
        plots['energy'] = energy_path

        return plots
