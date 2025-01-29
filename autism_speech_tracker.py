import pyaudio
import wave
import pyttsx3
import os
import speech_recognition as sr
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import time
import soundfile as sf

class PronunciationTrainer:
    def __init__(self):  # Fixed the constructor method
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Speed of speech
        self.engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)
        
        # Get available voices and set to a female voice if available
        voices = self.engine.getProperty('voices')
        if len(voices) > 1:  # If multiple voices are available
            self.engine.setProperty('voice', voices[1].id)  # Usually female voice
    
    def record_audio(self, file_name, duration=3):
        chunk = 1024
        sample_format = pyaudio.paInt16
        channels = 1
        rate = 44100
        
        p = pyaudio.PyAudio()
        
        # Countdown before recording
        for i in range(3, 0, -1):
            print(f"Recording will start in {i}...")
            time.sleep(1)
            
        print("Recording... Speak now!")
        
        stream = p.open(format=sample_format,
                       channels=channels,
                       rate=rate,
                       input=True,
                       frames_per_buffer=chunk)
        
        frames = []
        for _ in range(0, int(rate / chunk * duration)):
            data = stream.read(chunk)
            frames.append(data)
        
        print("Recording complete!")
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        wf = wave.open(file_name, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
        wf.close()
    
    def speak_word(self, word):
        print(f"\nListening to correct pronunciation of: {word}")
        self.engine.say(word)
        self.engine.runAndWait()
    
    def visualize_pronunciation(self, audio_file, attempt_number):
        try:
            # Read the audio file
            data, sample_rate = sf.read(audio_file)
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Plot waveform
            time_points = np.linspace(0, len(data)/sample_rate, len(data))
            ax1.plot(time_points, data)
            ax1.set_title('Waveform')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Amplitude')
            
            # Plot spectrogram
            frequencies, times, spectrogram = signal.spectrogram(data, sample_rate)
            ax2.pcolormesh(times, frequencies, 10 * np.log10(spectrogram))
            ax2.set_ylabel('Frequency [Hz]')
            ax2.set_xlabel('Time [sec]')
            ax2.set_title('Spectrogram')
            
            plt.tight_layout()
            plot_filename = f'pronunciation_plot_{attempt_number}.png'
            plt.savefig(plot_filename)
            plt.close()
            
            return plot_filename
        except Exception as e:
            print(f"Error creating visualization: {e}")
            return None
    
    def compare_pronunciation(self, input_word, user_audio_file):
        recognizer = sr.Recognizer()
        
        with sr.AudioFile(user_audio_file) as source:
            audio = recognizer.record(source)
        
        try:
            recognized_text = recognizer.recognize_google(audio)
            print(f"You said: {recognized_text}")
            
            # More flexible comparison
            recognized_words = recognized_text.lower().strip().split()
            input_words = input_word.lower().strip().split()
            
            # Calculate similarity score
            matches = sum(1 for word in input_words if word in recognized_words)
            similarity = matches / len(input_words)
            
            print(f"Pronunciation accuracy: {similarity * 100:.1f}%")
            return similarity > 0.7  # 70% threshold for acceptance
            
        except sr.UnknownValueError:
            print("Could not understand the audio. Please speak more clearly.")
            return False
        except sr.RequestError:
            print("Offline speech recognition not available. Check your internet connection.")
            return False
    
    def train(self):
        try:
            print("Welcome to Pronunciation Trainer!")
            input_word = input("\nEnter a word or phrase to practice: ")
            attempt_count = 0
            max_attempts = 3
            
            while attempt_count < max_attempts:
                print(f"\nAttempt {attempt_count + 1} of {max_attempts}")
                
                # Speak the word
                self.speak_word(input_word)
                time.sleep(1)  # Brief pause
                
                # Record user's pronunciation
                user_audio_file = f"user_pronunciation_{attempt_count}.wav"
                self.record_audio(user_audio_file)
                
                # Generate visualization
                plot_file = self.visualize_pronunciation(user_audio_file, attempt_count)
                if plot_file:
                    print(f"Visualization saved as: {plot_file}")
                
                # Compare pronunciations
                print("\nAnalyzing your pronunciation...")
                if self.compare_pronunciation(input_word, user_audio_file):
                    print("\nExcellent! Your pronunciation matches the word!")
                    break
                else:
                    attempt_count += 1
                    if attempt_count < max_attempts:
                        print(f"\nLet's try again!")
                    else:
                        print("\nKeep practicing! You'll get better with time.")
                
                # Clean up audio file
                try:
                    os.remove(user_audio_file)
                except:
                    pass
                
        except KeyboardInterrupt:
            print("\nProgram terminated by user.")
        except Exception as e:
            print(f"An error occurred: {e}")

def main():
    trainer = PronunciationTrainer()
    trainer.train()

if __name__ == "__main__":
    main()
