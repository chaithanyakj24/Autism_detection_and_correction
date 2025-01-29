import gradio as gr
from speech_analyzer import SpeechAnalyzer
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import tempfile
import os
import time
import re
from nltk.tokenize import word_tokenize
import nltk
import phonemizer
from phonemizer.backend import EspeakBackend
import pronouncing
# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Initialize components
analyzer = SpeechAnalyzer()
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
tts_engine = pyttsx3.init()


class LipMovementGenerator:
    def __init__(self):
        # Define viseme shapes for different sounds
        self.viseme_shapes = {
            'REST': {'width': 40, 'height': 15, 'roundness': 0.8},  # Neutral position
            'AI': {'width': 60, 'height': 25, 'roundness': 0.4},    # as in "day"
            'O': {'width': 40, 'height': 40, 'roundness': 1.0},     # as in "go"
            'U': {'width': 30, 'height': 30, 'roundness': 1.0},     # as in "mood"
            'E': {'width': 55, 'height': 20, 'roundness': 0.5},     # as in "bee"
            'I': {'width': 45, 'height': 20, 'roundness': 0.5},     # as in "bit"
            'A': {'width': 50, 'height': 40, 'roundness': 0.8},     # as in "cat"
            'FV': {'width': 40, 'height': 15, 'roundness': 0.7},    # as in "five"
            'BMP': {'width': 35, 'height': 10, 'roundness': 0.9},   # as in "bump"
            'L': {'width': 45, 'height': 25, 'roundness': 0.6},     # as in "lip"
            'WQ': {'width': 30, 'height': 25, 'roundness': 1.0},    # as in "quick"
            'TH': {'width': 40, 'height': 20, 'roundness': 0.6},    # as in "thin"
        }

        # Mapping from phonemes to visemes
        self.phoneme_to_viseme = {
            'AH': 'A', 'AE': 'A', 'AA': 'A',
            'IY': 'E', 'IH': 'I',
            'UW': 'U', 'UH': 'U',
            'OW': 'O', 'AO': 'O',
            'F': 'FV', 'V': 'FV',
            'B': 'BMP', 'M': 'BMP', 'P': 'BMP',
            'L': 'L',
            'W': 'WQ',
            'TH': 'TH', 'DH': 'TH',
        }

    def get_phonemes(self, word):
        """Convert word to phonemes using CMU Pronouncing Dictionary"""
        pronunciations = pronouncing.phones_for_word(word)
        if pronunciations:
            # Use the first pronunciation
            phonemes = pronunciations[0].split()
            return phonemes
        else:
            # Fallback: return each character as a phoneme
            return [char.upper() for char in word]

    def get_viseme_sequence(self, word):
        """Convert word to sequence of visemes with timing"""
        phonemes = self.get_phonemes(word)
        viseme_sequence = []
        
        for phoneme in phonemes:
            # Find matching viseme or use REST as default
            viseme = self.phoneme_to_viseme.get(phoneme, 'REST')
            viseme_sequence.append(viseme)
        
        return viseme_sequence

    def interpolate_shapes(self, shape1, shape2, factor):
        """Interpolate between two lip shapes"""
        return {
            'width': int(shape1['width'] * (1 - factor) + shape2['width'] * factor),
            'height': int(shape1['height'] * (1 - factor) + shape2['height'] * factor),
            'roundness': shape1['roundness'] * (1 - factor) + shape2['roundness'] * factor
        }

    def draw_face(self, frame, shape_params, word_text=""):
        """Draw the face with lips based on shape parameters"""
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2

        # Draw face circle
        cv2.circle(frame, (center_x, center_y), 100, (200, 200, 200), -1)

        # Draw eyes
        eye_y = center_y - 30
        cv2.circle(frame, (center_x - 40, eye_y), 10, (70, 70, 70), -1)
        cv2.circle(frame, (center_x + 40, eye_y), 10, (70, 70, 70), -1)

        # Draw lips
        mouth_y = center_y + 20
        lip_width = shape_params['width']
        lip_height = shape_params['height']
        roundness = shape_params['roundness']

        # Upper lip
        cv2.ellipse(frame, 
                    (center_x, mouth_y - lip_height//4),
                    (lip_width, lip_height//2),
                    0, 0, 180,
                    (150, 50, 50),
                    -1)
        
        # Lower lip
        cv2.ellipse(frame,
                    (center_x, mouth_y + lip_height//4),
                    (lip_width, int(lip_height//2 * roundness)),
                    0, 180, 360,
                    (150, 50, 50),
                    -1)

        # Add word text
        cv2.putText(frame, word_text, (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return frame

    def generate_animation(self, word, output_path, fps=24):
        """Generate a dynamic lip movement animation for the given word"""
        try:
            viseme_sequence = self.get_viseme_sequence(word)
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (400, 400))
            
            # Parameters for smooth animation
            frames_per_viseme = 8
            transition_frames = 4
            
            # Start with neutral position
            current_shape = self.viseme_shapes['REST']
            
            # Generate frames
            for i in range(len(viseme_sequence)):
                target_shape = self.viseme_shapes[viseme_sequence[i]]
                
                # Generate transition frames
                for j in range(transition_frames):
                    factor = j / transition_frames
                    interpolated_shape = self.interpolate_shapes(current_shape, target_shape, factor)
                    
                    frame = np.zeros((400, 400, 3), dtype=np.uint8)
                    frame = self.draw_face(frame, interpolated_shape, word)
                    out.write(frame)
                
                # Hold the viseme shape
                for _ in range(frames_per_viseme - transition_frames):
                    frame = np.zeros((400, 400, 3), dtype=np.uint8)
                    frame = self.draw_face(frame, target_shape, word)
                    out.write(frame)
                
                current_shape = target_shape
            
            # Return to neutral position
            for j in range(transition_frames):
                factor = j / transition_frames
                interpolated_shape = self.interpolate_shapes(
                    current_shape, self.viseme_shapes['REST'], factor)
                frame = np.zeros((400, 400, 3), dtype=np.uint8)
                frame = self.draw_face(frame, interpolated_shape, word)
                out.write(frame)
            
            out.release()
            return output_path
            
        except Exception as e:
            print(f"Error generating animation: {e}")
            return None

# Rest of your existing functions (analyze_audio, compare_audio, etc.) remain the same
def analyze_audio(file_path):
    try:
        features = analyzer.analyze_audio(file_path)
        classification, detailed_results = analyzer.classify_audio(features)
        return {
            "Classification": classification,
            "Pitch (Hz)": features["pitch"],
            "Pauses Duration (s)": features["pauses_duration"],
            "Speech Rate (syllables/sec)": features["speech_rate"],
            "Tone Spectral Centroid (Hz)": features["tone_spectral_centroid"],
            "Detailed Results": detailed_results,
        }
    except Exception as e:
        return {"error": str(e)}

def compare_audio(doctor_file, patient_file):
    try:
        plots = analyzer.generate_comparison_plots(doctor_file, patient_file)
        return plots['waveform'], plots['pitch'], plots['spectrogram'], plots['energy']
    except Exception as e:
        return f"Error: {str(e)}", "", "", ""

def process_lip_movement(video_path, word):
    try:
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, "processed_video.mp4")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Draw lip landmarks
                    for idx in [61, 62, 63, 64, 65, 66, 67, 68, 78, 80]:
                        x = int(face_landmarks.landmark[idx].x * width)
                        y = int(face_landmarks.landmark[idx].y * height)
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                    
                    # Calculate and display lip movement feedback
                    top_lip = face_landmarks.landmark[13].y
                    bottom_lip = face_landmarks.landmark[14].y
                    lip_distance = abs(top_lip - bottom_lip)
                    feedback = "Good lip movement!" if lip_distance > 0.05 else "Open mouth wider!"
                    cv2.putText(frame, feedback, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(frame, f"Say: '{word}'", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            out.write(frame)
            
        cap.release()
        out.release()
        return output_path
    except Exception as e:
        return str(e)

def record_webcam(word, duration=10):
    try:
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, "webcam_recording.mp4")
        
        cap = cv2.VideoCapture(0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 30
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        start_time = time.time()
        
        while (time.time() - start_time) < duration:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Draw lip landmarks
                    for idx in [61, 62, 63, 64, 65, 66, 67, 68, 78, 80]:
                        x = int(face_landmarks.landmark[idx].x * width)
                        y = int(face_landmarks.landmark[idx].y * height)
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                    
                    # Calculate and display lip movement feedback
                    top_lip = face_landmarks.landmark[13].y
                    bottom_lip = face_landmarks.landmark[14].y
                    lip_distance = abs(top_lip - bottom_lip)
                    feedback = "Good lip movement!" if lip_distance > 0.05 else "Open mouth wider!"
                    cv2.putText(frame, feedback, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            remaining_time = int(duration - (time.time() - start_time))
            cv2.putText(frame, f"Recording... {remaining_time}s", (width-200, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Say: '{word}'", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            out.write(frame)
            
        cap.release()
        out.release()
        return output_path
    except Exception as e:
        return str(e)

def generate_synthetic_movement(word):
    try:
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, "synthetic_movement.mp4")
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 10, (300, 300))
        
        for _ in range(3):  # Repeat animation 3 times
            for i in range(10):
                frame = np.zeros((300, 300, 3), dtype=np.uint8)
                mouth_height = 10 + (i * 5) if i < 5 else 50 - (i * 5)
                cv2.ellipse(frame, (150, 150), (50, mouth_height), 0, 0, 180, (0, 255, 0), -1)
                cv2.putText(frame, word, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                out.write(frame)
        
        out.release()
        return output_path
    except Exception as e:
        return str(e)

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Speech Analysis and Comparison Tool")
    
    # Single Audio Analysis Tab
    with gr.Tab("Analyze Single Audio"):
        audio_file = gr.Audio(type="filepath", label="Upload Audio File")
        analysis_output = gr.JSON(label="Audio Analysis Features")
        analyze_button = gr.Button("Analyze")
        analyze_button.click(analyze_audio, inputs=audio_file, outputs=analysis_output)
    
    # Audio Comparison Tab
    with gr.Tab("Compare Doctor vs. Patient"):
        doctor_file = gr.Audio(type="filepath", label="Upload Doctor's Audio")
        patient_file = gr.Audio(type="filepath", label="Upload Patient's Audio")
        waveform_output = gr.Image(type="filepath", label="Waveform Comparison")
        pitch_output = gr.Image(type="filepath", label="Pitch Comparison")
        spectrogram_output = gr.Image(type="filepath", label="Spectrogram Comparison")
        energy_output = gr.Image(type="filepath", label="Energy Comparison")
        compare_button = gr.Button("Compare")
        compare_button.click(
            compare_audio,
            inputs=[doctor_file, patient_file],
            outputs=[waveform_output, pitch_output, spectrogram_output, energy_output]
        )
    
    # Lip Movement Analysis Tab
    with gr.Tab("Lip Movement Analysis"):
        word_input = gr.Textbox(label="Enter word to practice")
        
        with gr.Tabs():
            # Upload Video Tab
            with gr.Tab("Upload Video"):
                video_input = gr.Video(label="Upload video")
                processed_video = gr.Video(label="Processed Video with Feedback")
                analyze_video_button = gr.Button("Analyze Uploaded Video")
                analyze_video_button.click(
                    process_lip_movement,
                    inputs=[video_input, word_input],
                    outputs=processed_video
                )
            
            # Record Video Tab
            with gr.Tab("Record Video"):
                duration_slider = gr.Slider(minimum=5, maximum=30, value=10, step=5, 
                                         label="Recording Duration (seconds)")
                recorded_video = gr.Video(label="Recorded Video with Feedback")
                record_button = gr.Button("Start Recording")
                record_button.click(
                    record_webcam,
                    inputs=[word_input, duration_slider],
                    outputs=recorded_video
                )
        
        # Synthetic Movement (available for both tabs)
        synthetic_video = gr.Video(label="AI Generated Lip Movement")
        generate_synthetic_button = gr.Button("Generate Synthetic Movement")
        generate_synthetic_button.click(
            generate_synthetic_movement,
            inputs=word_input,
            outputs=synthetic_video
        )

if __name__ == "__main__":
    demo.launch()