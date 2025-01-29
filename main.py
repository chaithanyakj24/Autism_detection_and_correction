import os
import datetime
from lip_tracker import detect_and_track_lips, initialize_csv
from audio_recorder import record_audio
from progress_analyzer import analyze_progress

def main():
    # Get patient details and word to practice
    patient_name = input("Enter patient name: ")
    test_date = datetime.datetime.now().strftime("%Y-%m-%d")
    input_word = input("Enter the word to practice pronunciation: ")

    # File paths
    lip_file = os.path.join("data", "lips", f"{patient_name}_{test_date}_lips.csv")
    audio_file = os.path.join("data", "audio", f"{patient_name}_{test_date}_audio.wav")

    # Ensure directories exist
    os.makedirs(os.path.dirname(lip_file), exist_ok=True)
    os.makedirs(os.path.dirname(audio_file), exist_ok=True)

    # Initialize lip tracking data file
    initialize_csv(lip_file)

    # Step 1: Track lip movement and save data
    print("Step 1: Tracking lip movements...")
    detect_and_track_lips(input_word, lip_file)

    # Step 2: Record audio
    print("Step 2: Recording audio...")
    record_audio(audio_file, duration=5)

    # Step 3: Analyze progress
    print("Step 3: Analyzing progress...")
    analyze_progress(patient_name, input_word)

    print("Process completed. Results saved in 'data/progress/'.")

if __name__ == "__main__":
    main()
