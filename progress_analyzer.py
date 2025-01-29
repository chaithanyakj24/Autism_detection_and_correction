import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_progress(patient_name, word):
    progress_file = f"data/progress/{patient_name}_progress.csv"

    # Check if the progress file exists
    if not os.path.exists(progress_file):
        print(f"No progress data found for {patient_name}.")
        return

    # Load progress data
    df = pd.read_csv(progress_file)

    # Generate the plot
    plt.figure(figsize=(8, 6))

    # Plot a bar graph or line graph
    plt.plot(df["Session"], df["Pitch"], marker='o', label="Average Pitch", color="blue", linestyle="--")
    # Uncomment below to create a bar graph instead
    # plt.bar(df["Session"], df["Pitch"], label="Average Pitch", color="skyblue")

    plt.title(f"Pronunciation Progress for '{word}'")
    plt.xlabel("Sessions")
    plt.ylabel("Pitch (Hz)")
    plt.legend()
    plt.grid(True)

    # Save the graph
    output_file = f"data/progress/{patient_name}_{word}_progress.png"
    plt.savefig(output_file)
    print(f"Progress graph saved at {output_file}")

    # Show the graph
    plt.show()





