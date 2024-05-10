import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def load_results(npy_file):
    """Load results from a numpy file."""
    return np.load(npy_file, allow_pickle=True)

def plot_results(results):
    """Plot the heart rate estimations from the model and Garmin data."""
    sns.set(style="whitegrid")
    
    # Create time series
    times = [result['start_time'] for result in results]
    model_hrs = [result['model_hr'] for result in results]
    garmin_hrs = [result['garmin_hr'] for result in results]

    plt.figure(figsize=(12, 6))

    plt.plot(times, model_hrs, label='Model-Inferred Heart Rate', marker='o', linestyle='-', color='blue')
    plt.plot(times, garmin_hrs, label='Garmin-Measured Heart Rate', marker='x', linestyle='-', color='red')

    plt.title('Heart Rate Comparison Over Time')
    plt.xlabel('Time')
    plt.ylabel('Heart Rate (bpm)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig('cphys_results_comparison.png')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize Heart Rate Data from cPhys Model')
    parser.add_argument('--npy_file', type=str, required=True, help='Path to the .npy results file')
    args = parser.parse_args()
    
    results = load_results(args.npy_file)
    plot_results(results)

if __name__ == '__main__':
    main()
