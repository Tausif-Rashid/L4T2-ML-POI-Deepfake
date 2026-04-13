import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import f1_score

def load_scores(directory):
    scores = []
    if not directory.exists():
        print(f"Directory not found: {directory}")
        return scores
        
    for npz_file in directory.glob("*.npz"):
        try:
            data = np.load(npz_file)
            if 'global_score' in data:
                score = float(data['global_score'])
                if not np.isnan(score):
                    scores.append(score)
        except Exception as e:
            pass
    return scores

def main():
    base_dir = "/home/tr/MEGA/programming2/L4T2/ML_prj_part2_v2"
    datasets = ["Mozilla_s1", "Sust", "yt1"]
    
    plt.figure(figsize=(10, 6))
    
    for dataset in datasets:
        print(f"Processing dataset: {dataset}...")
        real_dir = Path(base_dir) / f"Testing/{dataset}/output/real"
        fake_dir = Path(base_dir) / f"Testing/{dataset}/output/fake"
        
        real_scores = load_scores(real_dir)
        fake_scores = load_scores(fake_dir)
        
        if not real_scores and not fake_scores:
            print(f"  -> Skipping {dataset}: No data found.")
            continue
            
        print(f"  -> Loaded {len(real_scores)} real samples and {len(fake_scores)} fake samples.")
        
        # 0 for Real, 1 for Fake
        y_true = np.array([0] * len(real_scores) + [1] * len(fake_scores))
        y_scores = np.array(real_scores + fake_scores)
        
        if len(y_scores) == 0:
            continue
            
        # Range of thresholds to test
        thresholds = np.linspace(y_scores.min(), y_scores.max(), 1000)
        f1_values = []
        
        for t in thresholds:
            y_pred = (y_scores >= t).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            f1_values.append(f1)
            
        # Plot the curve for this dataset
        line, = plt.plot(thresholds, f1_values, label=f"{dataset}", linewidth=2)
        
        # Plot and tag the maximum point
        best_f1 = max(f1_values)
        best_idx = np.argmax(f1_values)
        best_t = thresholds[best_idx]
        
        plt.plot(best_t, best_f1, 'o', color=line.get_color())
        plt.annotate(
            f"Max: {best_f1:.2f}\n@ {best_t:.2f}",
            (best_t, best_f1),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=line.get_color(), lw=1, alpha=0.8)
        )

    plt.title("F1 Score vs Classification Threshold")
    plt.xlabel("Threshold Score")
    plt.ylabel("F1 Score")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    output_img = Path(base_dir) / "f1_score_vs_threshold.png"
    plt.savefig(output_img, dpi=300)
    print(f"\nGraph successfully generated and saved to: {output_img}")

if __name__ == "__main__":
    main()
