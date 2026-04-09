import os
import numpy as np
from pathlib import Path

def evaluate():
    base_dir = "/home/tr/MEGA/programming2/L4T2/ML_prj_part2_v2"
    real_dir = Path(base_dir) / "Testing/Mozilla_s1/output/real"
    fake_dir = Path(base_dir) / "Testing/Mozilla_s1/output/fake"
    results_file = Path(base_dir) / "Testing/Mozilla_s1/output/results-Mozilla_s1.txt"

    tp = 0 # True Positives: Ground truth Fake, Predicted Fake
    fn = 0 # False Negatives: Ground truth Fake, Predicted Real
    tn = 0 # True Negatives: Ground truth Real, Predicted Real
    fp = 0 # False Positives: Ground truth Real, Predicted Fake

    def process_dir(directory, is_fake):
        nonlocal tp, fn, tn, fp
        if not directory.exists():
            print(f"Directory not found: {directory}")
            return
            
        for npz_file in directory.glob("*.npz"):
            try:
                data = np.load(npz_file)
                if 'global_score' in data:
                    score = float(data['global_score'])
                    # score >= 0.2 means Fake, score < 0.2 means Real
                    pred_fake = (score >= 0.2)
                    
                    if is_fake:
                        if pred_fake:
                            tp += 1
                        else:
                            fn += 1
                    else:
                        if pred_fake:
                            fp += 1
                        else:
                            tn += 1
                else:
                    print(f"'global_score' missing in {npz_file}")
            except Exception as e:
                print(f"Error reading {npz_file}: {e}")

    # Process fake directory (Ground truth: Fake)
    process_dir(fake_dir, is_fake=True)
    # Process real directory (Ground truth: Real)
    process_dir(real_dir, is_fake=False)

    total_fake = tp + fn
    total_real = tn + fp
    total_samples = total_fake + total_real

    if total_samples == 0:
        print("No samples found! Check if the directories contain .npz files.")
        return

    accuracy = (tp + tn) / total_samples if total_samples > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    results = []
    results.append(f"total samples: {total_samples}")
    results.append(f"total fake: {total_fake}")
    results.append(f"total real: {total_real}")
    results.append("")
    results.append("Confusion Matrix:")
    results.append(f"{'':<13} | {'Predicted Fake':<15} | {'Predicted Real':<15}")
    results.append("-" * 51)
    results.append(f"{'Actual Fake':<13} | {tp:<15} | {fn:<15}")
    results.append(f"{'Actual Real':<13} | {fp:<15} | {tn:<15}")
    results.append("-" * 51)
    results.append("")
    results.append(f"Accuracy: {accuracy:.4f}")
    results.append(f"Precision: {precision:.4f}")
    results.append(f"Recall: {recall:.4f}")
    results.append(f"F1 score: {f1_score:.4f}")

    output_text = "\n".join(results)
    
    # Save to results.txt
    with open(results_file, "w") as f:
        f.write(output_text + "\n")
        
    print(output_text)
    print(f"\nResults successfully written to: {results_file}")

if __name__ == "__main__":
    evaluate()
