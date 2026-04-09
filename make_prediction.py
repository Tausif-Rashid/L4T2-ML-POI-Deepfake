import os
import csv
import random
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve

def load_data(directory, label):
    data_list = []
    if not directory.exists():
        print(f"Directory not found: {directory}")
        return data_list
        
    for npz_file in directory.glob("*.npz"):
        try:
            data = np.load(npz_file)
            if 'global_score' in data:
                score = float(data['global_score'])
                if np.isnan(score):
                    print(f"Warning: 'global_score' is NaN in {npz_file.name}, skipping.")
                    continue
                data_list.append({
                    'filename': npz_file.name,
                    'filepath': str(npz_file),
                    'global_score': score,
                    'label': label  # 0 for Real, 1 for Fake
                })
            else:
                print(f"'global_score' missing in {npz_file}")
        except Exception as e:
            print(f"Error reading {npz_file}: {e}")
    return data_list

def evaluate(method='percentile_90'):
    # method can be 'percentile_90', 'f1_optimized', 'eer'
    base_dir = "/home/tr/MEGA/programming2/L4T2/ML_prj_part2_v2"

    test_dataset = "Mozilla_s1"
    real_dir = Path(base_dir) / f"Testing/{test_dataset}/output/real"
    fake_dir = Path(base_dir) / f"Testing/{test_dataset}/output/fake"
    results_file = Path(base_dir) / f"Testing/{test_dataset}/output/results_prediction-{method}.txt"
    dataset_file = Path(base_dir) / f"Testing/{test_dataset}/output/all_scores_dataset.csv"

    # Load data
    real_data = load_data(real_dir, 0)
    fake_data = load_data(fake_dir, 1)

    if len(real_data) == 0 and len(fake_data) == 0:
         print("No samples found! Check if the directories contain .npz files.")
         return

    # Shuffle robustly with seed
    random.seed(1)
    random.shuffle(real_data)
    random.shuffle(fake_data)

    # Split sizes
    real_eval_size = int(0.3 * len(real_data))
    fake_eval_size = int(0.3 * len(fake_data))

    eval_data = []
    test_data = []

    if method == 'percentile_90':
        # Only use Real for eval
        for i, item in enumerate(real_data):
            if i < real_eval_size:
                item['split'] = 'eval'
                eval_data.append(item)
            else:
                item['split'] = 'test'
                test_data.append(item)
        
        for item in fake_data:
            item['split'] = 'test'
            test_data.append(item)
            
    elif method in ['f1_optimized', 'eer']:
        # Use both Real and Fake for eval
        for i, item in enumerate(real_data):
            if i < real_eval_size:
                item['split'] = 'eval'
                eval_data.append(item)
            else:
                item['split'] = 'test'
                test_data.append(item)
                
        for i, item in enumerate(fake_data):
            if i < fake_eval_size:
                item['split'] = 'eval'
                eval_data.append(item)
            else:
                item['split'] = 'test'
                test_data.append(item)
    else:
        print(f"Unknown method {method}")
        return

    all_data = eval_data + test_data
    
    # Save the dataset
    dataset_file.parent.mkdir(parents=True, exist_ok=True)
    with open(dataset_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'filepath', 'global_score', 'label', 'split'])
        for item in all_data:
            writer.writerow([item['filename'], item['filepath'], item['global_score'], item['label'], item['split']])
            
    print(f"Saved dataset with all scores to {dataset_file}\n")

    # Learn Threshold
    threshold = 0.5
    if method == 'percentile_90':
        eval_scores = [item['global_score'] for item in eval_data] # only contains real data here
        # 90% below threshold. So np.percentile(..., 90) gives the value.
        threshold = np.percentile(eval_scores, 90)
        #print(f"Learned Threshold (90th percentile of Real Eval): {threshold:.4f}")
    
    elif method == 'f1_optimized':
        y_true = np.array([item['label'] for item in eval_data])
        y_scores = np.array([item['global_score'] for item in eval_data])
        
        best_f1 = -1
        # test a range of thresholds
        for t in np.linspace(y_scores.min(), y_scores.max(), 1000):
            y_pred = (y_scores >= t).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                threshold = t
        #print(f"Learned Threshold (F1 Optimized - Best F1={best_f1:.4f}): {threshold:.4f}")
        
    elif method == 'eer':
        y_true = np.array([item['label'] for item in eval_data])
        y_scores = np.array([item['global_score'] for item in eval_data])
        
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        fnr = 1 - tpr
        eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
        threshold = eer_threshold
        #print(f"Learned Threshold (Equal Error Rate): {threshold:.4f}")

    # Testing
    y_test_true = np.array([item['label'] for item in test_data])
    y_test_scores = np.array([item['global_score'] for item in test_data])
    
    y_test_pred = (y_test_scores >= threshold).astype(int)
    
    cm = confusion_matrix(y_test_true, y_test_pred)
    # cm order:
    # [ [tn, fp],
    #   [fn, tp] ]
    if cm.shape == (2,2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Handle cases where only one class might be present in a small test set (unlikely but possible)
        tn, fp, fn, tp = confusion_matrix(y_test_true, y_test_pred, labels=[0, 1]).ravel()

    total_fake = tp + fn
    total_real = tn + fp
    total_samples = total_fake + total_real
    
    accuracy = accuracy_score(y_test_true, y_test_pred) if total_samples > 0 else 0
    precision = precision_score(y_test_true, y_test_pred, zero_division=0) if total_samples > 0 else 0
    recall = recall_score(y_test_true, y_test_pred, zero_division=0) if total_samples > 0 else 0
    f1 = f1_score(y_test_true, y_test_pred, zero_division=0) if total_samples > 0 else 0

    results = []
    results.append(f"Method used: {method}")
    results.append(f"Calculated threshold: {threshold:.4f}")
    results.append("-" * 51)
    results.append(f"Testing on Test Set (Total: {total_samples})")
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
    results.append(f"F1 score: {f1:.4f}")

    output_text = "\n".join(results)
    
    # Save to results.txt
    with open(results_file, "w") as f:
        f.write(output_text + "\n")
        
    print("\n" + output_text)
    print(f"\nResults written to: {results_file}")

if __name__ == "__main__":
    # change the method here to:
    # 1. 'percentile_90': Zero-day Fake assumption (uses 30% Real Data only for evaluation)
    # 2. 'f1_optimized': Finds ideal threshold maximizing F1 (uses 30% Real + 30% Fake data for eval)
    # 3. 'eer': Equal Error Rate (uses 30% Real + 30% Fake Data for eval)
    
    method = 'f1_optimized' # Option to change
    
    print(f"Starting evaluation using '{method}' threshold learning...")
    evaluate(method=method)
