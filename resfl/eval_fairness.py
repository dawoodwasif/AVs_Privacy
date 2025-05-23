# fairness_evaluation.py

import json
import os
import torch
from collections import defaultdict
from ultralytics import YOLO
from sklearn.metrics import average_precision_score
from tqdm import tqdm
import numpy as np
from scipy.stats import wasserstein_distance

FED_ALG = 'FedAvg'

GT_CLASS = 0  # Assuming that the ground truth class for the object is 0

def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    xi1 = max(x1_min, x2_min)
    yi1 = max(y1_min, y2_min)
    xi2 = min(x1_max, x2_max)
    yi2 = min(y1_max, y2_max)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area

def convert_to_xyxy(bounding_box_str):
    bounding_box = json.loads(bounding_box_str)
    x_min = bounding_box['x']
    y_min = bounding_box['y']
    x_max = x_min + bounding_box['width']
    y_max = y_min + bounding_box['height']
    return [x_min, y_min, x_max, y_max]

def get_skin_tones(gt):
    tones = []
    for i in range(1, 11):
        if gt[f'skin_tone_{i}'] > 0:
            tones.append(i)
    if not tones or gt[f'skin_tone_na'] > 0:
        tones.append('na')
    return tones

def calculate_map(preds, gts):
    if not preds or not gts:
        return 0.0
    y_true = []
    y_scores = []
    for pred, score in preds:
        matched = False
        for gt in gts:
            if calculate_iou(pred, gt) > 0.8:
                matched = True
                break
        y_true.append(int(matched))
        y_scores.append(score)
    return y_true, y_scores

def assess_fairness(model, test_images_path, metadata_path):
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Initialize containers for predictions and ground truths
    skin_tone_preds = defaultdict(list)
    skin_tone_gts = defaultdict(list)

    # Get the list of test images
    test_images = [os.path.join(test_images_path, f) for f in os.listdir(test_images_path) if f.endswith('.jpg')]

    # Perform inference on the test dataset
    for image_path in tqdm(test_images):
        results = model(image_path)
        
        result = results[0]
        image_id = (os.path.splitext(os.path.basename(result.path))[0])[3:]

        print(image_id)
        if image_id not in metadata:
            print(f"{image_id} is not found!")
            continue

        ground_truths = metadata[image_id]['persons']
        predictions = result.boxes.xyxy.cpu().numpy()  # move tensor to CPU and convert to numpy
        confidences = result.boxes.conf.cpu().numpy()
        pred_cls = result.boxes.cls.cpu().numpy()

        for gt in ground_truths:
            gt_box = convert_to_xyxy(gt['bounding_box'])
            gt_class = GT_CLASS
            skin_tones = get_skin_tones(gt)
            for skin_tone in skin_tones:
                skin_tone_gts[skin_tone].append(gt_box)


        for id, pred in enumerate(predictions):
            x_min, y_min, x_max, y_max = pred.tolist()
            conf, pcls = confidences[id], int(pred_cls[id])
            pred_box = [x_min, y_min, x_max, y_max]
            pred_score = conf
            for gt in ground_truths:
                if pcls == gt_class:
                    skin_tones = get_skin_tones(gt)
                    for skin_tone in skin_tones:
                        skin_tone_preds[skin_tone].append((pred_box, pred_score))
                    break


    # Calculate mAP and collect score distributions for each skin tone
    skin_tone_maps = {}
    for tone in skin_tone_gts:
        y_true, y_scores = calculate_map(skin_tone_preds[tone], skin_tone_gts[tone])
        if y_true:  # Only calculate mAP if there are valid entries
            skin_tone_maps[tone] = average_precision_score(y_true, y_scores)
        else:
            skin_tone_maps[tone] = 0.0

    # Print the mAP for each skin tone
    for tone, mAP in skin_tone_maps.items():
        print(f'Skin tone {tone}: Average mAP: {mAP:.4f}')


    # Calculate mAP and collect score distributions for each skin tone
    skin_tone_scores = {tone: calculate_map(skin_tone_preds[tone], skin_tone_gts[tone])[1] for tone in skin_tone_gts}

    # Calculate fairness metrics
    fairness_metrics = calculate_fairness_metrics(skin_tone_scores)
    print_fairness_metrics(fairness_metrics)

    return skin_tone_scores

def calculate_fairness_metrics(skin_tone_scores):
    # Convert to numpy arrays
    score_distributions = [np.array(scores) for scores in skin_tone_scores.values() if scores]

    # Calculate Worst-case Difference and Best-case Difference
    max_diff = float('-inf')
    min_diff = float('inf')
    
    for i in range(len(score_distributions)):
        for j in range(i + 1, len(score_distributions)):
            diff = np.abs(score_distributions[i].mean() - score_distributions[j].mean())
            max_diff = max(max_diff, diff)
            min_diff = min(min_diff, diff)

    # Calculate Wasserstein-2 Metric
    wasserstein_metrics = []
    for i in range(len(score_distributions)):
        for j in range(i + 1, len(score_distributions)):
            wasserstein_metrics.append(wasserstein_distance(score_distributions[i], score_distributions[j]))

    worst_case_wasserstein = max(wasserstein_metrics) if wasserstein_metrics else 0

    return {
        'Worst-case Difference': max_diff,
        'Best-case Difference': min_diff,
        'Wasserstein-2 Metric': worst_case_wasserstein
    }

def print_fairness_metrics(fairness_metrics):
    print("\nFairness Metrics:")
    print(f"Worst-case Difference: {fairness_metrics['Worst-case Difference']:.4f}")
    print(f"Best-case Difference: {fairness_metrics['Best-case Difference']:.4f}")
    print(f"Wasserstein-2 Metric: {fairness_metrics['Wasserstein-2 Metric']:.4f}")
def calculate_fairness_metrics_extended(skin_tone_scores, skin_tone_maps, skin_tone_gts, skin_tone_preds):
    # Convert to numpy arrays for mAP
    score_distributions = [np.array(scores) for scores in skin_tone_scores.values() if scores]

    # Calculate Worst-case Difference and Best-case Difference
    max_diff = float('-inf')
    min_diff = float('inf')
    
    for i in range(len(score_distributions)):
        for j in range(i + 1, len(score_distributions)):
            diff = np.abs(score_distributions[i].mean() - score_distributions[j].mean())
            max_diff = max(max_diff, diff)
            min_diff = min(min_diff, diff)

    # Calculate Wasserstein-2 Metric
    wasserstein_metrics = []
    for i in range(len(score_distributions)):
        for j in range(i + 1, len(score_distributions)):
            wasserstein_metrics.append(wasserstein_distance(score_distributions[i], score_distributions[j]))

    worst_case_wasserstein = max(wasserstein_metrics) if wasserstein_metrics else 0

    # Compute additional fairness metrics: Acc, |1-DI|, ∆EOP, ∆EODD
    accuracies = []
    tprs = []
    fprs = []
    
    for tone, preds in skin_tone_preds.items():
        # True positive rate (TPR)
        true_positive = sum([1 for pred, score in preds if score > 0.5])  # assume 0.5 as threshold
        total_positive = len(skin_tone_gts[tone])
        tpr = true_positive / total_positive if total_positive > 0 else 0
        tprs.append(tpr)

        # False positive rate (FPR)
        false_positive = len(preds) - true_positive
        fpr = false_positive / len(preds) if len(preds) > 0 else 0
        fprs.append(fpr)

        # Accuracy
        acc = skin_tone_maps[tone]  # assuming mAP as a proxy for accuracy
        accuracies.append(acc)

    # Calculate |1-DI|
    di_values = [tpr for tpr in tprs if tpr > 0]
    di = max(di_values) / min(di_values) if min(di_values) > 0 else float('inf')
    one_minus_di = abs(1 - di)

    # Calculate ∆EOP (difference in Equality of Opportunity, i.e., TPR difference)
    eop_diff = max(tprs) - min(tprs)

    # Calculate ∆EODD (difference in Equalized Odds, i.e., difference in both TPR and FPR)
    eodd_diff = max([(tprs[i] - fprs[i]) for i in range(len(tprs))]) - min([(tprs[i] - fprs[i]) for i in range(len(tprs))])

    return {
        'Worst-case Difference': max_diff,
        'Best-case Difference': min_diff,
        'Wasserstein-2 Metric': worst_case_wasserstein,
        'Average Accuracy': np.mean(accuracies),
        '|1-DI| (Disparate Impact)': one_minus_di,
        '∆EOP (Equality of Opportunity)': eop_diff,
        '∆EODD (Equalized Odds)': eodd_diff
    }

def print_extended_fairness_metrics(fairness_metrics):
    print("\nExtended Fairness Metrics:")
    print(f"Average Accuracy: {fairness_metrics['Average Accuracy']:.4f}")
    print(f"|1-DI| (Disparate Impact): {fairness_metrics['|1-DI| (Disparate Impact)']:.4f}")
    print(f"∆EOP (Equality of Opportunity): {fairness_metrics['∆EOP (Equality of Opportunity)']:.4f}")
    print(f"∆EODD (Equalized Odds): {fairness_metrics['∆EODD (Equalized Odds)']:.4f}")
    print(f"Worst-case Difference: {fairness_metrics['Worst-case Difference']:.4f}")
    print(f"Best-case Difference: {fairness_metrics['Best-case Difference']:.4f}")
    print(f"Wasserstein-2 Metric: {fairness_metrics['Wasserstein-2 Metric']:.4f}")

# Modify assess_fairness to include extended fairness metrics
def assess_fairness_with_extended_metrics(model, test_images_path, metadata_path):
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Initialize containers for predictions and ground truths
    skin_tone_preds = defaultdict(list)
    skin_tone_gts = defaultdict(list)

    # Get the list of test images
    test_images = [os.path.join(test_images_path, f) for f in os.listdir(test_images_path) if f.endswith('.jpg')]

    # Perform inference on the test dataset
    for image_path in tqdm(test_images):
        results = model(image_path)
        
        result = results[0]
        image_id = (os.path.splitext(os.path.basename(result.path))[0])[3:]

        print(image_id)
        if image_id not in metadata:
            print(f"{image_id} is not found!")
            continue

        ground_truths = metadata[image_id]['persons']
        predictions = result.boxes.xyxy.cpu().numpy()  # move tensor to CPU and convert to numpy
        confidences = result.boxes.conf.cpu().numpy()
        pred_cls = result.boxes.cls.cpu().numpy()

        for gt in ground_truths:
            gt_box = convert_to_xyxy(gt['bounding_box'])
            gt_class = GT_CLASS
            skin_tones = get_skin_tones(gt)
            for skin_tone in skin_tones:
                skin_tone_gts[skin_tone].append(gt_box)

        for id, pred in enumerate(predictions):
            x_min, y_min, x_max, y_max = pred.tolist()
            conf, pcls = confidences[id], int(pred_cls[id])
            pred_box = [x_min, y_min, x_max, y_max]
            pred_score = conf
            for gt in ground_truths:
                if pcls == gt_class:
                    skin_tones = get_skin_tones(gt)
                    for skin_tone in skin_tones:
                        skin_tone_preds[skin_tone].append((pred_box, pred_score))
                    break

    # Calculate mAP and collect score distributions for each skin tone
    skin_tone_maps = {}
    for tone in skin_tone_gts:
        y_true, y_scores = calculate_map(skin_tone_preds[tone], skin_tone_gts[tone])
        if y_true:  # Only calculate mAP if there are valid entries
            skin_tone_maps[tone] = average_precision_score(y_true, y_scores)
        else:
            skin_tone_maps[tone] = 0.0

    # Print the mAP for each skin tone
    for tone, mAP in skin_tone_maps.items():
        print(f'Skin tone {tone}: Average mAP: {mAP:.4f}')

    # Calculate mAP and collect score distributions for each skin tone
    skin_tone_scores = {tone: calculate_map(skin_tone_preds[tone], skin_tone_gts[tone])[1] for tone in skin_tone_gts}

    # Calculate extended fairness metrics
    fairness_metrics_extended = calculate_fairness_metrics_extended(skin_tone_scores, skin_tone_maps, skin_tone_gts, skin_tone_preds)
    print_extended_fairness_metrics(fairness_metrics_extended)

    return skin_tone_scores

# Example usage
if __name__ == '__main__':
    model_path = "/home/AVs_Privacy/privfair-fl/model_weights_baseline_training_groupwise_perturbation_squared/train/weights/last.pt"
    model_path = "/home/AVs_Privacy/proposed_approach/model_weights/AutoENNv2_FACET/AutoENNv2_FACET_comm_99.pt"
    model = YOLO(model_path)

    # Set paths for test images and metadata
    test_images_path = '/home/Virginia_Research/FACET/images/test'
    metadata_path = '/home/Virginia_Research/FACET/metadata.json'

    # Assess fairness with extended metrics
    skin_tone_scores = assess_fairness_with_extended_metrics(model, test_images_path, metadata_path)
