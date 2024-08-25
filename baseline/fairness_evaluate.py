import json
import os
from collections import defaultdict
from ultralytics import YOLO
from sklearn.metrics import average_precision_score
from tqdm import tqdm

GT_CLASS = 0

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
    return average_precision_score(y_true, y_scores)

def assess_fairness(model_path, test_images_path, metadata_path):
    # Load the trained model
    model = YOLO(model_path)

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



    # Calculate mAP for each skin tone
    skin_tone_maps = {tone: calculate_map(skin_tone_preds[tone], skin_tone_gts[tone]) for tone in skin_tone_gts}

    # Print results
    for tone, mAP in skin_tone_maps.items():
        print(f'Skin Tone {tone}: mAP = {mAP:.2f}')

    return skin_tone_maps

# Example usage
model_path = '/home/AVs_Privacy/baseline/facet_baseline_training/train/weights/last.pt'
test_images_path = '/home/Virginia_Research/FACET/images/test'
metadata_path = '/home/Virginia_Research/FACET/metadata.json'

skin_tone_maps = assess_fairness(model_path, test_images_path, metadata_path)
