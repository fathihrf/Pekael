import cv2
import json
import os
import glob
import numpy as np
from paddleocr import PaddleOCR
import re
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Constants
ANNOTATIONS_FILE = "annotations/annotations.json"
TRAIN_DIR = "images/train"
TEST_DIR = "images/test"
TEMPLATE_DIR = "templates"

def load_annotations():
    with open(ANNOTATIONS_FILE, 'r') as f:
        return json.load(f)

def iou(boxA, boxB):
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def extract_templates(annotations):
    os.makedirs(TEMPLATE_DIR, exist_ok=True)
    
    # Find a training image in annotations
    # We assume the user has annotated at least one image in images/train
    # In our logic, we look for an image that exists in images/train
    
    train_images = glob.glob(os.path.join(TRAIN_DIR, "*"))
    ref_image_path = None
    
    for path in train_images:
        filename = os.path.basename(path)
        if filename in annotations:
            ref_image_path = path
            break
            
    if not ref_image_path:
        print("No annotated reference image found in images/train/")
        return False

    print(f"Using Master Image: {ref_image_path}")
    img = cv2.imread(ref_image_path)
    filename = os.path.basename(ref_image_path)
    annots = annotations[filename]
    
    templates = {}
    
    for label, rect in annots.items():
        if rect is None: continue
        x, y, w, h = rect
        template = img[y:y+h, x:x+w]
        template_path = os.path.join(TEMPLATE_DIR, f"{label}_template.jpg")
        cv2.imwrite(template_path, template)
        templates[label] = template
        print(f"Saved template for {label} to {template_path}")
        
    return templates

def preprocess_for_ocr(img_roi):
    # PaddleOCR works best on natural images, but for 7-segment, we often need to "connect" the dots.
    # 1. Upscale
    candidate_list = []

    # Candidate 1: Natural @ 2x Scale (Best for Humidity/Solid Text)
    scale_low = 2
    h, w = img_roi.shape[:2]
    upscaled_low = cv2.resize(img_roi, (w*scale_low, h*scale_low), interpolation=cv2.INTER_CUBIC)
    
    cand1 = cv2.copyMakeBorder(upscaled_low, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[127, 127, 127])
    # Ensure BGR
    if len(cand1.shape) == 2:
        cand1 = cv2.cvtColor(cand1, cv2.COLOR_GRAY2BGR)
    
    candidate_list.append(cand1)

    # Candidate 2: Binary + Dilated @ 4x Scale (Best for 7-Segment Temperature)
    scale_high = 4
    upscaled_high = cv2.resize(img_roi, (w*scale_high, h*scale_high), interpolation=cv2.INTER_CUBIC)
    
    if len(upscaled_high.shape) == 3:
         gray = cv2.cvtColor(upscaled_high, cv2.COLOR_BGR2GRAY)
    else:
         gray = upscaled_high
         
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Normalize to White Text on Black
    h, w = thresh.shape
    corners = [thresh[0,0], thresh[0,w-1], thresh[h-1,0], thresh[h-1,w-1]]
    if np.mean(corners) > 127: 
        thresh = cv2.bitwise_not(thresh)
        
    # Dilate White Text
    kernel = np.ones((3,3), np.uint8)
    processed = cv2.dilate(thresh, kernel, iterations=1)
    
    # Invert back to Black on White
    processed = cv2.bitwise_not(processed)
    
    # Pad
    processed = cv2.copyMakeBorder(processed, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255]) 
    cand2 = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    
    candidate_list.append(cand2)
    
    return candidate_list

def analyze_image(img, templates, reader, ground_truth=None):
    """
    Analyzes a single image using the provided templates and OCR reader.
    Returns:
        processed_img: The image with annotations drawn.
        results: A dictionary of detected values {label: {'value': str, 'score': float, 'box': list}}.
    """
    results = {}
    
    for label, template in templates.items():
        # Template Match
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        # Predict Box
        h, w = template.shape[:2]
        top_left = max_loc
        pred_box = [top_left[0], top_left[1], w, h]
        
        score = max_val
        
        # Draw Prediction (Red)
        cv2.rectangle(img, (pred_box[0], pred_box[1]), (pred_box[0]+w, pred_box[1]+h), (0, 0, 255), 2)
        
        # --- OCR Section ---
        x1, y1, w1, h1 = pred_box
        x1 = max(0, x1)
        y1 = max(0, y1)
        roi_img_orig = img[y1:y1+h1, x1:x1+w1]
        
        # Expanded coords
        expand_w = int(w1 * 0.2)
        expand_h = int(h1 * 0.1)
        x2 = max(0, x1 - expand_w)
        y2 = max(0, y1 - expand_h)
        w2 = min(img.shape[1] - x2, w1 + 2 * expand_w)
        h2 = min(img.shape[0] - y2, h1 + 2 * expand_h)
        roi_img_exp = img[y2:y2+h2, x2:x2+w2]
        
        candidates = []
        if roi_img_orig.size > 0:
             candidates.extend(preprocess_for_ocr(roi_img_orig))
        if roi_img_exp.size > 0:
             candidates.extend(preprocess_for_ocr(roi_img_exp))
        
        final_text = ""
        if candidates:
            # Helper to parse result
            def parse_ocr_result(res):
                 text_out = ""
                 if not res: return text_out
                 
                 # Check for new Dictionary format (PaddleOCR 3.x / PaddleX)
                 if isinstance(res, list) and len(res) > 0 and isinstance(res[0], dict):
                     for item in res:
                         texts = item.get('rec_texts', [])
                         for text in texts:
                             cleaned = re.sub(r'[^0-9.]', '', text)
                             text_out += cleaned
                     return text_out
                 
                 # Fallback to list of lines format
                 for line in res:
                     if isinstance(line, list):
                        if len(line) > 0 and isinstance(line[0], list) and len(line[0]) == 4 and isinstance(line[0][0], list):
                            text = line[1][0]
                            cleaned = re.sub(r'[^0-9.]', '', text)
                            text_out += cleaned
                        else:
                            for subline in line:
                                if isinstance(subline, list) and len(subline) >= 2:
                                     text = subline[1][0]
                                     cleaned = re.sub(r'[^0-9.]', '', text)
                                     text_out += cleaned
                 return text_out

            best_text = ""
            for i, img_cand in enumerate(candidates):
                result = reader.ocr(img_cand)
                detected_text = parse_ocr_result(result)
                if len(detected_text) > len(best_text):
                    best_text = detected_text
            
            final_text = best_text
            
            # If still empty, try inverted of the Natural candidate (index 0)
            if not final_text and len(candidates) > 0:
                inverted_roi = cv2.bitwise_not(candidates[0])
                result_inv = reader.ocr(inverted_roi)
                final_text = parse_ocr_result(result_inv)
            
        results[label] = {
            'value': final_text,
            'score': score,
            'box': pred_box
        }

        # Evaluate if GT exists
        if ground_truth and label in ground_truth:
            gt_box = ground_truth[label]
            iou_val = iou(pred_box, gt_box)
            # Draw GT (Green)
            cv2.rectangle(img, (gt_box[0], gt_box[1]), (gt_box[0]+gt_box[2], gt_box[1]+gt_box[3]), (0, 255, 0), 2)
            results[label]['iou'] = iou_val

    return img, results

def process_test_images(templates, annotations, reader):
    test_images = glob.glob(os.path.join(TEST_DIR, "*"))
    
    print("\n--- Testing ---")
    for img_path in test_images:
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        
        # Ground Truth
        gt = annotations.get(filename)
        
        print(f"\nImage: {filename}")
        
        processed_img, results = analyze_image(img, templates, reader, ground_truth=gt)
        
        for label, data in results.items():
            print(f"  [{label}] Match Score: {data['score']:.4f}")
            print(f"  [{label}] Detected Value: {data['value']}")
            if 'iou' in data:
                print(f"  [{label}] IoU: {data['iou']:.4f}")
            else:
                print(f"  [{label}] No GT found.")

        # Save result image
        output_path = f"output_{filename}"
        cv2.imwrite(output_path, processed_img) 
        print(f"  Saved result to {output_path}")

def main():
    if not os.path.exists(ANNOTATIONS_FILE):
        print("No annotations found. Please run annotate.py first.")
        return

    annotations = load_annotations()
    
    # Phase A: Train
    templates = extract_templates(annotations)
    if not templates:
        return
        
    # Init OCR Reader once
    print("Initializing OCR Engine...")
    # Try passing det=False here to disable detection model
    reader = PaddleOCR(lang='en', use_angle_cls=False)

    # Phase B: Test
    process_test_images(templates, annotations, reader)

if __name__ == "__main__":
    main()
