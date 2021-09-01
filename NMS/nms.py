'''
Created on Feb 10, 2020

@author: bc
'''
import cv2
import numpy as np

def nms(bounding_boxes, confidence_score, threshold):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return picked_boxes, picked_score

def nms_index(bounding_boxes, confidence_score, threshold):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    indexes = []
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])
        indexes.append(index)

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return indexes

def get_iou(pred_box, gt_box):
    """
    pred_box : the coordinate for predict bounding box
    gt_box :   the coordinate for ground truth bounding box
    return :   the iou score
    the  left-down coordinate of  pred_box:(pred_box[0], pred_box[1])
    the  right-up coordinate of  pred_box:(pred_box[2], pred_box[3])
    """
    # 1.get the coordinate of inters
    ixmin = max(pred_box[0], gt_box[0])
    ixmax = min(pred_box[2], gt_box[2])
    iymin = max(pred_box[1], gt_box[1])
    iymax = min(pred_box[3], gt_box[3])

    iw = np.maximum(ixmax-ixmin+1., 0.)
    ih = np.maximum(iymax-iymin+1., 0.)

    # 2. calculate the area of inters
    inters = iw*ih

    # 3. calculate the area of union
    uni = ((pred_box[2]-pred_box[0]+1.) * (pred_box[3]-pred_box[1]+1.) +
           (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
           inters)

    # 4. calculate the overlaps between pred_box and gt_box
    iou = inters / uni
    return iou

def comput_precision(boxes, labels, true_boxes, true_labels, iou_thr):
    # TP/(TP+FP)
    TP = 0
    FP = 0
    for i in range(len(labels)):
        for j in range(len(true_labels)):
            box = boxes[i]
            true_box = true_boxes[j]
            if get_iou(box, true_box)>iou_thr:
                if labels[i] == true_labels[j]:
                    TP += 1
                else:
                    print(f"i,j={i}, {j}, FP")
                    FP += 1
    print(f"TP={TP}, FP={FP}")
    return TP/(TP+FP)

def main():
    # Image name
    image_name = 'nms.jpg'

    # Bounding boxes (xmin, ymin, xmax, ymax)
    bounding_boxes = [(187, 82, 337, 317), (150, 67, 305, 282), (246, 121, 368, 304), (350, 380, 448, 499)]
    confidence_score = [0.82, 0.75, 0.8, 0.68]

    # Read image
    image = cv2.imread(image_name)

    # Copy image as original
    org = image.copy()

    # Draw parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    # IoU threshold
    threshold = 0.5


    labels = [i for i in range(len(bounding_boxes))]
    # Draw bounding boxes and confidence score
    for (start_x, start_y, end_x, end_y), confidence, label in zip(bounding_boxes, confidence_score, labels):
        (w, h), baseline = cv2.getTextSize(str(confidence), font, font_scale, thickness)
        cv2.rectangle(org, (start_x, start_y - (2 * baseline + 5)), (start_x + w, start_y), (0, 255, 255), -1)
        cv2.rectangle(org, (start_x, start_y), (end_x, end_y), (0, 255, 255), 1)
        cv2.putText(org, f"{confidence}_{label}", (start_x, start_y), font, font_scale, (0, 0, 0), thickness)


    # Run non-max suppression algorithm
    #picked_boxes, picked_score = nms(bounding_boxes, confidence_score, threshold)


    select_ind = nms_index(bounding_boxes, confidence_score, threshold)

    picked_boxes = []
    picked_score = []
    picked_labels = []
    for ind in select_ind:
        picked_boxes.append(bounding_boxes[ind])
        picked_score.append(confidence_score[ind])
        picked_labels.append(labels[ind])



    # Draw bounding boxes and confidence score after non-maximum supression
    for (start_x, start_y, end_x, end_y), confidence, label in zip(picked_boxes, picked_score, picked_labels):
        (w, h), baseline = cv2.getTextSize(str(confidence), font, font_scale, thickness)
        cv2.rectangle(image, (start_x, start_y - (2 * baseline + 5)), (start_x + w, start_y), (0, 255, 255), -1)
        cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 255, 255), 1)
        cv2.putText(image, f"{confidence}_{label}", (start_x, start_y), font, font_scale, (0, 0, 0), thickness)

    
    

    cv2.imshow('Original', org)
    cv2.imshow('NMS', image)
    cv2.waitKey(0)
    
    
    

if __name__ == "__main__":
    main()

