import cv2
import numpy as np

CLASSES = [
    "011.exc_Adult",
    "012.ybtn_Adult",
    "013.gjxtn_Adult",
    "014.bdgclc_Adult",
    "028.sxdde_Adult",
    "032.mtn_Adult",
    "045.xlyc_Adult",
    "053.bce_larva",
    "054.sze_Adult",
    "058.bylc_Adult",
    "060.xmye_Adult",
    "065.llyj_Adult",
    "067.tzm_Adult",
    "069.stn_Adult",
    "071.hwcc_Adult",
    "009.ewze_Adult",
    "010.ewze_larva",
    "075.ydfd_Adult",
    "080.bxhjg_Adult",
    "087.belc_Adult",
    "089.djyc_Adult",
    "096.hyde_Adult",
    "099.hjtn_Adult",
    "103.lce_Adult",
    "104.lce_larva",
    "105.mgbe_Adult",
    "109.ccc_Adult",
    "110.clj_Adult",
    "111.cfd_Adult",
    "116.lg_Adult",
    "123.ctc_Adult",
    "132.mpc_Adult",
    "140.hzc_Adult"
    ]

def filter_Detections(results, thresh = 0.5):
    # if model is trained on 1 class only
    if len(results[0]) == 5:
        # filter out the detections with confidence > thresh
        considerable_detections = [detection for detection in results if detection[4] > thresh]
        considerable_detections = np.array(considerable_detections)
        return considerable_detections

    # if model is trained on multiple classes
    else:
        A = []
        for detection in results:

            class_id = detection[4:].argmax()
            confidence_score = detection[4:].max()

            new_detection = np.append(detection[:4],[class_id,confidence_score])

            A.append(new_detection)

        A = np.array(A)

        # filter out the detections with confidence > thresh
        considerable_detections = [detection for detection in A if detection[-1] > thresh]
        considerable_detections = np.array(considerable_detections)

        return considerable_detections

def NMS(boxes, conf_scores, iou_thresh = 0.55):

    #  boxes [[x1,y1, x2,y2], [x1,y1, x2,y2], ...]

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    areas = (x2-x1)*(y2-y1)

    order = conf_scores.argsort()

    keep = []
    keep_confidences = []

    while len(order) > 0:
        idx = order[-1]
        A = boxes[idx]
        conf = conf_scores[idx]

        order = order[:-1]

        xx1 = np.take(x1, indices= order)
        yy1 = np.take(y1, indices= order)
        xx2 = np.take(x2, indices= order)
        yy2 = np.take(y2, indices= order)

        keep.append(A)
        keep_confidences.append(conf)

        # iou = inter/union

        xx1 = np.maximum(x1[idx], xx1)
        yy1 = np.maximum(y1[idx], yy1)
        xx2 = np.minimum(x2[idx], xx2)
        yy2 = np.minimum(y2[idx], yy2)

        w = np.maximum(xx2-xx1, 0)
        h = np.maximum(yy2-yy1, 0)

        intersection = w*h

        # union = areaA + other_areas - intesection
        other_areas = np.take(areas, indices= order)
        union = areas[idx] + other_areas - intersection

        iou = intersection/union

        boleans = iou < iou_thresh

        order = order[boleans]

        # order = [2,0,1]  boleans = [True, False, True]
        # order = [2,1]

    return keep, keep_confidences

# function to rescale bounding boxes 
def rescale_back(results,img_w,img_h):
    cx, cy, w, h, class_id, confidence = results[:,0], results[:,1], results[:,2], results[:,3], results[:,4], results[:,-1]
    cx = cx/640.0 * img_w
    cy = cy/640.0 * img_h
    w = w/640.0 * img_w
    h = h/640.0 * img_h
    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2

    boxes = np.column_stack((x1, y1, x2, y2, class_id))
    keep, keep_confidences = NMS(boxes,confidence)
    # print(np.array(keep).shape)
    return keep, keep_confidences