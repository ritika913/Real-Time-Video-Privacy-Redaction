import os
import tempfile
import cv2
import gradio as gr
import numpy as np
from ultralytics import YOLO

_face_cascade = None
_plate_model = None
_coco_model = None
_phone_model = None
_idcard_model = None
_webcam_trackers_bundle = None
_webcam_trackers_key = None
WEBCAM_STREAM_INTERVAL_SEC = 0.06

_COCO_TV_LAPTOP_CLASSES = [62, 63]


def _get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)
    return _face_cascade


def _get_plate_model():
    global _plate_model
    if _plate_model is None:
        try:
            _plate_model = YOLO("hf://morsetechlab/yolov11-license-plate-detection")
        except Exception:
            _plate_model = YOLO("morsetechlab/yolov11-license-plate-detection")
    return _plate_model


def _get_coco_model():
    global _coco_model
    if _coco_model is None:
        _coco_model = YOLO("yolov8n.pt")
    return _coco_model


def _get_phone_model():
    global _phone_model
    if _phone_model is None:
        _phone_model = YOLO("https://huggingface.co/IndUSV/yolov8n-mobile-phone/resolve/main/yolov8n-mobile-phone.pt")
    return _phone_model


def _get_idcard_model():
    global _idcard_model
    if _idcard_model is None:
        _idcard_model = YOLO("https://huggingface.co/MiguelEscamilla/id-card-yolo/resolve/main/best.pt")
    return _idcard_model


def _iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    ar = (a[2] - a[0]) * (a[3] - a[1])
    br = (b[2] - b[0]) * (b[3] - b[1])
    union = ar + br - inter
    return inter / union if union > 1e-6 else 0.0


def _greedy_match(track_boxes, det_boxes, iou_threshold):
    n_t = len(track_boxes)
    n_d = len(det_boxes)
    if n_t == 0 or n_d == 0:
        return [], list(range(n_t)), list(range(n_d))
    mat = np.zeros((n_t, n_d), dtype=np.float64)
    for i in range(n_t):
        for j in range(n_d):
            mat[i, j] = _iou(track_boxes[i], det_boxes[j])
    used_t = set()
    used_d = set()
    pairs = []
    while True:
        best = -1.0
        bi = bj = -1
        for i in range(n_t):
            if i in used_t:
                continue
            for j in range(n_d):
                if j in used_d:
                    continue
                if mat[i, j] > best:
                    best = mat[i, j]
                    bi, bj = i, j
        if best < iou_threshold:
            break
        pairs.append((bi, bj))
        used_t.add(bi)
        used_d.add(bj)
    unmatched_t = [i for i in range(n_t) if i not in used_t]
    unmatched_d = [j for j in range(n_d) if j not in used_d]
    return pairs, unmatched_t, unmatched_d


def _smooth_box(prev, new, alpha):
    if prev is None:
        return tuple(new)
    a = float(alpha)
    return tuple(
        a * new[k] + (1.0 - a) * prev[k] for k in range(4)
    )


class FaceTracker:
    def __init__(
        self,
        iou_threshold=0.25,
        max_missed=7,
        smooth_alpha=0.55,
    ):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.smooth_alpha = smooth_alpha
        self._tracks = []
        self._next_id = 0

    def _new_track(self, box):
        tid = self._next_id
        self._next_id += 1
        return {
            "id": tid,
            "box": tuple(map(float, box)),
            "missed": 0,
        }

    def update(self, detections):
        dets = [tuple(map(float, b)) for b in detections]
        if not self._tracks:
            for d in dets:
                self._tracks.append(self._new_track(d))
            return [t["box"] for t in self._tracks]

        track_boxes = [t["box"] for t in self._tracks]
        pairs, unmatched_t, unmatched_d = _greedy_match(
            track_boxes, dets, self.iou_threshold
        )

        for ti, dj in pairs:
            t = self._tracks[ti]
            t["box"] = _smooth_box(t["box"], dets[dj], self.smooth_alpha)
            t["missed"] = 0

        for ti in unmatched_t:
            self._tracks[ti]["missed"] += 1

        for dj in unmatched_d:
            self._tracks.append(self._new_track(dets[dj]))

        self._tracks = [t for t in self._tracks if t["missed"] <= self.max_missed]
        return [t["box"] for t in self._tracks]

    def reset(self):
        self._tracks = []
        self._next_id = 0


def blur_boxes_bgr(img_bgr, boxes_xyxy, ksize=99):
    out = img_bgr.copy()
    h, w = out.shape[:2]

    k = int(ksize)
    if k % 2 == 0:
        k += 1
    k = max(3, k)

    for (x1, y1, x2, y2) in boxes_xyxy:
        x1 = max(0, min(w - 1, int(x1)))
        y1 = max(0, min(h - 1, int(y1)))
        x2 = max(0, min(w, int(x2)))
        y2 = max(0, min(h, int(y2)))
        if x2 <= x1 or y2 <= y1:
            continue

        roi = out[y1:y2, x1:x2]
        out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)

    return out


def detect_faces_opencv(img_bgr, scaleFactor=1.1, minNeighbors=5, max_dim=480):
    face_cascade = _get_face_cascade()
    h, w = img_bgr.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        proc_img = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    else:
        proc_img = img_bgr
        
    gray = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors
    )
    return [(int(x/scale), int(y/scale), int((x + w)/scale), int((y + h)/scale)) for (x, y, w, h) in faces]


def detect_plates_yolo(img_bgr, conf=0.25):
    model = _get_plate_model()
    results = model.predict(source=img_bgr, imgsz=320, conf=float(conf), verbose=False)
    boxes = []
    for r in results:
        if r.boxes is None:
            continue
        for b in r.boxes:
            x1, y1, x2, y2 = map(float, b.xyxy[0])
            boxes.append((x1, y1, x2, y2))
    return boxes


def detect_phones_yolo(img_bgr, conf=0.35):
    model = _get_phone_model()
    results = model.predict(source=img_bgr, imgsz=320, conf=float(conf), verbose=False)
    boxes = []
    for r in results:
        if r.boxes is None:
            continue
        for b in r.boxes:
            x1, y1, x2, y2 = map(float, b.xyxy[0])
            boxes.append((x1, y1, x2, y2))
    return boxes


def detect_tv_laptop_yolo(img_bgr, conf=0.35):
    model = _get_coco_model()
    results = model.predict(
        source=img_bgr,
        imgsz=320,
        classes=_COCO_TV_LAPTOP_CLASSES,
        conf=float(conf),
        verbose=False,
    )
    boxes = []
    for r in results:
        if r.boxes is None:
            continue
        for b in r.boxes:
            x1, y1, x2, y2 = map(float, b.xyxy[0])
            boxes.append((x1, y1, x2, y2))
    return boxes


def detect_idcard_yolo(img_bgr, conf=0.35):
    model = _get_idcard_model()
    results = model.predict(source=img_bgr, imgsz=320, conf=float(conf), verbose=False)
    boxes = []
    for r in results:
        if r.boxes is None:
            continue
        for b in r.boxes:
            x1, y1, x2, y2 = map(float, b.xyxy[0])
            boxes.append((x1, y1, x2, y2))
    return boxes


def _tracker_triplet(iou_threshold, max_missed, smooth_alpha):
    return FaceTracker(
        iou_threshold=float(iou_threshold),
        max_missed=int(max_missed),
        smooth_alpha=float(smooth_alpha),
    )


def _redaction_trackers_key(
    redact_faces,
    redact_plates,
    redact_phones,
    redact_screens,
    redact_cards_id,
    blur_ksize,
    scaleFactor,
    minNeighbors,
    plate_conf,
    yolo_conf,
    iou_threshold,
    max_missed,
    smooth_alpha,
):
    return (
        bool(redact_faces),
        bool(redact_plates),
        bool(redact_phones),
        bool(redact_screens),
        bool(redact_cards_id),
        int(blur_ksize),
        round(float(scaleFactor), 4),
        int(minNeighbors),
        round(float(plate_conf), 4),
        round(float(yolo_conf), 4),
        round(float(iou_threshold), 4),
        int(max_missed),
        round(float(smooth_alpha), 4),
    )


def _get_webcam_trackers(key):
    global _webcam_trackers_bundle, _webcam_trackers_key
    if _webcam_trackers_bundle is None or _webcam_trackers_key != key:
        iou_threshold = key[10]
        max_missed = key[11]
        smooth_alpha = key[12]
        _webcam_trackers_bundle = {
            "face": _tracker_triplet(iou_threshold, max_missed, smooth_alpha),
            "plate": _tracker_triplet(iou_threshold, max_missed, smooth_alpha),
            "phone": _tracker_triplet(iou_threshold, max_missed, smooth_alpha),
            "screen": _tracker_triplet(iou_threshold, max_missed, smooth_alpha),
            "card": _tracker_triplet(iou_threshold, max_missed, smooth_alpha),
        }
        _webcam_trackers_key = key
    return _webcam_trackers_bundle


def process_video_faces(
    video_path,
    redact_faces=True,
    redact_plates=False,
    redact_phones=False,
    redact_screens=False,
    redact_cards_id=False,
    blur_ksize=99,
    scaleFactor=1.1,
    minNeighbors=5,
    plate_conf=0.25,
    yolo_conf=0.35,
    iou_threshold=0.25,
    max_missed=7,
    smooth_alpha=0.55,
):
    if not video_path:
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-3:
        fps = 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w <= 0 or h <= 0:
        cap.release()
        return None

    fd, out_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        try:
            os.remove(out_path)
        except OSError:
            pass
        return None

    tracker = FaceTracker(
        iou_threshold=iou_threshold,
        max_missed=max_missed,
        smooth_alpha=smooth_alpha,
    )
    plate_tracker = FaceTracker(
        iou_threshold=iou_threshold,
        max_missed=max_missed,
        smooth_alpha=smooth_alpha,
    )
    phone_tracker = FaceTracker(
        iou_threshold=iou_threshold,
        max_missed=max_missed,
        smooth_alpha=smooth_alpha,
    )
    screen_tracker = FaceTracker(
        iou_threshold=iou_threshold,
        max_missed=max_missed,
        smooth_alpha=smooth_alpha,
    )
    card_tracker = FaceTracker(
        iou_threshold=iou_threshold,
        max_missed=max_missed,
        smooth_alpha=smooth_alpha,
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        boxes = []
        if redact_faces:
            dets = detect_faces_opencv(
                frame, scaleFactor=scaleFactor, minNeighbors=minNeighbors
            )
            boxes.extend(tracker.update(dets))
        if redact_plates:
            p = detect_plates_yolo(frame, conf=plate_conf)
            boxes.extend(plate_tracker.update(p))
        if redact_phones:
            p = detect_phones_yolo(frame, conf=yolo_conf)
            boxes.extend(phone_tracker.update(p))
        if redact_screens:
            p = detect_tv_laptop_yolo(frame, conf=yolo_conf)
            boxes.extend(screen_tracker.update(p))
        if redact_cards_id:
            p = detect_idcard_yolo(frame, conf=yolo_conf)
            boxes.extend(card_tracker.update(p))
        out_frame = blur_boxes_bgr(frame, boxes, ksize=blur_ksize)
        writer.write(out_frame)

    cap.release()
    writer.release()
    return out_path


def ui_process_video(
    video_path,
    redact_faces,
    redact_plates,
    redact_phones,
    redact_screens,
    redact_cards_id,
    blur_ksize,
    scaleFactor,
    minNeighbors,
    plate_conf,
    yolo_conf,
    iou_threshold,
    max_missed,
    smooth_alpha,
):
    if video_path is None:
        return None, "No video provided."
    out = process_video_faces(
        video_path,
        redact_faces=bool(redact_faces),
        redact_plates=bool(redact_plates),
        redact_phones=bool(redact_phones),
        redact_screens=bool(redact_screens),
        redact_cards_id=bool(redact_cards_id),
        blur_ksize=int(blur_ksize),
        scaleFactor=float(scaleFactor),
        minNeighbors=int(minNeighbors),
        plate_conf=float(plate_conf),
        yolo_conf=float(yolo_conf),
        iou_threshold=float(iou_threshold),
        max_missed=int(max_missed),
        smooth_alpha=float(smooth_alpha),
    )
    if out is None:
        return None, "Processing failed."
    return out, "Done."


def ui_webcam_stream(
    img_rgb,
    redact_faces,
    redact_plates,
    redact_phones,
    redact_screens,
    redact_cards_id,
    blur_ksize,
    scaleFactor,
    minNeighbors,
    plate_conf,
    yolo_conf,
    iou_threshold,
    max_missed,
    smooth_alpha,
):
    if img_rgb is None:
        return None
    key = _redaction_trackers_key(
        redact_faces,
        redact_plates,
        redact_phones,
        redact_screens,
        redact_cards_id,
        blur_ksize,
        scaleFactor,
        minNeighbors,
        plate_conf,
        yolo_conf,
        iou_threshold,
        max_missed,
        smooth_alpha,
    )
    tr = _get_webcam_trackers(key)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    boxes = []
    if redact_faces:
        dets = detect_faces_opencv(
            img_bgr,
            scaleFactor=float(scaleFactor),
            minNeighbors=int(minNeighbors),
        )
        boxes.extend(tr["face"].update(dets))
    if redact_plates:
        p = detect_plates_yolo(img_bgr, conf=float(plate_conf))
        boxes.extend(tr["plate"].update(p))
    if redact_phones:
        p = detect_phones_yolo(img_bgr, conf=float(yolo_conf))
        boxes.extend(tr["phone"].update(p))
    if redact_screens:
        p = detect_tv_laptop_yolo(img_bgr, conf=float(yolo_conf))
        boxes.extend(tr["screen"].update(p))
    if redact_cards_id:
        p = detect_idcard_yolo(img_bgr, conf=float(yolo_conf))
        boxes.extend(tr["card"].update(p))
    out_bgr = blur_boxes_bgr(img_bgr, boxes, ksize=int(blur_ksize))
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


def ui_reset_webcam_tracker():
    global _webcam_trackers_bundle, _webcam_trackers_key
    _webcam_trackers_bundle = None
    _webcam_trackers_key = None


def build_app():
    with gr.Blocks(title="Face blur (video)") as demo:
        gr.Markdown("## Video redaction (tracking + toggles)")
        gr.Markdown(
            "Models: **plates** [morsetechlab/yolov11-license-plate-detection](https://huggingface.co/morsetechlab/yolov11-license-plate-detection), "
            "**phones** [IndUSV/yolov8n-mobile-phone](https://huggingface.co/IndUSV/yolov8n-mobile-phone), "
            "**TV & laptop** COCO classes via `yolov8n.pt` (tv, laptop), "
            "**cards & ID regions** [MiguelEscamilla/id-card-yolo](https://huggingface.co/MiguelEscamilla/id-card-yolo) "
            "(document/ID-oriented; not a dedicated credit-card-only model)."
        )

        with gr.Tabs():
            with gr.Tab("Video file"):
                vin = gr.Video(label="Input video")
                vout = gr.Video(label="Redacted video")
                with gr.Row():
                    v_faces = gr.Checkbox(value=True, label="Faces (Haar)")
                    v_plates = gr.Checkbox(value=False, label="Number plates")
                    v_phones = gr.Checkbox(value=False, label="Mobile phones")
                with gr.Row():
                    v_screens = gr.Checkbox(
                        value=False, label="TV & laptop (COCO)"
                    )
                    v_cards = gr.Checkbox(
                        value=False, label="Cards & ID (YOLO regions)"
                    )
                with gr.Row():
                    v_blur = gr.Slider(15, 151, value=99, step=2, label="Blur strength")
                    v_scale = gr.Slider(
                        1.01, 1.5, value=1.1, step=0.01, label="Haar scaleFactor"
                    )
                    v_min_n = gr.Slider(2, 12, value=5, step=1, label="Haar minNeighbors")
                    v_plate_conf = gr.Slider(
                        0.05, 0.9, value=0.25, step=0.05, label="Plate confidence"
                    )
                    v_yolo_conf = gr.Slider(
                        0.05,
                        0.9,
                        value=0.35,
                        step=0.05,
                        label="Other YOLO confidence",
                    )
                with gr.Row():
                    v_iou = gr.Slider(
                        0.05, 0.6, value=0.25, step=0.01, label="Track IoU match"
                    )
                    v_miss = gr.Slider(1, 20, value=7, step=1, label="Max missed frames")
                    v_smooth = gr.Slider(
                        0.2, 0.95, value=0.55, step=0.05, label="Box smoothing"
                    )
                vstatus = gr.Markdown("")
                vbtn = gr.Button("Process video")
                vbtn.click(
                    ui_process_video,
                    inputs=[
                        vin,
                        v_faces,
                        v_plates,
                        v_phones,
                        v_screens,
                        v_cards,
                        v_blur,
                        v_scale,
                        v_min_n,
                        v_plate_conf,
                        v_yolo_conf,
                        v_iou,
                        v_miss,
                        v_smooth,
                    ],
                    outputs=[vout, vstatus],
                )

            with gr.Tab("Live webcam"):
                gr.Markdown(
                    "Allow camera access, then start **Stream** on the webcam panel. "
                    "Output updates in near real time; latency depends on CPU and Haar settings."
                )
                with gr.Row():
                    cam_in = gr.Image(
                        sources=["webcam"],
                        streaming=True,
                        type="numpy",
                        label="Webcam",
                    )
                    cam_out = gr.Image(type="numpy", label="Blurred (live)")
                with gr.Row():
                    c_faces = gr.Checkbox(value=True, label="Faces (Haar)")
                    c_plates = gr.Checkbox(value=False, label="Number plates")
                    c_phones = gr.Checkbox(value=False, label="Mobile phones")
                with gr.Row():
                    c_screens = gr.Checkbox(value=False, label="TV & laptop (COCO)")
                    c_cards = gr.Checkbox(value=False, label="Cards & ID (YOLO regions)")
                with gr.Row():
                    c_blur = gr.Slider(15, 151, value=99, step=2, label="Blur strength")
                    c_scale = gr.Slider(
                        1.01, 1.5, value=1.1, step=0.01, label="Haar scaleFactor"
                    )
                    c_min_n = gr.Slider(2, 12, value=5, step=1, label="Haar minNeighbors")
                    c_plate_conf = gr.Slider(
                        0.05, 0.9, value=0.25, step=0.05, label="Plate confidence"
                    )
                    c_yolo_conf = gr.Slider(
                        0.05,
                        0.9,
                        value=0.35,
                        step=0.05,
                        label="Other YOLO confidence",
                    )
                with gr.Row():
                    c_iou = gr.Slider(
                        0.05, 0.6, value=0.25, step=0.01, label="Track IoU match"
                    )
                    c_miss = gr.Slider(1, 20, value=7, step=1, label="Max missed frames")
                    c_smooth = gr.Slider(
                        0.2, 0.95, value=0.55, step=0.05, label="Box smoothing"
                    )
                rst = gr.Button("Reset tracking")
                rst.click(fn=lambda: ui_reset_webcam_tracker(), inputs=[], outputs=[])

                cam_in.stream(
                    ui_webcam_stream,
                    inputs=[
                        cam_in,
                        c_faces,
                        c_plates,
                        c_phones,
                        c_screens,
                        c_cards,
                        c_blur,
                        c_scale,
                        c_min_n,
                        c_plate_conf,
                        c_yolo_conf,
                        c_iou,
                        c_miss,
                        c_smooth,
                    ],
                    outputs=[cam_out],
                    stream_every=WEBCAM_STREAM_INTERVAL_SEC,
                )

        return demo


if __name__ == "__main__":
    app = build_app()
    app.queue()
    app.launch()