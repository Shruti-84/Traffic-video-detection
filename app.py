import streamlit as st
import cv2
import cvzone
import tempfile
import os
import math
from ultralytics import YOLO

# =========================
# Streamlit Config
# =========================
st.set_page_config(page_title="Helmet Detection – All Riders", page_icon="🛡️")
st.title("🛡️ Helmet Detection for ALL Bike Riders")

# =========================
# Load Models
# =========================
@st.cache_resource
def load_models():
    return YOLO("yolov8n.pt"), YOLO("best.pt")

coco_model, helmet_model = load_models()

PERSON_ID = 0
BIKE_ID = 3
helmet_classes = ["With Helmet", "Without Helmet"]

# =========================
# Tracking Containers
# =========================
tracked_riders = {}
rider_results = {}
next_rider_id = 0
DIST_THRESHOLD = 70

# =========================
# Helper Functions
# =========================
def centroid(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def same_rider(c1, c2):
    return math.hypot(c1[0] - c2[0], c1[1] - c2[1]) < DIST_THRESHOLD

def is_rider(person, bike):
    px1, py1, px2, py2 = person
    bx1, by1, bx2, by2 = bike
    cx, cy = centroid(person)

    inside = bx1 < cx < bx2 and by1 < cy < by2
    above = py1 < by1 + (by2 - by1) * 0.6
    return inside and above

# =========================
# Upload Video
# =========================
uploaded_video = st.file_uploader("Upload Traffic Video", type=["mp4", "avi", "mov"])

if uploaded_video:
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_input.write(uploaded_video.read())
    temp_input.close()

    cap = cv2.VideoCapture(temp_input.name)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    output_path = "annotated_output.avi"
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height)
    )

    # =========================
    # Process Video
    # =========================
    with st.spinner("Detecting riders & helmets..."):
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            coco_results = coco_model(frame, conf=0.4)
            persons, bikes = [], []

            for box in coco_results[0].boxes:
                cls = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if cls == PERSON_ID:
                    persons.append((x1, y1, x2, y2))
                elif cls == BIKE_ID:
                    bikes.append((x1, y1, x2, y2))

            for person in persons:
                for bike in bikes:
                    if not is_rider(person, bike):
                        continue

                    c = centroid(person)

                    # Rider matching
                    rider_id = None
                    for rid, data in tracked_riders.items():
                        if same_rider(c, data["centroid"]):
                            rider_id = rid
                            break

                    if rider_id is None:
                        rider_id = next_rider_id
                        tracked_riders[rider_id] = {"centroid": c}
                        rider_results[rider_id] = {
                            "Helmet Status": "Not Detected",
                            "Confidence (%)": 0.0
                        }
                        next_rider_id += 1

                    tracked_riders[rider_id]["centroid"] = c
                    px1, py1, px2, py2 = person

                    # =========================
                    # HELMET DETECTION (RETRY UNTIL FOUND)
                    # =========================
                    if rider_results[rider_id]["Helmet Status"] == "Not Detected":
                        head = frame[
                            py1:int(py1 + (py2 - py1) * 0.5),
                            px1:px2
                        ]

                        if head.size > 0:
                            h_results = helmet_model(head, conf=0.4)

                            if len(h_results[0].boxes) > 0:
                                hbox = h_results[0].boxes[0]
                                cls = int(hbox.cls[0])
                                conf = round(float(hbox.conf[0]) * 100, 1)

                                rider_results[rider_id] = {
                                    "Helmet Status": helmet_classes[cls],
                                    "Confidence (%)": conf
                                }

                    # =========================
                    # Draw Annotations
                    # =========================
                    cvzone.cornerRect(
                        frame,
                        (px1, py1, px2 - px1, py2 - py1),
                        colorC=(255, 255, 0)
                    )
                    cv2.putText(
                        frame,
                        f"Rider {rider_id}",
                        (px1, py1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2
                    )

                    status = rider_results[rider_id]["Helmet Status"]
                    conf = rider_results[rider_id]["Confidence (%)"]
                    color = (0, 255, 0) if status == "With Helmet" else (0, 0, 255)

                    cvzone.putTextRect(
                        frame,
                        f"{status} {conf}%",
                        (px1, max(35, py1)),
                        scale=1,
                        thickness=1,
                        colorR=color
                    )

            out.write(frame)

    cap.release()
    out.release()
    os.remove(temp_input.name)

    # =========================
    # DISPLAY RESULTS
    # =========================
    st.success("Detection Completed")

    st.subheader("Annotated Output Video")
    with open(output_path, "rb") as f:
        st.video(f.read())

    st.subheader("Helmet Status for ALL Riders")
    st.table([
        {"Rider ID": rid, **info}
        for rid, info in rider_results.items()
    ])
