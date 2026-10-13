
import cv2
from ultralytics import YOLO

# Load trained YOLO model
model = YOLO("best.pt")

# Open video file
cap = cv2.VideoCapture("input.mp4")

count = 0
counted_ids = set()
previous_positions = {}

# Horizontal counting line
LINE_Y = 400

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    # Detect and track objects
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.35,
        verbose=False
    )

    result = results[0]

    # Draw counting line
    cv2.line(
        frame,
        (0, LINE_Y),
        (frame.shape[1], LINE_Y),
        (0, 255, 255),
        3
    )

    if result.boxes is not None and result.boxes.id is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy().astype(int)
        classes = result.boxes.cls.cpu().numpy().astype(int)

        for box, track_id, class_id in zip(boxes, ids, classes):
            x1, y1, x2, y2 = map(int, box)

            # Object center
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Detect crossing from above to below
            if track_id in previous_positions:
                old_y = previous_positions[track_id]

                if (
                    old_y < LINE_Y <= center_y
                    and track_id not in counted_ids
                ):
                    count += 1
                    counted_ids.add(track_id)

            previous_positions[track_id] = center_y

            # Object label
            label = model.names[int(class_id)]

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"{label} ID:{track_id}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # Display total count
    cv2.putText(
        frame,
        f"Total Count: {count}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3
    )

    cv2.imshow("AI Bag Count Detection", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Final Count:", count)

