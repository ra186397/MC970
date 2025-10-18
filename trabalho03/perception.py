from ultralytics import YOLO
import cv2


class ObjectDetector:
    def __init__(self, model_name="yolov8n.pt", confidence_threshold=0.5):
        """
        Initializes the YOLOv8 object detector using the ultralytics library.
        """
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold

        # --- MODIFIED: Expanded class list ---
        self.target_class_names = [
            "person",
            "car",
            "motorcycle",
            "bicycle",
            "bus",
            "truck",
            "train",
            "traffic light",
            "stop sign",
            "fire hydrant",
            "parking meter",
            "bench",
            "backpack",
            "handbag",
            "suitcase",
            "skateboard",
            "bottle",
            "cup",
            "chair",
            "dining table",
            "tv",
            "laptop",
            "book",
            "clock",
            "vase",
            "dog",
            "cat",
            "potted plant",
            "couch",
        ]
        # -------------------------------------

    def detect(self, frame, conf_threshold=None):
        """
        --- MODIFIED ---
        Detects AND TRACKS objects in a frame using YOLOv8.
        Returns a list of filtered detections, now including a 'tracker_id'.
        """
        if conf_threshold is None:
            conf_threshold = self.confidence_threshold

        # --- MODIFIED: Use model.track() instead of model() ---
        # 'persist=True' tells the tracker to remember objects between frames
        results = self.model.track(
            frame,
            verbose=False,
            conf=conf_threshold,
            iou=0.45,
            imgsz=640,
            persist=True,  # <-- Key change for tracking
        )
        # ------------------------------------------------------

        filtered_detections = []
        # Process results for the first image (index 0)
        if results[0].boxes is None:
            return []

        for box in results[0].boxes:
            # --- ADDED: Get tracker ID ---
            # Only include if it's a tracked object
            if box.id is None:
                continue
            tracker_id = int(box.id[0])
            # -----------------------------

            # Get class name from class ID
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]

            # Filter for target classes
            if class_name in self.target_class_names:
                confidence = float(box.conf[0])

                # Additional filtering
                if (
                    class_name in ["bottle", "cup", "book", "clock"]
                    and confidence < 0.6
                ):
                    continue

                filtered_detections.append(
                    {
                        "bbox": [
                            int(coord) for coord in box.xyxy[0]
                        ],  # [xmin, ymin, xmax, ymax]
                        "confidence": confidence,
                        "class_name": class_name,
                        "tracker_id": tracker_id,  # <-- ADDED
                    }
                )

        return filtered_detections


if __name__ == "__main__":
    # Test rápido do módulo
    detector = ObjectDetector(model_name="yolov8m.pt")
    frame = cv2.imread("path/to/your/test_image.jpg")
    if frame is not None:
        detections = detector.detect(frame)  # Now this also tracks
        for det in detections:
            print(
                f"Detectado: {det['class_name']} (ID: {det['tracker_id']}) com confiança {det['confidence']:.2f}"
            )
            p1 = (det["bbox"][0], det["bbox"][1])
            p2 = (det["bbox"][2], det["bbox"][3])
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
        cv2.imshow("Detections", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
