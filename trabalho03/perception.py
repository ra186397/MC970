from ultralytics import YOLO
import cv2


class ObjectDetector:
    def __init__(self, model_name="yolov8x.pt", confidence_threshold=0.5):
        """
        Initializes the YOLOv8 object detector using the ultralytics library.

        Model options (in order of accuracy/speed tradeoff):
        - yolov8n.pt: Nano - fastest, least accurate
        - yolov8s.pt: Small - good balance
        - yolov8m.pt: Medium - better accuracy (RECOMMENDED)
        - yolov8l.pt: Large - high accuracy, slower
        - yolov8x.pt: Extra Large - best accuracy, slowest
        """
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold

        # You can get the class names directly from the model
        self.target_class_names = [
            "person",
            "car",
            "motorcycle",
            "bicycle",  # Added
            "bus",
            "truck",
            "traffic light",
            "stop sign",
            "fire hydrant",  # Added - good for navigation
            "bench",
            "backpack",
            "umbrella",
            "suitcase",
            "bottle",
            "cup",
            "chair",
            "dining table",
            "tv",
            "laptop",
            "book",
            "clock",
            "dog",  # Added - important for blind navigation
            "cat",  # Added
            "potted plant",  # Added - obstacle
            "couch",  # Added
        ]

    def detect(self, frame, conf_threshold=None):
        """
        Detects objects in a frame using YOLOv8.
        Returns a list of filtered detections.
        """
        if conf_threshold is None:
            conf_threshold = self.confidence_threshold

        # IMPROVEMENT: Added conf parameter to filter at inference time
        # IMPROVEMENT: Added imgsz for better detection on high-res images
        results = self.model(
            frame,
            verbose=False,
            conf=conf_threshold,  # Filter low-confidence detections
            iou=0.45,  # Non-max suppression threshold
            imgsz=640,  # Image size for inference
        )

        filtered_detections = []
        # Process results for the first image (index 0)
        for box in results[0].boxes:
            # Get class name from class ID
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]

            # Filter for target classes
            if class_name in self.target_class_names:
                confidence = float(box.conf[0])

                # Additional filtering for specific classes
                # Require higher confidence for less critical objects
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
                    }
                )

        return filtered_detections


if __name__ == "__main__":
    # Test rápido do módulo
    detector = ObjectDetector(model_name="yolov8m.pt")  # Using medium model
    # Make sure to provide a valid path to a test image
    frame = cv2.imread("path/to/your/test_image.jpg")
    if frame is not None:
        detections = detector.detect(frame)
        for det in detections:
            print(
                f"Detectado: {det['class_name']} com confiança {det['confidence']:.2f}"
            )
            p1 = (det["bbox"][0], det["bbox"][1])
            p2 = (det["bbox"][2], det["bbox"][3])
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
        cv2.imshow("Detections", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
