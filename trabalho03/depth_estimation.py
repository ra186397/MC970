import torch
import cv2
import numpy as np


class DepthEstimator:
    def __init__(self, model_type="MiDaS_small"):
        """
        Initializes the MiDaS depth estimator.
        Models available: "MiDaS_small", "DPT_Large", "DPT_Hybrid"
        """
        self.model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model.to(self.device)
        self.model.eval()

        # Load the correct transform for the chosen model
        midas_transforms = torch.hub.load(
            "intel-isl/MiDaS", "transforms", trust_repo=True
        )
        if model_type in ["DPT_Large", "DPT_Hybrid"]:
            self.transform = midas_transforms.dpt_transform
        else:
            # This is the corrected line for "MiDaS_small"
            self.transform = midas_transforms.small_transform

    def estimate(self, frame):
        """
        Estimates the depth map of a video frame.
        Returns a raw depth map and a display-friendly version.
        """
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()

        # Normalize the map for visualization (0-255)
        output_display = cv2.normalize(
            depth_map, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U
        )
        output_display = cv2.cvtColor(output_display, cv2.COLOR_GRAY2BGR)

        return depth_map, output_display


if __name__ == "__main__":
    # Test rápido do módulo
    estimator = DepthEstimator()
    frame = cv2.imread("path/to/your/test_image.jpg")
    if frame is not None:
        _, depth_display = estimator.estimate(frame)
        cv2.imshow("Depth Map", depth_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
