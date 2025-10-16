import numpy as np

# Definindo as faixas de proximidade
PROXIMITY_RANGES = {
    "próximo": (0, 2.0),
    "médio": (2.0, 5.0),
    "longe": (5.0, float("inf")),
}

# CRITICAL FIX: MiDaS outputs inverse depth (disparity)
# Higher MiDaS values = CLOSER objects
# We need to convert disparity to actual distance
# This is a baseline - you'll need to calibrate with real measurements
BASELINE_DEPTH_SCALE = 3  # Adjust this value based on testing


def get_distance_in_meters(depth_map, bbox):
    """
    Estima a distância de um objeto em metros.
    Usa a mediana da profundidade dentro da bounding box para robustez.

    IMPORTANT: MiDaS outputs inverse depth (disparity), not actual depth.
    Higher values = closer objects, lower values = farther objects.
    """
    x1, y1, x2, y2 = bbox

    # Add safety bounds checking
    y1, y2 = max(0, y1), min(depth_map.shape[0], y2)
    x1, x2 = max(0, x1), min(depth_map.shape[1], x2)

    object_depth_view = depth_map[y1:y2, x1:x2]

    if object_depth_view.size == 0:
        return float("inf"), "desconhecida"

    # Use median for robustness against outliers
    median_disparity = np.median(object_depth_view)

    # FIX: Convert disparity to distance
    # Formula: distance ≈ baseline / disparity
    # We normalize by the max disparity value to get relative depth
    if median_disparity > 1e-6:  # Avoid division by zero
        # Normalize disparity to 0-1 range based on typical MiDaS output (0-255 after normalization)
        normalized_disparity = median_disparity / 255.0

        # Convert to distance (higher disparity = closer = smaller distance)
        distance = BASELINE_DEPTH_SCALE / (normalized_disparity + 1e-6)
    else:
        distance = float("inf")

    # Debug output
    print(f"[DEBUG] Disparity: {median_disparity:.2f}, Distance: {distance:.2f}m")

    # Categorize distance
    for name, (min_d, max_d) in PROXIMITY_RANGES.items():
        if min_d <= distance < max_d:
            return distance, name
    return distance, "muito longe"


def get_most_relevant_object(detections, frame_width, frame_height):
    """
    Define o objeto mais relevante com base em uma zona de interesse frontal (ROI).
    Prioriza objetos maiores e mais centrais.
    """
    if not detections:
        return None

    # ROI: terço central da imagem (zona de caminhada)
    roi_x_start = frame_width / 3
    roi_x_end = 2 * frame_width / 3

    relevant_objects = []
    for det in detections:
        x_center = (det["bbox"][0] + det["bbox"][2]) / 2
        if roi_x_start < x_center < roi_x_end:
            # Calcula a área da bbox como um critério de "importância"
            area = (det["bbox"][2] - det["bbox"][0]) * (det["bbox"][3] - det["bbox"][1])
            det["area"] = area
            relevant_objects.append(det)

    if not relevant_objects:
        return None

    # Retorna o maior objeto dentro da ROI
    return max(relevant_objects, key=lambda x: x["area"])


def get_all_relevant_objects(detections, frame_width, frame_height, top_n=5):
    """
    NEW FUNCTION: Returns multiple relevant objects instead of just one.
    Useful for drawing boxes around all detected objects.
    """
    if not detections:
        return []

    # ROI: terço central da imagem
    roi_x_start = frame_width / 3
    roi_x_end = 2 * frame_width / 3

    relevant_objects = []
    for det in detections:
        x_center = (det["bbox"][0] + det["bbox"][2]) / 2
        # Include objects in ROI or very close to camera (regardless of position)
        area = (det["bbox"][2] - det["bbox"][0]) * (det["bbox"][3] - det["bbox"][1])
        det["area"] = area

        if roi_x_start < x_center < roi_x_end or area > (
            frame_width * frame_height * 0.1
        ):
            relevant_objects.append(det)

    # Return top N largest objects
    relevant_objects.sort(key=lambda x: x["area"], reverse=True)
    return relevant_objects[:top_n]


def check_collision_risk(distance_meters):
    """
    Regra simples de risco com base na distância.
    """
    if distance_meters < PROXIMITY_RANGES["próximo"][1]:  # < 2.0 metros
        return True
    return False
