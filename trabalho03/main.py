# main.py
import cv2
import time
from collections import deque
from perception import ObjectDetector
from depth_estimation import DepthEstimator
from navigation_logic import (
    get_most_relevant_object,
    get_all_relevant_objects,
    get_distance_in_meters,
    check_collision_risk,
    analyze_zones,
    get_navigation_advice,
)
from speech_synthesis import Speaker


def get_filtered_distance(history_deque):
    """Calculates the average of valid distances, ignoring 'inf'."""
    valid_dists = [d for d in history_deque if d != float("inf")]
    if not valid_dists:
        return float("inf")
    return sum(valid_dists) / len(valid_dists)


def main():
    # --- Inicialização ---
    detector = ObjectDetector(model_name="yolov8n.pt")
    depth_estimator = DepthEstimator(model_type="MiDaS_small")
    speaker = Speaker()

    video_path = "./new_york_walk.mp4"
    cap = cv2.VideoCapture(video_path)
    # cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: Não foi possível abrir a câmera.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- State for navigation commands ---
    last_nav_advice = None
    last_nav_time = time.time()
    NAV_ADVICE_COOLDOWN = 4.0

    cv2.namedWindow("Navegação Assistida", cv2.WINDOW_NORMAL)
    SHOW_DEPTH_MAP = True
    DISTANCE_COLORS = {
        "próximo": (0, 0, 255),  # Red
        "médio": (0, 255, 255),  # Yellow
        "longe": (0, 255, 0),  # Green
        "muito longe": (0, 255, 0),
        "desconhecida": (255, 0, 0),
    }

    # --- Frame skipping logic ---
    frame_counter = 0
    PROCESS_EVERY_N_FRAMES = 3

    # --- Caches for stale data ---
    cached_detections = []
    cached_depth_map = None
    cached_depth_display = None

    # --- For average FPS calculation ---
    fps_history = deque(maxlen=20)

    # --- For zone distance filtering ---
    FILTER_LENGTH_ZONES = 5
    left_dist_hist = deque(maxlen=FILTER_LENGTH_ZONES)
    center_dist_hist = deque(maxlen=FILTER_LENGTH_ZONES)
    right_dist_hist = deque(maxlen=FILTER_LENGTH_ZONES)

    # --- ADDED: For tracking and narration ---
    # For P3: Object Narration
    narrated_objects = {}  # {tracker_id: last_narration_time}
    NARRATION_COOLDOWN = 10.0  # Seconds before re-narrating same object

    # For filtered bounding box distances
    object_distance_histories = {}  # {tracker_id: deque(maxlen=5)}
    FILTER_LENGTH_OBJECT = 5
    # ------------------------------------------

    print("Sistema de Navegação Assistida iniciado. Pressione 'q' para sair.")

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1

        if frame_counter % PROCESS_EVERY_N_FRAMES == 0:
            cached_detections = detector.detect(frame)  # Now this tracks
            cached_depth_map, cached_depth_display = depth_estimator.estimate(frame)

        if cached_depth_map is None:
            cached_depth_map, cached_depth_display = depth_estimator.estimate(frame)
            continue

        # 3. Lógica de Navegação - VISUAL (Todos os Objetos)
        all_relevant_objects = get_all_relevant_objects(
            cached_detections, frame_width, frame_height
        )

        for obj in all_relevant_objects:
            # --- MODIFIED: Use tracker_id for filtered distance ---
            if "tracker_id" not in obj:
                continue

            tracker_id = obj["tracker_id"]

            # Get raw distance for this frame
            distance_m_raw, distance_category = get_distance_in_meters(
                cached_depth_map, obj["bbox"]
            )

            # Get or create the distance history for this object
            if tracker_id not in object_distance_histories:
                object_distance_histories[tracker_id] = deque(
                    maxlen=FILTER_LENGTH_OBJECT
                )

            # Add new raw distance to its history
            if distance_m_raw != float("inf"):
                object_distance_histories[tracker_id].append(distance_m_raw)

            # Get the stable, filtered distance
            distance_m_filtered = get_filtered_distance(
                object_distance_histories[tracker_id]
            )
            if distance_m_filtered == float("inf"):
                distance_m_filtered = distance_m_raw  # Fallback
            # ----------------------------------------------------

            color = DISTANCE_COLORS.get(distance_category, (255, 0, 0))
            x1, y1, x2, y2 = obj["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # --- MODIFIED: Label now shows filtered distance and ID ---
            label = f"ID {tracker_id}: {obj['class_name']} ({distance_m_filtered:.2f}m)"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        # 4. Lógica de Áudio (Prioridade de Comandos)
        most_relevant_object = get_most_relevant_object(
            cached_detections, frame_width, frame_height
        )

        is_critical_stop = False
        if most_relevant_object:
            # P1 uses RAW distance for instant safety
            distance_m_raw, _ = get_distance_in_meters(
                cached_depth_map, most_relevant_object["bbox"]
            )

            # P1: ALERTA DE COLISÃO (Prioridade Máxima)
            if check_collision_risk(distance_m_raw):
                message = "Pare! Obstáculo perto."
                speaker.speak(message, is_critical=True)
                is_critical_stop = True
                last_nav_advice = message
                last_nav_time = time.time()

        # P2: ACONSELHANDO DE NAVEGAÇÃO (Zonas filtradas)
        if not is_critical_stop:
            current_time = time.time()

            if current_time - last_nav_time > NAV_ADVICE_COOLDOWN:
                # Get RAW distances for zones
                dist_l_raw, dist_c_raw, dist_r_raw = analyze_zones(
                    cached_detections, cached_depth_map, frame_width, 10.0
                )

                # Add to history
                left_dist_hist.append(dist_l_raw)
                center_dist_hist.append(dist_c_raw)
                right_dist_hist.append(dist_r_raw)

                # Get FILTERED distances for advice
                dist_l = get_filtered_distance(left_dist_hist)
                dist_c = get_filtered_distance(center_dist_hist)
                dist_r = get_filtered_distance(right_dist_hist)

                advice = get_navigation_advice(dist_l, dist_c, dist_r)

                if advice and advice != last_nav_advice:
                    speaker.speak(advice, is_critical=False)
                    last_nav_advice = advice
                    last_nav_time = current_time
                elif not advice:
                    last_nav_advice = None

        # --- MODIFIED: P3: NARRAÇÃO DE OBJETOS (com Tracking) ---
        if not is_critical_stop:
            current_time = time.time()
            # Find the most relevant object *that hasn't been narrated recently*
            for obj in all_relevant_objects:
                if "tracker_id" not in obj:
                    continue

                tracker_id = obj["tracker_id"]

                # Check if it's new or the cooldown has passed
                if (
                    current_time - narrated_objects.get(tracker_id, 0)
                ) > NARRATION_COOLDOWN:

                    # Get its filtered distance
                    if tracker_id in object_distance_histories:
                        dist_m = get_filtered_distance(
                            object_distance_histories[tracker_id]
                        )

                        # Only narrate if it's in a meaningful range
                        if dist_m < 10.0 and dist_m != float("inf"):
                            # Get category name from raw distance
                            _, distance_category = get_distance_in_meters(
                                cached_depth_map, obj["bbox"]
                            )

                            if (
                                distance_category != "longe"
                                and distance_category != "muito longe"
                            ):
                                message = f"{obj['class_name']} {distance_category}."
                                speaker.speak(message, is_critical=False)
                                narrated_objects[tracker_id] = current_time

                                # Break after one narration to avoid spam
                                break
        # --------------------------------------------------------

        # --- ADDED: Cleanup stale trackers from memory ---
        current_tracker_ids = {obj.get("tracker_id") for obj in all_relevant_objects}

        stale_ids_narration = [
            tid for tid in narrated_objects if tid not in current_tracker_ids
        ]
        for tid in stale_ids_narration:
            del narrated_objects[tid]

        stale_ids_distance = [
            tid for tid in object_distance_histories if tid not in current_tracker_ids
        ]
        for tid in stale_ids_distance:
            del object_distance_histories[tid]
        # ------------------------------------------------

        # --- Medição de Desempenho e Visualização ---
        end_time = time.time()
        frame_time = (end_time - start_time) if (end_time - start_time) > 0 else 0
        fps_history.append(frame_time)

        if len(fps_history) > 0:
            avg_frame_time = sum(fps_history) / len(fps_history)
            avg_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            fps_text = f"FPS: {avg_fps:.2f}"
        else:
            fps_text = "FPS: N/A"

        cv2.putText(
            frame, fps_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )

        if SHOW_DEPTH_MAP:
            combined_view = cv2.hconcat([frame, cached_depth_display])
        else:
            combined_view = frame

        cv2.imshow("Navegação Assistida", combined_view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Encerrando...")
            break

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)


if __name__ == "__main__":
    main()
