# main.py
import cv2
import time
from collections import deque  # <-- IMPORTED
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


def main():
    # --- Inicialização ---
    detector = ObjectDetector(model_name="yolov8n.pt")  # Using Nano model for speed
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

    last_narrated_object = None
    last_narration_time = time.time()

    # --- ADDED: State for navigation commands ---
    last_nav_advice = None
    last_nav_time = time.time()
    NAV_ADVICE_COOLDOWN = 4.0  # Seconds between navigation commands
    # ---------------------------------------------

    cv2.namedWindow("Navegação Assistida", cv2.WINDOW_NORMAL)

    # Configuration: Set to False to hide depth map
    SHOW_DEPTH_MAP = True

    # Color mapping for distance categories (BGR format)
    DISTANCE_COLORS = {
        "próximo": (0, 0, 255),  # Red
        "médio": (0, 255, 255),  # Yellow
        "longe": (0, 255, 0),  # Green
        "muito longe": (0, 255, 0),  # Also Green
        "desconhecida": (255, 0, 0),  # Blue for unknown
    }

    # --- ADDED: Frame skipping logic ---
    frame_counter = 0
    PROCESS_EVERY_N_FRAMES = 3  # Only run AI models every 3 frames

    # --- ADDED: Caches for stale data ---
    cached_detections = []
    cached_depth_map = None
    cached_depth_display = None
    # -----------------------------------

    # --- ADDED: For average FPS calculation ---
    fps_history = deque(maxlen=20)
    # -----------------------------------------

    print("Sistema de Navegação Assistida iniciado. Pressione 'q' para sair.")

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Fim do vídeo ou erro na leitura do frame.")
            break

        frame_counter += 1

        # --- MODIFIED: AI processing block ---
        # Only run AI on designated frames
        if frame_counter % PROCESS_EVERY_N_FRAMES == 0:
            # 1. Detecção de Objetos
            cached_detections = detector.detect(frame)

            # 2. Estimativa de Profundidade
            cached_depth_map, cached_depth_display = depth_estimator.estimate(frame)

        # --- Safety check in case depth map isn't ready ---
        if cached_depth_map is None:
            # On the first few frames, just get the depth map to avoid errors
            cached_depth_map, cached_depth_display = depth_estimator.estimate(frame)
            continue
        # -----------------------------------------------

        # 3. Lógica de Navegação - VISUAL (Todos os Objetos)
        # --- MODIFIED: Use cached data ---
        all_relevant_objects = get_all_relevant_objects(
            cached_detections, frame_width, frame_height
        )

        # Loop through all detected objects to draw them
        for obj in all_relevant_objects:
            # Get distance and category for this specific object
            distance_m, distance_category = get_distance_in_meters(
                cached_depth_map, obj["bbox"]
            )

            # Get the correct color from our map
            color = DISTANCE_COLORS.get(
                distance_category, (255, 0, 0)
            )  # Default to blue

            # Draw the bounding box
            x1, y1, x2, y2 = obj["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw the label
            label = f"{obj['class_name']}: {distance_m:.2f}m"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        # --- MODIFIED: 4. Lógica de Áudio (Prioridade de Comandos) ---
        # --- MODIFIED: Use cached data ---
        most_relevant_object = get_most_relevant_object(
            cached_detections, frame_width, frame_height
        )

        is_critical_stop = False
        if most_relevant_object:
            # --- MODIFIED: Use cached data ---
            distance_m, distance_category = get_distance_in_meters(
                cached_depth_map, most_relevant_object["bbox"]
            )

            # P1: ALERTA DE COLISÃO (Prioridade Máxima)
            if check_collision_risk(distance_m):
                # --- MODIFIED: Shortened phrase ---
                message = "Pare! Obstáculo perto."
                # ----------------------------------
                speaker.speak(message, is_critical=True)
                is_critical_stop = True

                # Reset navigation advice state since "Stop" overrides it
                last_nav_advice = message
                last_nav_time = time.time()

        # P2: ACONSELHAMENTO DE NAVEGAÇÃO (Se não houver parada crítica)
        if not is_critical_stop:
            current_time = time.time()

            # Only check for navigation advice every few seconds to avoid spam
            if current_time - last_nav_time > NAV_ADVICE_COOLDOWN:

                # Analyze all three zones using ALL detections
                # --- MODIFIED: Use cached data ---
                dist_l, dist_c, dist_r = analyze_zones(
                    cached_detections, cached_depth_map, frame_width, 10.0
                )

                # Get the spoken advice
                advice = get_navigation_advice(dist_l, dist_c, dist_r)

                # Speak the advice *only if* it's new and not empty
                if advice and advice != last_nav_advice:
                    # Use is_critical=False so it doesn't interrupt, just queues
                    speaker.speak(advice, is_critical=False)
                    last_nav_advice = advice
                    last_nav_time = current_time
                elif not advice:
                    # If advice is None (clear path), reset last_nav_advice
                    # so it will speak again when an obstacle *does* appear
                    last_nav_advice = None

        # P3: NARRAÇÃO DE OBJETOS (Prioridade Baixa)
        # (Your commented-out code remains here)

        # --- MODIFIED: Medição de Desempenho e Visualização ---
        end_time = time.time()

        # Calculate average FPS
        frame_time = (end_time - start_time) if (end_time - start_time) > 0 else 0
        fps_history.append(frame_time)

        if len(fps_history) > 0:
            # Calculate average time per frame, then convert to FPS
            avg_frame_time = sum(fps_history) / len(fps_history)
            avg_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            fps_text = f"FPS: {avg_fps:.2f}"
        else:
            fps_text = "FPS: N/A"

        cv2.putText(
            frame,
            fps_text,  # <-- Use the new averaged text
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        # Concatena as visualizações
        if SHOW_DEPTH_MAP:
            # --- MODIFIED: Use cached display ---
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
