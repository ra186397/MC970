# main.py
import cv2
import time
from perception import ObjectDetector
from depth_estimation import DepthEstimator
from navigation_logic import (
    get_most_relevant_object,
    get_all_relevant_objects,
    get_distance_in_meters,
    check_collision_risk,
)
from speech_synthesis import Speaker


def main():
    # --- Inicialização ---
    detector = ObjectDetector(model_name="yolov8x.pt")
    depth_estimator = DepthEstimator(model_type="MiDaS_small")
    speaker = Speaker()

    # Use 0 para a webcam padrão. Se for um celular via app (DroidCam, etc.), pode ser outro índice.
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

    # FIX: Create the window ONCE before the loop
    cv2.namedWindow("Navegação Assistida", cv2.WINDOW_NORMAL)

    # Configuration: Set to False to hide depth map
    SHOW_DEPTH_MAP = True

    print("Sistema de Navegação Assistida iniciado. Pressione 'q' para sair.")

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Fim do vídeo ou erro na leitura do frame.")
            break

        # --- Pipeline de Processamento ---
        # 1. Detecção de Objetos
        detections = detector.detect(frame)

        # 2. Estimativa de Profundidade
        depth_map, depth_display = depth_estimator.estimate(frame)

        # 3. Lógica de Navegação e Risco
        relevant_object = get_most_relevant_object(
            detections, frame_width, frame_height
        )

        if relevant_object:
            distance_m, distance_category = get_distance_in_meters(
                depth_map, relevant_object["bbox"]
            )

            # Lógica de Alerta de Colisão (Prioridade Máxima)
            if check_collision_risk(distance_m):
                message = "Pare! Obstáculo muito próximo."
                speaker.speak(message, is_critical=True)

            # Lógica de Narração de Objetos
            else:
                # Evita spam de voz para o mesmo objeto
                if (
                    relevant_object["class_name"] != last_narrated_object
                    or time.time() - last_narration_time > 5
                ):
                    message = f"{relevant_object['class_name']} na faixa {distance_category}, a aproximadamente {distance_m:.1f} metros."
                    speaker.speak(message)
                    last_narrated_object = relevant_object["class_name"]
                    last_narration_time = time.time()

            # Desenha informações no frame para depuração
            x1, y1, x2, y2 = relevant_object["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{relevant_object['class_name']}: {distance_m:.2f}m"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # --- Medição de Desempenho e Visualização ---
        end_time = time.time()
        fps = 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0
        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        # Concatena as visualizações
        if SHOW_DEPTH_MAP:
            combined_view = cv2.hconcat([frame, depth_display])
        else:
            combined_view = frame

        # FIX: Use the pre-created window
        cv2.imshow("Navegação Assistida", combined_view)

        # FIX: Proper key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Encerrando...")
            break

    # FIX: Proper cleanup
    cap.release()
    cv2.destroyAllWindows()

    # FIX: Force window destruction (sometimes needed on Linux/Mac)
    cv2.waitKey(1)


if __name__ == "__main__":
    main()
