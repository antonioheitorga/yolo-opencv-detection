import cv2
import numpy as np
import os
import time

# Limiares
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

# Pasta raiz do projeto (a pasta deste arquivo)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Video que sera analisado (use 0 para a webcam)
VIDEO_PATH = os.path.join(BASE_DIR, "samples", "videoTeste.mp4")

# Modelo: "yolov3" (mais preciso) ou "yolov3-tiny" (mais rápido)
MODEL_NAME = "yolov3"
MODEL_CFG = os.path.join(BASE_DIR, "models", MODEL_NAME + ".cfg")
MODEL_WEIGHTS = os.path.join(BASE_DIR, "models", MODEL_NAME + ".weights")

# Classes do COCO dataset (uma por linha, na ordem do treino)
with open(os.path.join(BASE_DIR, "models", "coco.names"), encoding="utf-8") as f:
    class_labels = f.read().splitlines()

# Cores
class_colors = ["0,255,0", "0,0,255", "255,0,0", "255,255,0", "0,255,255"]
class_colors = [np.array(c.split(",")).astype("int") for c in class_colors]
class_colors = np.tile(np.array(class_colors), (16, 1))

if not os.path.exists(MODEL_WEIGHTS):
    raise FileNotFoundError(f"{MODEL_WEIGHTS} não encontrado. Rode: python download_models.py {MODEL_NAME}")

yolo_model = cv2.dnn.readNetFromDarknet(MODEL_CFG, MODEL_WEIGHTS)
# Usa a GPU via OpenCL (~2.5x mais rápido); sem OpenCL, o OpenCV volta sozinho para a CPU
yolo_model.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
yolo_layers = yolo_model.getLayerNames()
yolo_output_layer = yolo_model.getUnconnectedOutLayersNames()


def detect_objects(frame):
    # Obter dimensões da imagem
    img_height = frame.shape[0]
    img_width = frame.shape[1]

    # Pré-processamento da imagem
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416,416), swapRB=True, crop=False)
    yolo_model.setInput(blob)
    obj_detection_layers = yolo_model.forward(yolo_output_layer)

    # Listas criadas a cada frame
    class_ids_list = []
    boxes_list = []
    confidences_list = []

    # Coleta das caixas
    for object_detection_layer in obj_detection_layers:
        for object_detection in object_detection_layer:
            all_scores = object_detection[5:]
            predicted_class_id = np.argmax(all_scores)
            predicted_confidence = all_scores[predicted_class_id]

            #Desenha as caixas delimitadoras apenas se a confiança for maior que o limiar
            if predicted_confidence > CONF_THRESHOLD:
                # Obter coordenadas da caixa delimitadora
                bounding_box = object_detection[0:4] * np.array([img_width, img_height, img_width, img_height])
                (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype("int")

                start_x_pt = int(box_center_x - (box_width / 2))
                start_y_pt = int(box_center_y - (box_height / 2))

                # Adiciona as informações da caixa delimitadora e da confiança às listas
                class_ids_list.append(predicted_class_id)
                confidences_list.append(float(predicted_confidence))
                boxes_list.append([start_x_pt, start_y_pt, int(box_width), int(box_height)])

    # NMS (Non-Maximum Suppression) para eliminar caixas sobrepostas
    max_value_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, CONF_THRESHOLD, NMS_THRESHOLD)

    # Desenha as caixas delimitadoras finais
    # (dtype=int: sem nenhuma caixa, o array vazio viraria float e quebraria o índice)
    for idx in np.array(max_value_ids, dtype=int).flatten():
        start_x_pt, start_y_pt, box_width, box_height = boxes_list[idx]
        end_x_pt = start_x_pt + box_width
        end_y_pt = start_y_pt + box_height

        predicted_class_id = class_ids_list[idx]
        box_color = [int(c) for c in class_colors[predicted_class_id]]
        label = f"{class_labels[predicted_class_id]}: {confidences_list[idx] * 100:.0f}%"

        cv2.rectangle(frame, (start_x_pt, start_y_pt), (end_x_pt, end_y_pt), box_color, 2)
        cv2.putText(frame, label, (start_x_pt, start_y_pt - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)


if __name__ == "__main__":
    cap = cv2.VideoCapture(VIDEO_PATH)

    # No arquivo de vídeo, espera o tempo de cada frame para não tocar acelerado
    # (a webcam já entrega o frame mais recente, então não precisa esperar)
    is_webcam = isinstance(VIDEO_PATH, int)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = 0 if is_webcam else 1 / video_fps
    fps = 0

    while True:
        frame_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        detect_objects(frame)

        # FPS (média móvel, para o número não ficar pulando)
        elapsed = time.perf_counter() - frame_start
        fps = 0.9 * fps + 0.1 / elapsed if fps else 1 / elapsed
        cv2.putText(frame, f"{MODEL_NAME} | {fps:.1f} FPS", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Mostra o frame
        cv2.imshow("YOLO Object Detection", frame)

        wait_ms = max(1, int((frame_interval - elapsed) * 1000))
        if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
