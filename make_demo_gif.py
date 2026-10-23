"""Gera um GIF com a detecção rodando num vídeo (o assets/demo.gif do README foi feito assim).

Usa o mesmo modelo do yolo_detect_video.py (MODEL_NAME). Precisa do Pillow,
que não faz parte do requirements.txt:
    pip install pillow
    python make_demo_gif.py
    python make_demo_gif.py --source meu_video.mp4 --output meu_teste.gif
"""

import argparse
from pathlib import Path

import cv2
from PIL import Image

from yolo_detect_video import BASE_DIR, VIDEO_PATH, detect_objects


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera um GIF da detecção para o README.")
    parser.add_argument("--source", default=VIDEO_PATH)
    parser.add_argument("--output", default=str(Path(BASE_DIR) / "assets" / "demo.gif"))
    parser.add_argument("--width", type=int, default=640, help="largura do GIF em pixels")
    parser.add_argument("--step", type=int, default=2, help="usa 1 a cada N frames (GIF menor)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.source)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30

    frames = []
    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % args.step == 0:
            # Reduz antes de detectar, senão as legendas encolhem junto e ficam ilegíveis.
            # A rede recebe 416x416 de qualquer jeito, então a detecção praticamente não muda
            scale = args.width / frame.shape[1]
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            detect_objects(frame)
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            print(f"\r{len(frames)} frames", end="", flush=True)
        frame_index += 1
    cap.release()
    print()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Duração de cada frame = tempo real que ele representa no vídeo original
    frame_ms = int(1000 * args.step / video_fps)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=frame_ms, loop=0, optimize=True)
    print(f"{output} ({output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
