"""Baixa os pesos do YOLOv3 para a pasta models/.

Os arquivos .weights são grandes demais para o GitHub (o yolov3 tem ~237 MB),
por isso ficam fora do repositório.

Exemplos:
    python download_models.py                 # baixa os dois modelos
    python download_models.py yolov3-tiny     # só o modelo leve
"""

import argparse
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"

WEIGHTS_URLS = {
    "yolov3": "https://pjreddie.com/media/files/yolov3.weights",
    "yolov3-tiny": "https://pjreddie.com/media/files/yolov3-tiny.weights",
}


def show_progress(block_count: int, block_size: int, total_size: int) -> None:
    if total_size > 0:
        percent = min(block_count * block_size / total_size, 1)
        print(f"\r  {percent:.0%} de {total_size / 1e6:.0f} MB", end="", flush=True)


def download(model_name: str) -> None:
    destination = MODELS_DIR / f"{model_name}.weights"
    if destination.exists():
        print(f"{destination.name} já existe, pulando.")
        return

    print(f"Baixando {destination.name}...")
    temp_path = destination.with_suffix(".part")
    urllib.request.urlretrieve(WEIGHTS_URLS[model_name], temp_path, show_progress)
    temp_path.rename(destination)  # só renomeia se o download terminou
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Baixa os pesos do YOLOv3.")
    parser.add_argument("models", nargs="*", help=f"modelos a baixar: {', '.join(WEIGHTS_URLS)} (padrão: todos)")
    args = parser.parse_args()

    unknown = set(args.models) - WEIGHTS_URLS.keys()
    if unknown:
        parser.error(f"modelo desconhecido: {', '.join(sorted(unknown))}")

    for model_name in args.models or WEIGHTS_URLS:
        download(model_name)


if __name__ == "__main__":
    main()
