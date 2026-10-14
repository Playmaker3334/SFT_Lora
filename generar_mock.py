import argparse
import json
from pathlib import Path

from sft_qwen35.corpus import leer_filas


def main():
    p = argparse.ArgumentParser(description="Replica ejemplos reales con ids nuevos para medir tiempos de entrenamiento.")
    p.add_argument("archivos", nargs="+", help="dataset.jsonl del pipeline, en orden: la ultima aparicion de cada id gana.")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--salida", default="datos/mock.jsonl")
    a = p.parse_args()

    filas = leer_filas(a.archivos)
    destino = Path(a.salida)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with open(destino, "w", encoding="utf-8") as f:
        for k in range(a.n):
            fila = filas[k % len(filas)]
            fuente = f"mock_{fila['fuente']}"
            id_documento = f"{fila['id_documento']}-{k}"
            copia = {**fila, "id": f"{fuente}-{id_documento}", "fuente": fuente, "id_documento": id_documento}
            f.write(json.dumps(copia, ensure_ascii=False) + "\n")
    print(f"{a.n} ejemplos a partir de {len(filas)} reales -> {destino}")


if __name__ == "__main__":
    main()
