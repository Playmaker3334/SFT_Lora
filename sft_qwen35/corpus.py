import hashlib
import json
from collections import Counter

TRAZABILIDAD = ("id", "fuente", "id_documento", "version_prompts", "tipo_tarea", "particion")


def particion(clave, fraccion_val):
    h = int(hashlib.sha256(f"{clave[0]}\x1f{clave[1]}".encode()).hexdigest()[:8], 16)
    return "val" if h / 0xFFFFFFFF < fraccion_val else "train"


def leer_filas(archivos):
    filas = {}
    for ruta in archivos:
        with open(ruta, encoding="utf-8") as f:
            for fila in (json.loads(l) for l in f if l.strip()):
                filas[fila["id"]] = fila
    return list(filas.values())


def pasa_filtros(fila, filtros):
    return all(not permitidos or fila.get(campo) in permitidos for campo, permitidos in (filtros or {}).items())


def documentos(cfg):
    docs = []
    for fila in leer_filas(cfg["archivos"]):
        if not pasa_filtros(fila, cfg.get("filtros")):
            continue
        if fila["messages"][-1]["role"] != "assistant":
            raise SystemExit(f"{fila['id']}: el ultimo mensaje no es del asistente.")
        clave = (fila["fuente"], str(fila["id_documento"]))
        docs.append(
            {
                "mensajes": json.dumps(fila["messages"], ensure_ascii=False),
                "tokens_dataset": int(fila.get("tokens") or 0) if fila.get("tokens_exactos") else 0,
                "id": fila["id"],
                "fuente": fila["fuente"],
                "id_documento": clave[1],
                "version_prompts": fila.get("version_prompts") or "",
                "tipo_tarea": fila.get("tipo_tarea") or "",
                "particion": particion(clave, cfg["fraccion_val"]),
            }
        )
    conteo = Counter((d["particion"], d["tipo_tarea"]) for d in docs)
    for (part, tipo), n in sorted(conteo.items()):
        print(f"{part:<5} {tipo:<12} {n}")
    return docs
