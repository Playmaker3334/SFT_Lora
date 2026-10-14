import difflib
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from peft import PeftModel

from .tokenizacion import IGNORAR


def _hash(items):
    return hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()


def _claves(segmentos):
    columnas = segmentos.select_columns(["fuente", "id_documento"])
    return set(zip(columnas["fuente"], (str(x) for x in columnas["id_documento"])))


def construir_lotes(cfg, dir_datos, semilla):
    destino = Path(cfg["lotes"])
    destino.mkdir(parents=True, exist_ok=True)
    ruta = destino / "val.json"
    val = load_from_disk(str(Path(dir_datos) / "val" / "segmentos"))
    claves_train = _claves(load_from_disk(str(Path(dir_datos) / "train" / "segmentos")))
    if ruta.exists():
        items = json.loads(ruta.read_text())["items"]
        contaminados = {(it["fuente"], str(it["id_documento"])) for it in items} & claves_train
        if contaminados:
            raise SystemExit(
                f"Lote val: {len(contaminados)} documentos están en train (¿cambió fraccion_val?). "
                f"Borra {ruta} para regenerarlo; dejará de ser comparable con corridas anteriores."
            )
        print("lote val: ya existe, no se regenera; sin documentos en train")
        return
    if not len(val):
        print("lote val: sin ejemplos de validación")
        return
    rng = np.random.default_rng(semilla)
    elegidos = sorted(rng.choice(len(val), min(cfg["n_ejemplos"], len(val)), replace=False).tolist())
    items = [
        {
            "input_ids": r["input_ids"][: cfg["max_tokens"]],
            "labels": r["labels"][: cfg["max_tokens"]],
            "id": r["id"],
            "fuente": r["fuente"],
            "tipo_tarea": r["tipo_tarea"],
            "id_documento": r["id_documento"],
        }
        for r in val.select(elegidos)
    ]
    ruta.write_text(json.dumps({"sha256": _hash(items), "items": items}))
    print(f"lote val: {len(items)} ejemplos")


def cargar_lotes(cfg):
    lotes = {}
    for ruta in sorted(Path(cfg["lotes"]).glob("*.json")):
        datos = json.loads(ruta.read_text())
        if _hash(datos["items"]) != datos["sha256"]:
            raise SystemExit(f"El lote {ruta} fue modificado: el hash no coincide.")
        lotes[ruta.stem] = datos["items"]
    return lotes


def _nucleo(modelo):
    base = modelo.get_base_model() if isinstance(modelo, PeftModel) else modelo
    return base.model, base.lm_head


def _ocultos(backbone, ids):
    return backbone(input_ids=ids, use_cache=False).last_hidden_state[0]


def _metricas(v):
    t = max(v[0], 1.0)
    loss, loss_base = v[1] / t, v[4] / t
    return {
        "tokens": int(v[0]),
        "loss": loss,
        "perplejidad": math.exp(min(loss, 50)),
        "entropia": v[2] / t,
        "kl_contra_base": v[3] / t,
        "loss_base": loss_base,
        "ganancia": loss_base - loss,
    }


@torch.no_grad()
def sondear(modelo, lotes, fragmento=512):
    estaba = modelo.training
    modelo.eval()
    backbone, cabeza = _nucleo(modelo)
    comparar = isinstance(modelo, PeftModel)
    dispositivo = cabeza.weight.device
    resultado = {}
    for nombre, items in lotes.items():
        acumulado = defaultdict(lambda: np.zeros(5))
        for it in items:
            ids = torch.tensor([it["input_ids"]], device=dispositivo)
            objetivo = torch.tensor(it["labels"][1:], device=dispositivo)
            h = _ocultos(backbone, ids)[:-1]
            if comparar:
                with modelo.disable_adapter():
                    hb = _ocultos(backbone, ids)[:-1]
            suma = np.zeros(5)
            for a in range(0, len(objetivo), fragmento):
                v = objetivo[a : a + fragmento] != IGNORAR
                if not v.any():
                    continue
                t = objetivo[a : a + fragmento][v]
                lp = F.log_softmax(cabeza(h[a : a + fragmento][v]).float(), dim=-1)
                nll = -lp.gather(1, t[:, None]).squeeze(1)
                p = lp.exp()
                ent = -(p * lp).sum(-1)
                if comparar:
                    lq = F.log_softmax(cabeza(hb[a : a + fragmento][v]).float(), dim=-1)
                    kl = (p * (lp - lq)).sum(-1)
                    nll_base = -lq.gather(1, t[:, None]).squeeze(1)
                else:
                    kl = torch.zeros_like(nll)
                    nll_base = nll
                suma += [v.sum().item(), nll.sum().item(), ent.sum().item(), kl.sum().item(), nll_base.sum().item()]
            for clave in ("total", f"fuente={it['fuente']}", f"tipo_tarea={it['tipo_tarea']}"):
                acumulado[clave] += suma
        resultado[nombre] = {k: _metricas(v) for k, v in sorted(acumulado.items())}
    modelo.train(estaba)
    return resultado


@torch.no_grad()
def generar(modelo, tok, items, n, max_nuevos):
    dispositivo = _nucleo(modelo)[1].weight.device
    fin = [t for t in dict.fromkeys((tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id)) if t is not None]
    salida = []
    for it in items[:n]:
        inicio = next((i for i, y in enumerate(it["labels"]) if y != IGNORAR), None)
        if inicio is None:
            continue
        ids = torch.tensor([it["input_ids"][:inicio]], device=dispositivo)
        out = modelo.generate(
            input_ids=ids,
            max_new_tokens=max_nuevos,
            do_sample=False,
            use_cache=True,
            eos_token_id=fin,
            pad_token_id=fin[0],
        )
        nuevos = out[0, ids.shape[1] :].tolist()
        referencia = it["input_ids"][inicio:]
        salida.append(
            {
                "id": it["id"],
                "tokens": len(nuevos),
                "termino": bool(nuevos) and nuevos[-1] in fin,
                "similitud_literal": difflib.SequenceMatcher(a=nuevos, b=referencia, autojunk=False).ratio(),
                "texto": tok.decode(nuevos, skip_special_tokens=True),
                "referencia": tok.decode(referencia, skip_special_tokens=True),
            }
        )
    return salida


def evaluar(modelo, tok, cfg, destino):
    lotes = cargar_lotes(cfg)
    resultado = {"sondeo": sondear(modelo, lotes, cfg["fragmento"])}
    if cfg.get("generar") and lotes.get("val"):
        resultado["generacion"] = generar(modelo, tok, lotes["val"], cfg["generar"], cfg["max_nuevos_tokens"])
    Path(destino).parent.mkdir(parents=True, exist_ok=True)
    Path(destino).write_text(json.dumps(resultado, ensure_ascii=False, indent=2))
    return resultado
