import json

from datasets import Dataset, Features, Value

from .corpus import TRAZABILIDAD

IGNORAR = -100

ENTRADA = Features(
    {
        "mensajes": Value("string"),
        "tokens_dataset": Value("int64"),
        **{c: Value("string") for c in TRAZABILIDAD},
    }
)


def ids_plantilla(tok, mensajes, generacion):
    ids = tok.apply_chat_template(mensajes, tokenize=True, add_generation_prompt=generacion)
    return list(ids["input_ids"] if hasattr(ids, "keys") else ids)


class Segmentador:
    def __init__(self, tokenizador):
        self.tok = tokenizador

    def __call__(self, lote):
        salida = {k: [] for k in ("input_ids", "labels", "n", "n_prompt", "difiere_dataset", *TRAZABILIDAD)}
        for i, crudo in enumerate(lote["mensajes"]):
            mensajes = json.loads(crudo)
            completo = ids_plantilla(self.tok, mensajes, False)
            prompt = ids_plantilla(self.tok, mensajes[:-1], True)
            if completo[: len(prompt)] != prompt:
                raise ValueError(f"{lote['id'][i]}: el prompt de generacion no es prefijo de la conversacion completa.")
            esperado = lote["tokens_dataset"][i]
            salida["input_ids"].append(completo)
            salida["labels"].append([IGNORAR] * len(prompt) + completo[len(prompt) :])
            salida["n"].append(len(completo))
            salida["n_prompt"].append(len(prompt))
            salida["difiere_dataset"].append(bool(esperado) and esperado != len(completo))
            for campo in TRAZABILIDAD:
                salida[campo].append(lote[campo][i])
        return salida


def construir(docs, tokenizador, cfg, destino):
    base = Dataset.from_list(docs, features=ENTRADA)
    segmentos = base.map(
        Segmentador(tokenizador),
        batched=True,
        batch_size=64,
        remove_columns=base.column_names,
        num_proc=min(cfg["num_proc"], max(len(base), 1)),
    )
    difieren = sum(segmentos["difiere_dataset"])
    if difieren:
        print(f"aviso: {difieren} ejemplos no coinciden con el conteo de tokens del dataset; revisar la plantilla de chat")
    exceden = segmentos.filter(lambda n: n > cfg["seq_len"], input_columns="n")
    if len(exceden):
        print(f"descartados por exceder seq_len={cfg['seq_len']}: {exceden['id']}")
    segmentos = segmentos.filter(lambda n: n <= cfg["seq_len"], input_columns="n").remove_columns("difiere_dataset")
    for part in ("train", "val"):
        sub = segmentos.filter(lambda p: p == part, input_columns="particion")
        sub.save_to_disk(str(destino / part / "segmentos"))
        tokens = sum(sub["n"])
        con_perdida = tokens - sum(sub["n_prompt"])
        print(f"{part}: ejemplos={len(sub)} tokens={tokens} con_perdida={con_perdida} ({con_perdida / max(tokens, 1):.4f})")
