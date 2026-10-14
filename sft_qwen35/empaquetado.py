import json
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from sortedcontainers import SortedList
from torch.utils.data import Dataset
from transformers.data.data_collator import DataCollatorWithFlattening


def bfd(longitudes, max_len):
    libres = SortedList()
    contenedores = []
    for i in np.argsort(-longitudes, kind="stable"):
        n = int(longitudes[i])
        j = libres.bisect_left((n, -1))
        if j < len(libres):
            capacidad, c = libres.pop(j)
            contenedores[c].append(int(i))
            if capacidad > n:
                libres.add((capacidad - n, c))
        else:
            contenedores.append([int(i)])
            if max_len > n:
                libres.add((max_len - n, len(contenedores) - 1))
    return contenedores


def empaquetar(dir_particion, max_len):
    dir_particion = Path(dir_particion)
    n = np.asarray(load_from_disk(str(dir_particion / "segmentos"))["n"], dtype=np.int64)
    contenedores = bfd(n, max_len)
    (dir_particion / "contenedores.json").write_text(json.dumps(contenedores))
    print(
        f"{dir_particion.name}: ejemplos={len(n)} contenedores={len(contenedores)} "
        f"ocupacion={int(n.sum()) / max(len(contenedores) * max_len, 1):.4f}"
    )


class Contenedores(Dataset):
    def __init__(self, dir_particion):
        dir_particion = Path(dir_particion)
        self.seg = load_from_disk(str(dir_particion / "segmentos")).select_columns(["input_ids", "labels"])
        self.contenedores = json.loads((dir_particion / "contenedores.json").read_text())

    def __len__(self):
        return len(self.contenedores)

    def __getitem__(self, i):
        filas = self.seg[self.contenedores[i]]
        return list(zip(filas["input_ids"], filas["labels"]))


class Colador:
    def __init__(self):
        self.aplanar = DataCollatorWithFlattening(
            return_position_ids=True, return_flash_attn_kwargs=True, return_seq_idx=True
        )

    def __call__(self, lote):
        if len(lote) != 1:
            raise ValueError("per_device_train_batch_size debe ser 1: cada elemento ya es un contenedor.")
        return self.aplanar([{"input_ids": x, "labels": y} for x, y in lote[0]])
