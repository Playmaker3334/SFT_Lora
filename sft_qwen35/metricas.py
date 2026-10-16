import json
import os
import platform
import subprocess
import threading
import time
from importlib import metadata
from pathlib import Path

import psutil
import pynvml
import torch
from datasets import load_from_disk

PAQUETES = (
    "torch",
    "transformers",
    "peft",
    "kernels",
    "liger-kernel",
    "flash-linear-attention",
    "causal-conv1d",
    "bitsandbytes",
    "datasets",
)


def rango():
    return int(os.environ.get("RANK", 0))


def _serializable(x):
    return x.item() if hasattr(x, "item") else str(x)


def _nvml(f, *args):
    try:
        return f(*args)
    except pynvml.NVMLError:
        return None


def _version(paquete):
    try:
        return metadata.version(paquete)
    except metadata.PackageNotFoundError:
        return None


def _git():
    raiz = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=raiz, capture_output=True, text=True, check=True)
        cambios = subprocess.run(["git", "status", "--porcelain"], cwd=raiz, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return {"commit": commit.stdout.strip(), "cambios_sin_commit": bool(cambios.stdout.strip())}


def entorno():
    gpus = []
    for i in range(pynvml.nvmlDeviceGetCount()):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        limite = _nvml(pynvml.nvmlDeviceGetEnforcedPowerLimit, h)
        gpus.append(
            {
                "indice": i,
                "nombre": pynvml.nvmlDeviceGetName(h),
                "vram_mib": pynvml.nvmlDeviceGetMemoryInfo(h).total / 2**20,
                "limite_w": limite / 1000 if limite is not None else None,
            }
        )
    return {
        "python": platform.python_version(),
        "paquetes": {p: _version(p) for p in PAQUETES},
        "cuda": torch.version.cuda,
        "driver": pynvml.nvmlSystemGetDriverVersion(),
        "gpus": gpus,
        "procesos": int(os.environ.get("WORLD_SIZE", 1)),
        "cpus": os.cpu_count(),
        "git": _git(),
    }


def datos_corrida(cfg_datos):
    raiz = Path(cfg_datos["salida"])
    particiones = {}
    for part in ("train", "val"):
        seg = load_from_disk(str(raiz / part / "segmentos")).select_columns(["n", "n_prompt", "fuente", "tipo_tarea"])
        columnas = seg.to_dict()
        contenedores = json.loads((raiz / part / "contenedores.json").read_text())
        columnas["contenedores"] = [sum(columnas["n"][i] for i in c) for c in contenedores]
        particiones[part] = columnas
    descartados = raiz / "descartados.json"
    return {
        "seq_len": cfg_datos["seq_len"],
        "particiones": particiones,
        "descartados": json.loads(descartados.read_text()) if descartados.exists() else None,
    }


def picos_vram():
    dispositivo = int(os.environ.get("LOCAL_RANK", 0))
    local = {
        "rango": rango(),
        "asignada_mib": torch.cuda.max_memory_allocated(dispositivo) / 2**20,
        "reservada_mib": torch.cuda.max_memory_reserved(dispositivo) / 2**20,
    }
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        todos = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(todos, local)
        return todos
    return [local]


def ram_contenedor():
    try:
        stat = dict(linea.split() for linea in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines())
        return int(stat["anon"]) / 2**20
    except (OSError, KeyError, ValueError):
        return None


class Registro:
    def __init__(self, salida):
        base = Path(salida) / "metricas"
        self.dir, k = base, 1
        while (self.dir / "corrida.json").exists():
            k += 1
            self.dir = base.with_name(f"metricas-{k}")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.corrida = {}

    def anexar(self, archivo, fila):
        with open(self.dir / archivo, "a", encoding="utf-8") as f:
            f.write(json.dumps(fila, ensure_ascii=False, default=_serializable) + "\n")

    def escribir(self, archivo, contenido):
        texto = json.dumps(contenido, ensure_ascii=False, indent=2, default=_serializable)
        (self.dir / archivo).write_text(texto, encoding="utf-8")

    def guardar(self):
        self.escribir("corrida.json", self.corrida)

    def marcar(self, momento):
        self.corrida["tiempos"][momento] = time.time()
        self.guardar()


class Monitor(threading.Thread):
    def __init__(self, registro, intervalo_s, latido_s, latido):
        super().__init__(daemon=True)
        self.registro = registro
        self.intervalo_s = intervalo_s
        self.latido_s = latido_s
        self.latido = latido
        self.alto = threading.Event()
        self.manijas = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        self.arbol = psutil.Process(os.getppid()) if "LOCAL_RANK" in os.environ else psutil.Process()

    def ram_procesos(self):
        try:
            procesos = [self.arbol, *self.arbol.children(recursive=True)]
        except psutil.Error:
            return None
        total = 0
        for p in procesos:
            try:
                total += p.memory_info().rss
            except psutil.Error:
                pass
        return total / 2**20

    def muestra(self):
        gpus = []
        for i, h in enumerate(self.manijas):
            w = _nvml(pynvml.nvmlDeviceGetPowerUsage, h)
            uso = _nvml(pynvml.nvmlDeviceGetUtilizationRates, h)
            memoria = _nvml(pynvml.nvmlDeviceGetMemoryInfo, h)
            gpus.append(
                {
                    "i": i,
                    "w": w / 1000 if w is not None else None,
                    "energia_mj": _nvml(pynvml.nvmlDeviceGetTotalEnergyConsumption, h),
                    "uso": uso.gpu if uso is not None else None,
                    "vram_mib": memoria.used / 2**20 if memoria is not None else None,
                    "temp": _nvml(pynvml.nvmlDeviceGetTemperature, h, pynvml.NVML_TEMPERATURE_GPU),
                }
            )
        self.registro.anexar(
            "gpu.jsonl",
            {"t": time.time(), "gpus": gpus, "ram_contenedor_mib": ram_contenedor(), "ram_procesos_mib": self.ram_procesos()},
        )

    def iniciar(self):
        self.muestra()
        self.start()

    def run(self):
        proximo = time.time() + self.latido_s
        while not self.alto.wait(self.intervalo_s):
            try:
                self.muestra()
            except Exception as e:
                print(f"monitor: {e}")
            if self.latido_s and time.time() >= proximo:
                proximo += self.latido_s
                try:
                    self.latido()
                except Exception as e:
                    print(f"latido: {e}")

    def detener(self):
        self.alto.set()
        self.join()
        self.muestra()
