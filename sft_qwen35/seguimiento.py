import math
import signal
import time
import traceback
from pathlib import Path

import psutil
import pynvml
from transformers import TrainerCallback

from . import graficas
from .avisos import Latido, Slack, texto_arranque, texto_resumen
from .metricas import Monitor, Registro, datos_corrida, entorno, picos_vram, rango


class Seguimiento:
    def __init__(self, cfg, salida, reanudar=None):
        self.activo = rango() == 0
        self.no_finita = False
        self.estado = {}
        self.registro = None
        self.monitor = None
        self.callback = Metricas(self)
        if not self.activo:
            return
        pynvml.nvmlInit()
        monitoreo = cfg.get("monitoreo") or {}
        self.registro = Registro(salida)
        c = self.registro.corrida
        c.update(
            nombre=Path(salida).name,
            salida=str(Path(salida).resolve()),
            estado="en_curso",
            reanudado_desde=reanudar,
            tiempos={"proceso": psutil.Process().create_time(), "inicio": time.time()},
            entorno=entorno(),
            monitoreo=monitoreo,
            config=cfg,
        )
        datos = datos_corrida(cfg["datos"])
        self.registro.escribir("datos.json", datos)
        self.slack = Slack()
        self.slack.mensaje(texto_arranque(c, graficas.resumen_datos(datos)))
        c["slack_hilo"] = self.slack.hilo
        c["slack_canal"] = self.slack.canal
        self.registro.guardar()
        self.monitor = Monitor(
            self.registro,
            monitoreo.get("intervalo_s") or 5,
            (monitoreo.get("latido_min") or 0) * 60,
            Latido(self.slack, c["nombre"], self.estado),
        )
        self.monitor.iniciar()

    def modelo_cargado(self, modelo):
        if not self.activo:
            return
        base = [(n, p.numel()) for n, p in modelo.named_parameters() if "lora_" not in n]
        self.registro.corrida["modelo"] = {
            "parametros": sum(p.numel() for p in modelo.parameters()),
            "entrenables": sum(p.numel() for p in modelo.parameters() if p.requires_grad),
            "parametros_flops": sum(k for n, k in base if "embed_tokens" not in n),
        }
        self.registro.marcar("modelo_cargado")

    def anotar(self, clave, valor):
        if self.activo:
            self.registro.corrida[clave] = valor
            self.registro.guardar()

    def cerrar(self, error=None):
        if not self.activo:
            return
        c = self.registro.corrida
        if error is None:
            c["estado"] = "detenido_loss_no_finita" if self.no_finita else "terminado"
        else:
            sigterm = isinstance(error, SystemExit) and error.code == 128 + signal.SIGTERM
            c["estado"] = "interrumpido" if isinstance(error, KeyboardInterrupt) or sigterm else "fallo"
            c["error"] = {
                "tipo": type(error).__name__,
                "mensaje": str(error),
                "traza": "".join(traceback.format_exception(error))[-4000:],
            }
        try:
            self.monitor.detener()
        except Exception as e:
            print(f"monitor: {e}")
        self.registro.marcar("fin")
        try:
            c["resumen"] = graficas.resumen(graficas.leer(self.registro.dir))
            self.registro.guardar()
            fila = {k: c.get(k) for k in ("nombre", "salida", "estado", "tiempos", "entorno", "config", "resumen")}
            fila["metricas"] = str(self.registro.dir.resolve())
            self.registro.anexar(Path(c["salida"]).parent / "corridas.jsonl", fila)
            self.slack.mensaje(texto_resumen(c))
        except Exception:
            traceback.print_exc()
            self.slack.mensaje(f"*{c['nombre']}* {c['estado']}; no se pudo armar el resumen")
        try:
            self.slack.imagenes(graficas.generar(self.registro.dir), "Gráficas de la corrida")
        except Exception as e:
            print(f"graficas: {e}")
            self.slack.mensaje(f"No se pudieron generar las gráficas: {e}")


class Metricas(TrainerCallback):
    def __init__(self, seguimiento):
        self.s = seguimiento
        self.t_paso = None
        self.t_eval = None
        self.paso_eval = None

    def on_train_begin(self, args, state, control, **kwargs):
        if self.s.activo:
            self.s.estado["max_pasos"] = state.max_steps
            self.s.registro.corrida["pasos_totales"] = state.max_steps
            self.s.registro.marcar("entrenamiento_inicio")

    def on_step_end(self, args, state, control, **kwargs):
        self.t_paso = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        if "loss" in logs and not math.isfinite(logs["loss"]):
            self.s.no_finita = True
            control.should_training_stop = True
        if self.s.activo:
            ahora = time.time()
            self.s.registro.anexar("pasos.jsonl", {"t": ahora, "step": state.global_step, **logs})
            if "loss" in logs:
                self.s.estado.update(
                    paso=state.global_step, loss=logs["loss"], tokens=logs.get("num_input_tokens_seen"), t=ahora
                )

    def on_evaluate(self, args, state, control, **kwargs):
        self.t_eval = time.time()
        self.paso_eval = state.global_step

    def on_save(self, args, state, control, **kwargs):
        if self.s.activo:
            inicio = self.t_eval if self.paso_eval == state.global_step else self.t_paso
            self.s.registro.anexar(
                "eventos.jsonl", {"evento": "guardado", "step": state.global_step, "inicio": inicio, "fin": time.time()}
            )

    def on_train_end(self, args, state, control, **kwargs):
        vram = picos_vram()
        if self.s.activo:
            self.s.registro.corrida.update(
                vram=vram,
                pasos=state.global_step,
                mejor_checkpoint=state.best_model_checkpoint,
                mejor_eval_loss=state.best_metric,
            )
            self.s.registro.marcar("entrenamiento_fin")
