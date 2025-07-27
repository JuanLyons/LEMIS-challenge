import time
import os

# Ruta del archivo de log a monitorear
LOG_FILE_PATH = "/home/jclyons/endovis/miccai2025/LEMIS/region_proposals/outputs/LEMIS/log.txt"  # ← cámbiala a tu ruta

# Línea clave que activa la ejecución
TRIGGER_LINE = "fvcore.common.checkpoint INFO: Saving checkpoint to"


# Función que se ejecuta cuando se detecta la línea
def ejecutar_accion(model_weights):
    name = model_weights.split(os.sep)[-1].split(".")[0]
    print("¡Línea detectada! Ejecutando código...")
    os.system(
        f"python train_net.py --num-gpus 1 --eval-only --config-file configs/led/LED_SwinL_train.yaml DATASETS.DATA_PATH data OUTPUT_DIR outputs/LEMIS/test/{name} MODEL.WEIGHTS {model_weights} MODEL.TEXT Dataset_Embeddings"
    )
    print("Código ejecutado.")


def extraer_ruta(linea):
    if "to " in linea:
        return linea.split("to ", 1)[1].strip()
    return None


def monitorear_archivo():
    lineas_vistas = set()
    num = 1
    while True:
        try:
            with open(LOG_FILE_PATH, "r") as f:
                for linea in f:
                    if (
                        TRIGGER_LINE in linea
                        and linea not in lineas_vistas
                        and num % 2 == 0
                        and num >= 16
                    ):
                        model_weights = extraer_ruta(linea)
                        ejecutar_accion(model_weights)
                        lineas_vistas.add(linea)
                        num += 1
                    elif TRIGGER_LINE in linea and linea not in lineas_vistas:
                        lineas_vistas.add(linea)
                        num += 1

        except FileNotFoundError:
            print("Archivo no encontrado. Esperando que aparezca...")

        time.sleep(0.5)


if __name__ == "__main__":
    monitorear_archivo()
