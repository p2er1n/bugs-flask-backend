from .run_detect import run_detect
from .run_obb import run_obb

def run_task(task, image, engine, model):
    if task == "detect":
        return run_detect(image, engine, model)
    elif task == "obb":
        return run_obb(image, engine, model)
