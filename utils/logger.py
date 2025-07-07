# utils/logger.py

import os
import json
from abc import ABC, abstractmethod
from typing import Dict, Any
from pprint import pprint

class BaseLogger(ABC):
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Any], step: int):
        pass

    @abstractmethod
    def log_images(self, images: Dict[str, Any], step: int):
        pass

class WandbLogger(BaseLogger):
    def __init__(self, config):
        import wandb
        self.wandb = wandb
        self.wandb.init(
            entity=config["entity"],
            project=config["project"],
            name=config["name"],
            tags=config["tags"],
            group="debug",
            config=config,
        )

    def log_metrics(self, metrics, step):
        self.wandb.log({**metrics, "step": step})

    def log_images(self, images, step):
        log_images = {k: self.wandb.Image(v) for k, v in images.items()}
        self.wandb.log({**log_images, "step": step})

class TensorboardLogger(BaseLogger):
    def __init__(self, config):
        from torch.utils.tensorboard import SummaryWriter
        log_dir = os.path.join(config["log_dir"], "tb_logger")
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_metrics(self, metrics, step):
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, step)

    def log_images(self, images, step):
        for k, v in images.items():
            self.writer.add_image(k, v, step, dataformats='HWC')

class LocalLogger(BaseLogger):
    def __init__(self, config):
        log_dir = os.path.join(config["log_dir"], "local_logger")
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.metrics_file = os.path.join(log_dir, "metrics.jsonl")
        self.images_dir = os.path.join(log_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)

    def log_metrics(self, metrics, step):
        with open(self.metrics_file, "a") as f:
            metrics = {k: str(v) for k, v in metrics.items()}
            f.write(json.dumps({"step": step, **metrics}) + "\n")

    def log_images(self, images, step):
        for k, v in images.items():
            # v is expected to be a numpy array (HWC, uint8)
            from PIL import Image
            img = Image.fromarray(v)
            img.save(os.path.join(self.images_dir, f"{k}_step{step}.png"))

class ConsoleLogger(BaseLogger):
    def __init__(self, cfg):
        pprint(cfg, compact=True)

    def log_metrics(self, metrics, step):
        print(f"[Step {step}] Metrics: " + ", ".join(f"{k}: {v}" for k, v in metrics.items()))

    def log_images(self, images, step):
        print(f"[Step {step}] Images logged: {', '.join(images.keys())}")


class CompositeLogger(BaseLogger):
    def __init__(self, loggers, cfg):
        available_loggers = {"wandb": WandbLogger,
                            "tb": TensorboardLogger,
                            "json": TensorboardLogger,
                            "local": LocalLogger,
                            "console": ConsoleLogger}
        self.loggers = []
        for logger in loggers:
            assert logger in available_loggers.keys(), f"Logger \"{logger}\" not known. Available loggers are: {available_loggers.keys()}" 
            logger_class = available_loggers[logger]
            self.loggers.append(logger_class(cfg))


    def log_metrics(self, metrics, step):
        for logger in self.loggers:
            logger.log_metrics(metrics, step)

    def log_images(self, images, step):
        for logger in self.loggers:
            logger.log_images(images, step)

    def log_checkpoint(self, checkpoint, step):
        for logger in self.loggers:
            logger.log_checkpoint(checkpoint, step)