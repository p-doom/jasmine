import os
import json
from abc import ABC, abstractmethod
from typing import Dict, Any
from pprint import pprint
import numpy as np
import jax

class BaseLogger(ABC):
    """
    Abstract base class for all loggers.

    Defines the interface for logging metrics and images.
    """

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """
        Log metrics at a given step.

        Args:
            metrics (Dict[str, Any]): Dictionary of metric names and values.
            step (int): The current step or epoch.
        """
        pass

    @abstractmethod
    def log_images(self, images: Dict[str, Any], step: int):
        """
        Log images at a given step.

        Args:
            images (Dict[str, Any]): Dictionary of image names and image data.
            step (int): The current step or epoch.
        """
        pass

class WandbLogger(BaseLogger):
    """
    Logger for Weights & Biases (wandb) integration.

    Logs metrics and images to the wandb dashboard.
    """

    def __init__(self, config):
        """
        Initialize the WandbLogger.

        Args:
            config (dict): Configuration dictionary containing wandb parameters.
        """
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
        """
        Log metrics to wandb.

        Args:
            metrics (dict): Dictionary of metric names and values.
            step (int): The current step or epoch.
        """
        self.wandb.log({**metrics, "step": step})

    def log_images(self, images, step):
        """
        Log images to wandb.

        Args:
            images (dict): Dictionary of image names and image data.
            step (int): The current step or epoch.
        """
        log_images = {k: self.wandb.Image(v) for k, v in images.items()}
        self.wandb.log({**log_images, "step": step})

class TensorboardLogger(BaseLogger):
    """
    Logger for TensorBoard integration.

    Logs metrics and images to TensorBoard.
    """

    def __init__(self, config):
        """
        Initialize the TensorboardLogger.

        Args:
            config (dict): Configuration dictionary containing log directory and experiment name.
        """
        from tensorboardX import SummaryWriter
        base_log_dir = os.path.join(config["log_dir"], "tb_logger", config["name"])
        log_dir = base_log_dir
        idx = 1
        while os.path.exists(log_dir):
            log_dir = f"{base_log_dir}-{idx}"
            idx += 1
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_metrics(self, metrics, step):
        """
        Log metrics to TensorBoard.

        Args:
            metrics (dict): Dictionary of metric names and values.
            step (int): The current step or epoch.
        """
        for k, v in metrics.items():
            self.writer.add_scalar(f"metrics/{k}", v, step)

    def log_images(self, images, step):
        """
        Log images to TensorBoard.

        Args:
            images (dict): Dictionary of image names and image data.
            step (int): The current step or epoch.
        """
        for k, v in images.items():
            self.writer.add_image(f"media/{k}", v, step, dataformats='HWC')

class LocalLogger(BaseLogger):
    """
    Logger for local filesystem logging.

    Logs metrics to a JSONL file and images as PNGs in a directory.
    """

    def __init__(self, config):
        """
        Initialize the LocalLogger.

        Args:
            config (dict): Configuration dictionary containing log directory and experiment name.
        """
        base_log_dir = os.path.join(config["log_dir"], "local_logger", config["name"])
        log_dir = base_log_dir
        idx = 1
        while os.path.exists(log_dir):
            log_dir = f"{base_log_dir}-{idx}"
            idx += 1
        os.makedirs(log_dir, exist_ok=True)
        self.metrics_file = os.path.join(log_dir, "metrics.jsonl")
        self.images_dir = os.path.join(log_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)

    def log_metrics(self, metrics, step):
        """
        Log metrics to a local JSONL file.

        Args:
            metrics (dict): Dictionary of metric names and values.
            step (int): The current step or epoch.
        """
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps({"step": step, **metrics}) + "\n")

    def log_images(self, images, step):
        """
        Log images as PNG files to the local filesystem.

        Args:
            images (dict): Dictionary of image names and image data (numpy arrays).
            step (int): The current step or epoch.
        """
        for k, v in images.items():
            # v is expected to be a numpy array (HWC, uint8)
            from PIL import Image
            img = Image.fromarray(v)
            img.save(os.path.join(self.images_dir, f"{k}_step{step}.png"))

class ConsoleLogger(BaseLogger):
    """
    Logger for console output.

    Prints metrics and image logging information to the console.
    """

    def __init__(self, cfg):
        """
        Initialize the ConsoleLogger.

        Args:
            cfg (dict): Configuration dictionary to print at initialization.
        """
        pprint(cfg, compact=True)

    def log_metrics(self, metrics, step):
        """
        Print metrics to the console.

        Args:
            metrics (dict): Dictionary of metric names and values.
            step (int): The current step or epoch.
        """
        print(f"[Step {step}] Metrics: " + ", ".join(f"{k}: {v}" for k, v in metrics.items()))

    def log_images(self, images, step):
        """
        Print image logging information to the console.

        Args:
            images (dict): Dictionary of image names and image data.
            step (int): The current step or epoch.
        """
        print(f"[Step {step}] Images logged: {', '.join(images.keys())}")


class CompositeLogger(BaseLogger):
    """
    Logger that combines multiple logger backends.

    Forwards logging calls to all specified loggers.
    """

    def __init__(self, loggers, cfg):
        """
        Initialize the CompositeLogger.

        Args:
            loggers (list): List of logger names to instantiate.
            cfg (dict): Configuration dictionary to pass to each logger.
        """
        available_loggers = {"wandb": WandbLogger,
                            "tb": TensorboardLogger,
                            "local": LocalLogger,
                            "console": ConsoleLogger}
        self.loggers = []
        for logger in loggers:
            assert logger in available_loggers.keys(), f"Logger \"{logger}\" not known. Available loggers are: {available_loggers.keys()}" 
            logger_class = available_loggers[logger]
            self.loggers.append(logger_class(cfg))


    def log_metrics(self, metrics, step):
        """
        Log metrics to all contained loggers.

        Args:
            metrics (dict): Dictionary of metric names and values.
            step (int): The current step or epoch.
        """
        metrics = jax.tree.map(
            lambda x: x.item() if isinstance(x, (jax.Array, np.ndarray)) else x, metrics
        )
        for logger in self.loggers:
            logger.log_metrics(metrics, step)

    def log_images(self, images, step):
        """
        Log images to all contained loggers.

        Args:
            images (dict): Dictionary of image names and image data.
            step (int): The current step or epoch.
        """
        for logger in self.loggers:
            logger.log_images(images, step)
