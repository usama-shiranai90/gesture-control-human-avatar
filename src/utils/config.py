"""
Configuration loader utility.

Loads YAML configuration files using OmegaConf and provides
a unified config object for the entire application.
"""

from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, OmegaConf
from loguru import logger


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"


def load_config(config_name: str, config_dir: Optional[Path] = None) -> DictConfig:
    """
    Load a YAML configuration file.

    Args:
        config_name: Name of the config file (with or without .yaml extension).
        config_dir: Directory containing configs. Defaults to project configs/.

    Returns:
        DictConfig object with the loaded configuration.
    """
    if config_dir is None:
        config_dir = CONFIGS_DIR

    if not config_name.endswith(".yaml"):
        config_name = f"{config_name}.yaml"

    config_path = config_dir / config_name

    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}. Using empty config.")
        return OmegaConf.create({})

    cfg = OmegaConf.load(config_path)
    logger.debug(f"Loaded config from {config_path}")
    return cfg


def load_all_configs() -> DictConfig:
    """
    Load and merge all configuration files from the configs directory.

    Returns:
        Merged DictConfig containing all configurations.
    """
    configs = {}
    config_files = ["camera", "gesture", "segmentation", "model"]

    for name in config_files:
        cfg = load_config(name)
        configs.update(OmegaConf.to_container(cfg, resolve=True))

    merged = OmegaConf.create(configs)
    logger.info("All configurations loaded and merged.")
    return merged


def get_output_dir(subdir: str = "") -> Path:
    """Get an output directory path, creating it if needed."""
    output_path = PROJECT_ROOT / "outputs" / subdir
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def get_data_dir(subdir: str = "") -> Path:
    """Get a data directory path, creating it if needed."""
    data_path = PROJECT_ROOT / "data" / subdir
    data_path.mkdir(parents=True, exist_ok=True)
    return data_path
