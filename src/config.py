"""读取配置；路径始终相对于项目根目录。"""
from configparser import ConfigParser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Path = PROJECT_ROOT / "config.ini") -> ConfigParser:
    config = ConfigParser(interpolation=None)
    if path == PROJECT_ROOT / "config.ini" and not path.exists():
        path = PROJECT_ROOT / "config.ini.example"
    with path.open(encoding="utf-8") as file:
        config.read_file(file)
    return config
