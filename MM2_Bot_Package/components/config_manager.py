import json
import sys
from typing import Dict
from colorama import Fore, Style

def load_config(file_path: str) -> Dict:
    """
    Loads a JSON configuration file.

    Args:
        file_path: The path to the configuration file.

    Returns:
        A dictionary containing the configuration.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Configuration file not found: {file_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Invalid JSON in configuration file: {file_path}")
        sys.exit(1)
