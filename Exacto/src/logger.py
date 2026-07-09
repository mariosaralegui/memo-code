"""
Guardado de Logs del Terminal
"""
import logging
import os
from datetime import datetime

def setup_logger(nombre_modulo: str, log_dir: str = "./outputs/logs/") -> logging.Logger:
    """
    Configura un logger que emite mensajes tanto por consola como a un archivo .log.
    """
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{nombre_modulo}_{timestamp}.log")

    logger = logging.getLogger(nombre_modulo)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger