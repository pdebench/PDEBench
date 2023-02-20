import os
from os.path import dirname, abspath
import logging
# Defaults
ROOT_DIR   = dirname(dirname(abspath(__file__))) # Project Root
HOME_DIR   = os.getenv("HOME") # User home dir
DATA_DIR   = os.path.join(ROOT_DIR, 'data')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'out')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
from .utils import launch_logger
