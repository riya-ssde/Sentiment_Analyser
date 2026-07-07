from configparser import ConfigParser
from utils.logger import logger

config = ConfigParser()

config_path = "config/config.ini"
config.read(config_path)