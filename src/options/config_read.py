import yaml
import sys
import os.path


# Функция для чтения конфигурационного файла
def get_config_data():
    with open('C:\\Users\\Zak\\PycharmProjects\\auto_placement_accents\\src\\options\\config.yaml', 'r') as file_option:
        options = yaml.safe_load(file_option)
    return options
