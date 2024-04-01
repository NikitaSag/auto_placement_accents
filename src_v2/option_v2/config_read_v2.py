import yaml
import sys
import os.path


# Функция для чтения конфигурационного файла
def get_config_data():
    with open('C:\\Users\\Zak\\PycharmProjects\\auto_placement_accents\\src_v2\\option_v2\\config_v2.yaml',
              'r') as file_option:
        options = yaml.safe_load(file_option)
    return options
