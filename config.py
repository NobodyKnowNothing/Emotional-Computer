import json

def get_config_context():
    with open('config.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
        return data
    return {}