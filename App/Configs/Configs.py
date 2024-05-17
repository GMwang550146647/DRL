import os

CODE_PATH = os.path.dirname(os.path.dirname(__file__))
PROJECT_PATH = os.path.dirname(CODE_PATH)
DATA_PATH = os.path.join(PROJECT_PATH, 'Input')
OUTPUT_PATH = os.path.join(PROJECT_PATH, 'Output')
PACKAGE_PATH = os.path.join(PROJECT_PATH, "App", 'Derivative')
os.makedirs(DATA_PATH,exist_ok=True)
os.makedirs(OUTPUT_PATH,exist_ok=True)
os.makedirs(PACKAGE_PATH,exist_ok=True)
RAY_SERVER_IP_PORT = "127.0.0.1:6379"

