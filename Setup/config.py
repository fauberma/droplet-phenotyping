__author__ = 'Florian Aubermann'
__email__ = 'florian.aubermann@mr.mpg.de'
__status__ = 'development'

import os
import yaml
from pathlib import Path

script_dir = Path(__file__).parents[1]
os.environ['SCRIPT_DIR'] = str(script_dir)

with open(os.path.join(os.getenv('SCRIPT_DIR'), 'Setup', 'config.yml'), 'r') as f:
    config = yaml.safe_load(f)

os.environ['ANALYSES_DIR'] = config['ANALYSES_DIR']
os.environ['DB_DIR'] = config['DB_DIR']
os.environ['MODEL_DIR'] = config['MODEL_DIR']

if __name__ == "__main__":
    pass
