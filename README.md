# droplet-phenotyping
This project was developed to analyze microscopy data of immune cell- target cell interactions confined in mcirofluidic droplets.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contact](#contact)

## Installation

### Prerequisites

It is recommended to setup a new environment before starting installing dependecies.
```bash
conda create -n phenotyping python==3.9
```

### Getting Started

1. **Clone the repository**

    ```bash
    git clone https://github.com/fauberma/droplet-phenotyping.git
    cd droplet-phenotyping
    ```

2. **Install dependencies**

    The dependencies can be installed via pip. Depending on the platform being used, tensorflow can require additional steps for installation.

    ```bash
    pip install -r requirements.txt
    ```

3. **Setup environment variables**

   The pipeline requires the setup of envionment variables. They can be edited in the '.env' file.
   The .env file defines 3 directories that are required to store results and data during analysis.

   All results as well as experiment specific files will be found in ANALYSES_DIR, where each experiment has an experiment ID (expID)
   The droplet database is stored in DB_DIR, since the databases can grow large with multiple experiments, this could be a location on an external drive.
   Trained CNN models will be stored in MODEL_DIR.


5. **Run the application**

  The pipeline can be executed step-by-step in the 'full_pipeline.ipynb' notebook.

## Usage

...

## Configuration
...

## Contact

Email: florian.aubermann@mr.mpg.de
GitHub Issues: https://github.com/fauberma/droplet-phenotyping/issues
