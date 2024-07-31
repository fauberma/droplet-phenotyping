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
conda create -n phenotyping python==3.12
conda activate phenotyping
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

   The enviornment variables are set to default.
   If the '/Experiment', '/Droplet_db' or 'CNN_models' directories should be moved to different locations (e.g. to an external drive) the corresponding paths need to be updated.
   


5. **Run the application**

  The pipeline can be executed step-by-step in the 'full_pipeline.ipynb' notebook.

## Usage

...

## Configuration
...

## Contact

Email: florian.aubermann@mr.mpg.de
GitHub Issues: https://github.com/fauberma/droplet-phenotyping/issues
