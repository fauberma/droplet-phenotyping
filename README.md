# droplet-phenotyping
This project was developed to analyze microscopy data of immune cell- target cell interactions confined in mcirofluidic droplets.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contribution](#contribution)
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

   The enviornment variables are set to default in the `.env` file.
   The directories `/Experiments`, `/Droplet_db` or `CNN_models` can be moved to different locations (e.g. external drives).
   The new location of the directories must be updated in the `.env`.

5. **Run the application**

   A step-by-step guide for the full execution of the pipeline is demonstrated `full_pipeline.ipynb` notebook.

## Usage

1. **Perform your experiment**
   You just co-encapsulated your immune and target cells into droplets and observed their interactions via time-lapse microscopy, great!
   Hopefully, this experiment will be one of many experiments that you will cconduct. To not get lost with all the different experiments, let's assign systematic IDs to the experiemnts.
   An experiment ID serves to uniquely identify the data from a single experiment during analysis, so they should follow a regular expression that can be parsed during pipeline execution.
   I chose to assign experiment IDs according to a general structure of `AAAA_AA_000`, their cognate regex pattern is defined in the `.env`file.
   Now, create a new directory `[your_expID]` in the `/Experiments`directory.
   
2. **The `setup.xlsx` file** 
   Every experiment is different. This complicates the execution of the pipeline without user intervention.
   Therefore, before starting the analysis, it is necessary to generate a `setup.xlsx` file in the experiment directory. An example of a setup file is shown in the `NKIP_FA_058` experiment.
   This file holds all information on the experiment's structure and metadata.
   Most importantly, the location and structure of the raw image data is supplied with the file. In addition, it holds information about the different channels that were used during image acquisition.
   

## Contribution
So far, only .lif files from Leica microscopes are supported. To enable compatibility for other microscopy platforms, the `RawLoader` class can be adapted and included.
If you want to contribute, reach out to me.

## Contact

Email: florian.aubermann@mr.mpg.de
GitHub Issues: https://github.com/fauberma/droplet-phenotyping/issues
