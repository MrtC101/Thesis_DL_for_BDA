![](images/UncuyoLogo.png)

# Deep Learning applied on Building Damage Assessment for satellite images after natural disasters

This is a thesis project for a degree in computer science from "Universidad Nacional de Cuyo facultad de ingeniería" related to **Deep Learning** applied on **Building Damage Assessment** for VHR satellite images taken after a natural disaster from the dataset xBD from the competition xView2. [Preview](http://mrtc101.github.io/thesis_dl_for_bda)

This project is based on a Microsoft Siames Convolution Neural Network from <a target="_blank" href="https://github.com/microsoft/building-damage-assessment-cnn-siamese">this repository</a>. It involves a reimplementation and introduction of new features and modification to the training pipeline.

This project have two main branches
- master: Contains a re-implemented version of the training pipeline from main branch of Microsoft repository  
- web_page: Contains a webpage to showcase the inference capabilities of the trained model.

**Jump to:**
1. [Master branch folder structure](#master-branch-folder-structure)
1. [Data sources](#data-sources)
1. [Data processing](#data-processing)
1. [Data splits & augmentation](#data-splits-&-augmentation)
1. [Overview of the model](#overview-of-the-model)
1. [Running experiments](#running-experiments)
1. [Results](#results)
1. [Setup](#setup)


## Master branch folder structure

    ├── README.md  <- The top-level README for developers using this project.
    ├── LICENSE
    ├── environment.yml
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │   └── figures        <- Generated graphics and figures to be used in reporting
    ├── data
    │   ├── constants
    │   └── xBD
    |       └── raw            <- The original, immutable data dump.
    ├── out     <- The output files generated during training
    ├── images  <- Images used by README.md file
    ├── notebooks          <- Jupyter notebooks.
    ├── models             <- Pretrained and serialized model parameters
    ├── src                <- Source code for use in this project.
    │   ├── models              <- Scripts that defines the model architecture
    │   ├── preprocessing       <- Scripts for data preprocessing
    │   ├── train               <- Scripts for model training and evaluation
    │   ├── utils               <- Scripts that implements utilities for all the other packages
    │   └── train_pipeline.py   <- Starting point of model training pipeline project
    └── submit  <- Files used for training in cluster

## Data Sources

We used [xBD dataset](https://xview2.org/), a publicly available dataset, to train and evaluate our proposed network performance. Detailed information about this dataset is provided in ["xBD: A Dataset for Assessing Building Damage from Satellite Imagery"](https://arxiv.org/abs/1911.09296) by Ritwik Gupta et al.

## Data processing

For data preprocessing we followed next steps:
1. Specific disasters data selection  
    For this project did not used all the xBD dataset..
1. Creation of disaster target masks.  
    Because not all datasets have an image we did a step of mask generation using a modified version of <a target="_blank" href="https://github.com/DIUx-xView/xView2_baseline/blob/master/utils/mask_polygons.py">this script</a> from 
    <a target="_blank" href="https://github.com/DIUx-xView/xView2_baseline">xView baseline repository</a>.
1. Crooped all the images
1. Did some agumetnation
1. Balance the dataset
1. Create shards different from originals

## Overview of the model

The model proposed in the original repository shares some characteristics with ["An Attention-Based System for Damage Assessment Using Satellite Imagery"](https://arxiv.org/pdf/2004.06643v1.pdf) by Hanxiang Hao et al. However, they do not incorporate any attention mechanism in the network and used a fewer number of convolutional layers for the segmentation arm, which is a UNet approach. Details of our architecture are shown below:

![Network Architecture Schema](./images/my_model.png)

This model has next characteristics:
- 
- 
- 

## Running experiments

### Training

The training has some characteristics...

#### Running

The training process can be configured editing and running `training_pipeline.py`
```
python src/training_pipeline.py
```

The experiment's progress can be monitored via tensorboard.

```
tensorboard --host 0.0.0.0 --logdir ./outputs/experiment_name/logs/ --port 8009
```
## Evaluation metrics

For the evaluation step we used a few new things...

## Results

We show the results on validation and test sets of our splits along with some segmenation maps with damage level.
![Results]()

### Inference

Code for inference step can be found in code `inference.py` and have this characteristics.

We developed a web page for showcase of the inference capabilities of the trained model here (URL) o utilizar la imagen de docker para levantar la página web de manera local

#### Docker

(Dockerización de página web?)

## Setup

### Creating the conda environment

At the root directory of this repo, use environment.yml to create a conda virtual environment called `develop`:

```
conda env create --file environment.yml
```

## Author

- Martín Cogo Belver
- Tutor: Dra. Ana Carolina Olivera

## Acknowledgment
- Mi familia
- Toko
- Docentes
- jefa de carrera
- Jurado