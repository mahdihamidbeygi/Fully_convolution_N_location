# Fully Convolutional Network for Earthquake Location

This repository contains a deep learning implementation for earthquake location using a Fully Convolutional Network (FCN). The model processes seismic waveform data to predict earthquake locations in 3D space.

## Project Overview

The project implements a deep learning approach to automate earthquake location using seismic waveform data. The FCN architecture is designed to process raw waveform data and output probability distributions for earthquake locations in a 3D grid.

## Features

- Fully Convolutional Network architecture for earthquake location
- Support for processing seismic waveform data
- 3D location prediction capabilities
- Integration with IRIS and ADRM data sources
- Training and prediction scripts

## Repository Structure

- `fcn_train.py`: Main training script for the FCN model
- `fcn_predict.py`: Script for making predictions with the trained model
- `getting_data_ADRM.py`: Data acquisition from ADRM
- `getting_data_IRIS.py`: Data acquisition from IRIS
- `sgydata.py`: Data processing utilities
- `iris_station.py`: IRIS station management
- `station`: Station configuration file
- Training and testing data files:
  - `training_samples.txt`
  - `testing_samples.txt`
  - `training_eq.png`
  - `testing_eq.png`
  - `study_region.png`

## Requirements

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Additional dependencies for data acquisition and processing

## Usage

### Training the Model

```bash
python fcn_train.py
```

### Making Predictions

```bash
python fcn_predict.py
```

## Data Sources

The project supports data acquisition from:
- IRIS (Incorporated Research Institutions for Seismology)
- ADRM (Additional Data Repository)

## Model Architecture

The FCN architecture consists of:
- Multiple convolutional layers with ReLU activation
- Max pooling layers for downsampling
- Up-sampling layers for reconstruction
- Dropout layers for regularization
- Final sigmoid activation for probability output

## License

[Add your license information here]

## Contact

[Add your contact information here]
