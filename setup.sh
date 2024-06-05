#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Variables
MINICONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
MINICONDA_URL="https://repo.anaconda.com/miniconda/$MINICONDA_INSTALLER"
ENV_NAME="myenv"
PYTHON_VERSION="3.11"

# Download Miniconda installer
echo "Downloading Miniconda installer..."
wget $MINICONDA_URL -O $MINICONDA_INSTALLER

# Install Miniconda
echo "Installing Miniconda..."
bash $MINICONDA_INSTALLER -b -p $HOME/miniconda

# Initialize Conda
echo "Initializing Conda..."
export PATH="$HOME/miniconda/bin:$PATH"
source $HOME/miniconda/etc/profile.d/conda.sh
conda init

# Check if conda is available
if ! command -v conda &> /dev/null
then
    echo "conda could not be found"
    exit 1
fi

# Create Conda environment
echo "Creating Conda environment..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# Activate the environment
echo "Activating the Conda environment..."
source activate $ENV_NAME

# Install packages
echo "Installing required packages..."
conda install -c conda-forge keras-tuner pandas-ta scikit-learn numpy pandas tensorflow matplotlib -y
pip install ccxt

# Cleanup
echo "Cleaning up..."
rm $MINICONDA_INSTALLER

echo "Setup completed successfully. To activate the environment, run 'conda activate $ENV_NAME'."
