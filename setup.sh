#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Update package lists and install system dependencies
echo "Updating package lists..."
sudo apt update

echo "Installing system dependencies..."
sudo apt install -y build-essential libssl-dev libffi-dev python3-dev \
                    libatlas-base-dev gfortran libhdf5-dev libc-ares-dev libeigen3-dev \
                    python3-pip python3-venv

# Clone the repository (only if not already done)
# Uncomment and modify the following lines if you haven't cloned the repo yet
# echo "Cloning the repository..."
# git clone <your-repo-url>
# cd your-project-directory

# Create a virtual environment
# echo "Creating virtual environment..."
# python3 -m venv venv

# Activate the virtual environment
# echo "Activating virtual environment..."
# source venv/bin/activate

# Install Python dependencies using requirements.txt
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Setup complete. Your environment is ready."
