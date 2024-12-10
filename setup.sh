#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Step 1: Check if Python 3.11 is installed
if ! command -v python3.11 &> /dev/null
then
    echo "Error: Python 3.11 is not installed. Please install it before running this script."
    exit 1
fi

# Step 2: Delete existing virtual environment if it exists
if [ -d "knn_env" ]; then
    echo "Existing virtual environment 'knn_env' found."
    read -p "Do you want to delete and recreate it? (y/n): " choice
    if [[ "$choice" == "y" || "$choice" == "Y" ]]; then
        echo "Deleting existing virtual environment..."
        rm -rf knn_env
    else
        echo "Using the existing virtual environment."
        if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
            echo "Activate it using: knn_env\\Scripts\\activate.bat"
        else
            echo "Activate it using: source knn_env/bin/activate"
        fi
        exit 0
    fi
fi

# Step 3: Create a new virtual environment
echo "Creating a new virtual environment named 'knn_env'..."
python3.11 -m venv knn_env

# Step 4: Activate the virtual environment
echo "Activating the virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    knn_env\\Scripts\\activate.bat
else
    # Mac/Linux
    source knn_env/bin/activate
fi

# Step 5: Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Step 6: Install dependencies
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "Error: requirements.txt not found. Please ensure it is included in the project folder."
    deactivate
    exit 1
fi

# Step 7: Notify the user
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "Setup is complete. Activate the virtual environment using: knn_env\\Scripts\\activate.bat"
else
    echo "Setup is complete. Activate the virtual environment using: source knn_env/bin/activate"
fi
echo "Run your project script using: python your_script.py"
