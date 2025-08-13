# Microfluidics Droplet Prediction Tool

A user-friendly GUI application for predicting and designing microfluidic droplet parameters, built with Python and machine learning models. This tool supports both single and double emulsion droplet analysis, providing researchers with efficient parameter prediction and directional design capabilities.


## Project Overview

This application integrates machine learning models to offer comprehensive microfluidic droplet analysis, including:
- Prediction of droplet diameter (single/double emulsion)
- Prediction of droplet generation rate
- Directional design of target droplets based on input parameters

The intuitive graphical interface allows researchers to input experimental conditions and obtain real-time predictions, streamlining the microfluidic experiment design process.


## Key Features

1. **Single Emulsion Analysis**
   - Prediction of single emulsion droplet diameter
   - Prediction of single emulsion droplet generation rate
   - Directional design of target single emulsion droplets

2. **Double Emulsion Analysis**
   - Prediction of outer diameter of double emulsion droplets
   - Prediction of inner diameter of double emulsion droplets
   - Prediction of double emulsion droplet generation rate
   - Directional design of target double emulsion droplets


## Software Requirements

- **Operating System**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **Python Version**: 3.7-3.9 (compatible with TensorFlow)
- **Required Libraries**:
  - `tkinter` (for GUI)
  - `tensorflow==2.10.0`
  - `numpy>=1.19.5`
  - `pandas>=1.3.5`
  - `keras==2.10.0`
  - `Pillow>=9.0.0`


## Hardware Requirements

- **Minimum Configuration**:
  - CPU: Intel Core i5 or equivalent
  - RAM: 8GB
  - Storage: 1GB free space (for application and model files)
  
- **Recommended Configuration**:
  - CPU: Intel Core i7 or equivalent
  - RAM: 16GB
  - GPU: NVIDIA GTX 1050 Ti or better (for accelerated model inference)
  - Storage: 2GB free space


## Installation Guide

1. Clone this repository:
   ```bash
   git clone https://github.com/LH666-hash/Microfludics-droplet.git
   cd Microfludics-droplet
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Create a `requirements.txt` file with the libraries listed in "Software Requirements")*

3. Prepare model files:
   The model can be trained by using the model code we provide,and after training, you just need to replace the location of our MICA model.


## Usage Instructions

1. Run the application:
   ```bash
   python main.py  # Replace with your actual script filename
   ```

2. Select a function from the dropdown menu (e.g., "Prediction of single emulsion droplet diameter")

3. Input parameters according to the specified ranges (e.g., 50μm < L1 < 250μm)

4. Click "OK" to generate predictions

5. View results in the pop-up window, including units (μm, Hz, μL/min, etc.)


## Notes

- All input parameters must be within the specified ranges to ensure valid predictions
- Model performance depends on the training data; results should be validated with actual experiments
- For best performance, close other memory-intensive applications when running the tool
- The GUI supports multi-window operation, with parameter input windows and result windows opening separately



## Contact

For questions or issues, please contact the project maintainer at [18660636636@dlut.mail.edu.cn]
