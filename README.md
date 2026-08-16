# corneal-crimp-analysis
Python image-processing pipeline for semi-automatic and automatic measurement of collagen fibre crimp in Forward SHG corneal images.

# Corneal Collagen Crimp Analysis

Python image-processing pipeline developed to analyse collagen fibre crimp in Forward Second Harmonic Generation (SHG) images of the cornea.

## What the code does

The analysis:

1. Loads a Forward SHG image from a CZI file.
2. Performs percentile normalisation.
3. Applies CLAHE contrast enhancement.
4. Applies Gaussian smoothing.
5. Estimates collagen fibre orientation using a structure tensor.
6. Extracts a selected analysis patch.
7. Enhances collagen ridges using a Sato filter.
8. Provides a semi-automatic single-fibre measurement.
9. Automatically traces fibres across the image patch.
10. Calculates fibre waviness.

Fibre waviness is calculated as:

Waviness = fibre path length / end-to-end distance

## Files

`crimp_analysis.py`  
Main collagen crimp analysis script.

`requirements.txt`  
Python packages required to run the analysis.

`data/`  
Folder where the microscopy CZI image should be placed.

## Installation

Install the required Python packages using:

pip install -r requirements.txt

## Running the analysis

Place the CZI file inside the `data` folder.

The default filename expected by the script is:

Rabbit_Central_cornea_low_pressure.czi

Run:

python crimp_analysis.py

## Image selection

The main image-selection parameters are located at the top of the script:

SCENE  
TIME  
CHANNEL  
Z_SLICE  
PATCH_SIZE  
CENTRE_X  
CENTRE_Y

These can be changed depending on the image and region being analysed.

## Semi-automatic measurement

During the semi-automatic stage, the selected image patch will appear.

Click:

1. The start of a collagen fibre.
2. The end of the same collagen fibre.

The program will calculate the most likely fibre path between these points and determine its waviness.

## Automatic measurement

The automatic stage detects ridge peaks across the image, links them into potential fibre paths, filters unsuitable tracks, and calculates waviness for accepted fibres.

The output includes:

- number of accepted fibres
- path length
- end-to-end distance
- waviness ratio
- mean waviness
- median waviness
- standard deviation
- minimum waviness
- maximum waviness

## Notes

Automatic fibre tracing is image-dependent and the resulting fibre tracks should be visually inspected.

Microscopy data are not stored in this repository.
