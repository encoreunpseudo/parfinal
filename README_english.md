# Video Detection and Analysis System for Cognitive Development Study - Master 1 Project

This project offers a complete pipeline for the detection, classification, analysis, and visualization of objects present in videos of children interacting. It is based on a modular architecture using deep learning (fine-tuned YOLOv8) and includes spatial and temporal metrics useful for studying cognitive development.

## Main Features
* **Automatic detection** of 6 classes of objects in videos
* **Spatial and temporal analysis** of interactions
* **Interactive visualization** of results via Streamlit
* **Data export** in CSV format

## Project Structure

```
├── detector/
│   ├── classes.csv     # Class definitions and YOLO assignment
│   ├── denoise         # Video denoising
│   ├── main2.py        # Main processing pipeline
│   ├── run.py          # Launch script
│   ├── yolov8_model_new.pt  # Fine-tuned YOLOv8 model for SO detection
├── exemples/           # Visual results of processed videos
│   ├── output.mp4      # Example video
├── indicateurs/
│   ├── distance.py     # Calculation of distance between hands and small objects (SO)
│   ├── pos_moyenne.py  # Spatial averages of detected classes
│   └── tps_presence.py # Average time of presence per class
```

## Video Processing Protocol
The video analysis follows a protocol in **four successive stages**:

1. Preprocessing
   * Denoising of the visual signal to improve detection quality
   * Preparation of frames for analysis

2. Object Detection
   * Using a **fine-tuned YOLOv8 model** on our specific data
   * Detection of **6 operational classes**:
      * `visages`: faces of participants
      * `mains`: hands of participants
      * `A`: animals
      * `OAG`: large artificial objects
      * `ONG`: large natural objects
      * `OP`: small objects (merging natural/artificial)

3. Data Structuring
   * Results organized as **CSV tables** including:
      * Detected classes
      * Spatial coordinates
      * Timestamps
      * Confidence scores

4. Visualization and Analysis
   * Interactive interfaces to explore data
   * Advanced metrics on behaviors and interactions

## Provided Indicators

| Script | Function | Application |
|--------|----------|-------------|
| `distance.py` | Measures the distance between hands and small objects | Analysis of child-object interactions |
| `pos_moyenne.py` | Calculates the average position of classes in space | Mapping of the interaction space |
| `tps_presence.py` | Estimates the average time of presence for each class | Analysis of attention and engagement |

## User Guide

1. Launch the main processing

```
python detector/run.py
```

2. Run analyses via Streamlit

```
# Distance calculation
streamlit run indicateurs/distance.py

# Average position
streamlit run indicateurs/pos_moyenne.py

# Presence time
streamlit run indicateurs/tps_presence.py
```

## Input and Output Data
* **Input**: Videos to be analyzed should be placed in the `exemples/` folder in `.mp4` format
* **Output**:
   * Annotated frames (available in `exemples/`)
   * CSV files of metrics
   * Interactive visualizations via Streamlit

## Research Context
This pipeline was developed as part of a research project at École Centrale de Lyon on cognitive development. It aims to automate the study of child-object interactions and produce usable indicators for researchers in cognitive sciences.
