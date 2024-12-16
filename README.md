# OCC
Code and Models for X-ray Pulmonary Disease Screening with Multi-level Feature Fusion

# Run:

The /dataset subfolder contains the multi-level features extracted from each sample (including deep encoding features and initial screening anomaly scores obtained in the second research task), as well as the data partitioning situation (consistent with the second research task).

The /feature_extraction subfolder contains code for extracting lung sub-region and various category features.

The /img_data subfolder contains the remaining samples after the difficult case selection in the first research task, along with their corresponding lung masks and region-specific masks (the samples are the same as those used in the second research task).

The /OC-SVM folder contains the screening-related code:

/OC-SVM/result stores the relevant screening models, factor analysis models, data normalization models, and selected features, etc.;
`/OC-SVM/ensemble_manual.py` implements the screening process using only manually selected features;
`/OC-SVM/ensemble_DL_As.py` integrates the initial screening anomaly scores obtained in the second research task into the model built with manual features;
`/OC-SVM/ensemble_all.py` performs the screening process with the complete model that combines all multi-level features and initial screening anomaly scores.

# Test:

`/OC-SVM/ensemble_test.py` is the testing process for the complete model.
`/OC-SVM/feature_select.py` contains the code for feature selection.


# Requirements
Some important required packages include:
* torch == 2.3.0
* torchvision == 0.18.0
* TensorBoardX == 2.6.22
* Python == 3.10.14
* numpy == 1.26.4
* opencv-python == 4.10.0.84
* scipy == 1.14.0