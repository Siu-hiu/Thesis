# Master Thesis - Data Science & Society

## Project Title
Alzheimer's Disease Detection Using EEG Topographic Images

---

## Project Overview
The project explores feature representation for EEG-based Alzheimer's Disease detection, comparing 1D raw EEG signal features with 2D topographic image-like features derived from EEG signals. Both representations are evaluated using on Convolutional Neural Network-based architectures. 

The goal is to evaluate whether representing EEG signals with explicit spatial mapping improves classification performance. The repo implements image feature extraction, nested cross-validation and model comparison. 

## Repository Structure
```
THESIS_CODES/
│
├── data/
│   ├── derivatives/                     # Preprocessed EEG signals directory
│   ├── features/                        # 2D image features directory
│   ├── raw/                             # Unprocessed EEG signals directory
│   ├── CHANGES
│   ├── dataset_description.json           
│   ├── participants.json                # Meta data mapping dictionary
│   ├── participants.tsv                 # Meta data by each subject
│   └── README                           # Dataset description
│
├── jobs/
│   └── eeg_prep_gpu.sh                  # GPU job submission scripts
│
├── notebooks/                  
│   ├── eda.ipynb                        # Explore EEG signals with MNE library
│   ├── image.ipynb                      # Present the image feature
│   └── results.ipynb                    # Post-training analysis
│
├── src/
│   ├── models/                          # Model files directory
│   ├── dataset.py                       # Custom PyTorch dataset generator 
│   ├── eeg_processor.py                 # Transform signals into images for a single subject
│   ├── feature_loader.py                # Load image feature and synchronise with labels
│   ├── model_trainer.py                 # Train, evaluate and predict
│   ├── model_tuner.py                   # Hyperparameter tuning
│   ├── subject_processor.py             # Image transformation for all subjects
│   ├── util.py                          # Helper functions
│   └── __init__.py
│
├── tests/
│   ├── test_processor.py                # Test eeg_processor.py 
│   ├── test_subject.py                  # Test subject_processor.py
│   └── __init__.py
│
├── cross_validation.py                  # Nested cross validation
├── eegnet_baseline.py                   # 1D EEG signal feature training pipeline
├── experiment.py                        # Single train/test split for quick model behaviour check
├── main.py                              # 2D image feature training pipeline
├── environment.gpu.yml                  # Environment setting for GPU
├── environment.local.yml                # Environment setting for CPU 
├── .gitignore
├── .gitlab-ci.yml
├── README.md
└── requirements.txt

```

## Dataset
Dataset is publicly available on OpenNeuro: https://openneuro.org/datasets/ds004504/versions/1.0.7. 

Resting-state eyes-closed EEG recordings with 19 channels at 500Hz sampling rate. The subject groups include Alzheimer's Disease, Frontaltemporal Dimentia, and healthy controls. 

## Usage

### Dataset Download 
Download the dataset at https://openneuro.org/datasets/ds004504/versions/1.0.7/download. 

Choose a root directory when prompted and create a subfolder named ```data```. 

### Image Extraction 
To extract images from the EEG signals, run:
```
python -m src.subject_processor
```
The default configurations, including frequency band and the sliding window size, can be adjusted by changing the arguments when calling the methods of ```SubjectProcessor``` class.
```
processor.choose_band(band_name="alpha")
processor.choose_window_size(window_size=4)
```

The images will be saved under a folder name which corresponds to the band name of your choice under ```/data/features/```. 

By implementing the above two steps, your data repository will be arranged as below: 

```
├── data/
│   ├── derivatives/
│   │   ├── sub-001/
│   │   │   └── eeg/
│   │   │       └── sub-001_task-eyesclosed_eeg.set
│   │   ├── sub-002/
│   │   ├── sub-003/
│   │   ├── ...
│   │   ├── sub-086/
│   │   ├── sub-087/
│   │   └── sub-088/
│   │
│   ├── features/
│   │   ├── alpha/
│   │   │       └── sub-001_alpha_psd.npy
│   │   └── ...
│   │
│   ├── raw/
│   │   ├── sub-001/
│   │   │   └── eeg/
│   │   │       ├── sub-001_task-eyesclosed_channels.tsv
│   │   │       ├── sub-001_task-eyesclosed_eeg.json
│   │   │       └── sub-001_task-eyesclosed_eeg.set
│   │   ├── sub-002/
│   │   ├── sub-003/
│   │   ├── ...
│   │   ├── sub-086/
│   │   ├── sub-087/
│   │   └── sub-088/
│   │
│   ├── CHANGES
│   ├── dataset_description.json
│   ├── participants.json
│   ├── participants.tsv
│   └── CHANGES
```

To test for image extraction steps in ```eeg_processor.py``` and ```subject_processor.py```, run: 
```
pytest -v
```

### Run Experiment

## 1D feature
1D CNN model pipeline is implemented separately because the original EEGNet architecture is written in Keras. To run signal extraction and nested cross validation:
```
python eegnet_baseline.py
```

## 2D feature
```
python main.py
```

## Requirements
```bash
python==3.10
torch==2.1.0
torchmetrics==1.2.0
numpy==1.26.4
mne==1.7.0
scikit-learn==1.3.0
matplotlib==3.8.0
```

## Support
For any issue, please contact xiaoxiao.yang@outlook.com.
