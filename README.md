# Relational Embedded Few-Shot Network for GI image classification

## Overview
A deep learning model for classifying gastrointestinal diseases using endoscopic images. It utilizes Few-Shot Learning, data augmentation, relational embedding, and bi-level routing attention to accurately analyze medical images.
![arct](readme/arct.png)

## Approach
- Few-Shot Learning (FSL)
<img src="readme/FSLparadigm.png" alt="FSL" title="FSL" width="400"/>

- Data Augmentation via Task Interpolation
<img src="readme/taskinterpolation.png" alt="Task" title="Task" width="400"/>

- Relational Embedding
  - *Self Correlation Representation (SCR)*  
  <img src="readme/scr.png" alt="scr" title="scr" width="400"/>
  
  - *Cross Correlation Representation (CCR)*  
  <img src="readme/ccr.png" alt="ccr" title="ccr" width="400"/>

- Bi-Level Routing Attention Mechanism
<img src="readme/biattn.png" alt="biattn" title="biattn" width="400"/>

## Installation
```bash
git clone https://github.com/ygritte723/gi_image_proj.git
cd gi_image_proj
pip install -r requirements.txt
```
## Usage
Detailed instructions for data preparation, training, and evaluation.

## Dataset
Tested on Hyper-Kvasir and Mini-ImageNet datasets.

## Results
Achieved notable performance metrics on the Kvasir dataset.

| Metric     | Value  |
|------------|--------|
| Accuracy   | 0.901  |
| Precision  | 0.845  |
| Recall     | 0.942  |
| F1 Score   | 0.891  |

