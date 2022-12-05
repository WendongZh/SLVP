# Self-Labeling Video Prediction  
  
Code for paper "Self-Labeling Video prediction".  

In this project, we provide the code of our paper "Self-Labeling Video prediction" which is submitted to journal for possible publication.  

In this paper, we propose a novel self-labeling framework that can easily equip with popular video prediction models and help resolve the multi-modal entanglement over the latent variables.  


## Prerequisites
- Python 3.7
- PyTorch 1.8
- NVIDIA GPU + CUDA cuDNN (>=12GB Memory)
- scikit-image >= 0.18.1 

## Setup 
In this project, we provide the code of our method on the KTH action dataset. We have provided the pre-clustered pseudo labels in 'Right_kth_kmeans6_label_allkth_64nonorm.npy' with 6 categories. 

The pre-preparations are list as follows:

1) Download the KTH action dataset from this [link](https://www.csc.kth.se/cvap/actions/).

2) Modify the dataset directory '--train_data_paths' and '--valid_data_paths' in both train.sh

3) For training, run
```bash
sh train.sh
```

4) For testing, please first change the directory for '--pretrained_model' in test.sh, and then run
```bash
sh test.sh
```

## Appreciation
The codes refer to [CPL](https://github.com/jc043/CPL). Thanks for the authors！

