# PASBR
Aiming at the problems that the session-based recommendation method combined with graph neural network in e-commerce scene adopts the construction method of session graph with information loss,and fails to fully consider the influencing factors of feature modeling,a price-aware information lossless model is proposed for session-based recommendation.The overall framework of PASBR is depicted bellow.

<img src="https://github.com/JXY66/PILL/assets/104990190/afe73d9f-434b-4d72-af68-0fc7710e07f0" width = "500px" align=center />

## Requirements
- PyTorch  1.6.0  （cu102版本）
- NumPy  1.19.1
- Pandas  1.1.3
- DGL-cu102  0.6.1
- CUDA    10.2
- cuDNN    7.6
- Python    3.7.9
- Notice: The above library versions can also run with higher versions, such as pytorch 1.7.0 + cu110, dgl-cu110 (0.6.1). All library dependencies are also stored in requirements.txt and can be installed directly using the above commands.
## Parameters
- Fixed parameters: learning rate lr=1e-3, batch size=512, epoch=30, weight decay=1e-4, patience=5. 

- The parameter --dataset-dir is used to specify different datasets for input:
diginetica/yoochoose1_10/yoochoose1_4
## Usage
(1) Experimental Setup:
Set the parameters in main.py according to the following configurations, and then run main.py.
For the diginetica dataset: embedding-dim=32, feat-drop=0.2, num_layers=4
For the Yoochoose1_10 dataset: embedding-dim=32, feat-drop=0.5, num_layers=4
For the Yoochoose1_4 dataset: embedding-dim=64, feat-drop=0.5, num_layers=4

(2) Ablation Experiment:
For different datasets, after setting the parameters as mentioned in (1), make the following additional modifications:
A. Intent Module Ablation:
In main.py, set --without_intent=1 (change default=0 to 1), keep --without_price=0, and then run main.py.
B. Price Tolerance Factor Module Ablation:
In main.py, set --without_price=1 (change default=0 to 1), keep --without_intent=0, and then run main.py.
C. Dynamic Interest Module Ablation:
a. Keep --without_intent=0 and --without_price=0 in main.py.
b. In the pill.py file:
   - Uncomment line 372 and comment out line 373.
   - Uncomment line 680 and comment out line 678.
   This will exclude the concatenation of dynamic interest representation (pos_g).

Note: When running comparative algorithms, for algorithms in the "other/baselines" folder, run main.py in each folder. For GCSAN, run GCSAN.py. For GRU4Rec, follow the instructions in the readme.md file to set up the environment and then run main.py. For pop and STAMP algorithms, simply run pop.py and STAMP.py, respectively. Adjust the dataset name according to the content in each .py file.

(3) Hyperparameter Experiment:
Set different values for the hyperparameters in main.py and then run main.py.
A. embedding-dim: Try values of 16, 32, 64, 96, 128.
B. feat-drop: Try values of 0.1, 0.2, 0.3, 0.4, 0.5.
C. num-layers: Try values of 2, 4, 6, 8.
D. For experiments with different feature fusion methods, in the pill.py file, uncomment one of the four fusion methods from lines 596-607, and comment out the other three methods. The four methods are: Mul (line 596), Add (line 598), Concat (lines 600-601), FG (lines 603-607). Then run main.py.
## Datasets
    - [Diginetica](https://competitions.codalab.org/competitions/11161)
    - [Gowalla](https://snap.stanford.edu/data/loc-Gowalla.html)
    - [Last.fm](http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html)
## Citation
```
@inproceedings{chen2020lessr,
    title="Handling Information Loss of Graph Neural Networks for Session-based Recommendation",
    author="Tianwen {Chen} and Raymond Chi-Wing {Wong}",
    booktitle="Proceedings of the 26th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '20)",
    pages="1172-–1180",
    year="2020"
}
```
