# PILL
Aiming at the problems that the session-based recommendation method combined with graph neural network in e-commerce scene adopts the construction method of session graph with information loss,and fails to fully consider the influencing factors of feature modeling,a price-aware information lossless model is proposed for session-based recommendation.
## Requirements
- PyTorch  1.6.0  （cu102版本）
- NumPy  1.19.1
- Pandas  1.1.3
- DGL-cu102  0.6.1
- CUDA    10.2
- cuDNN    7.6
- Python    3.7.9
- Notice: The above library versions can also run with higher versions, such as pytorch 1.7.0 + cu110, dgl-cu110 (0.6.1). All library dependencies are also stored in requirements.txt and can be installed directly using the above commands.
  
## Usage
1. Install the requirements.  
    If you use Anaconda, you can create a conda environment with the required packages using the following command.
    ```sh
    conda env create -f packages.yml
    ```
    Activate the created conda environment.
    ```
    conda activate lessr
    ```

2. Download and extract the datasets.
    - [Diginetica](https://competitions.codalab.org/competitions/11161)
    - [Gowalla](https://snap.stanford.edu/data/loc-Gowalla.html)
    - [Last.fm](http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html)

3. Preprocess the datasets using `preprocess.py`.  
    For example, to preprocess the *Diginetica* dataset, extract the file *train-item-views.csv* to the folder `datasets/` and run the following command:
    ```sh
    python preprocess.py -d diginetica -f datasets/train-item-views.csv
    ```
    The preprocessed dataset is stored in the folder `datasets/diginetica`.  
    You can see the detailed usage of `preprocess.py` by running the following command:
    ```sh
    python preprocess.py -h
    ```

4. Train the model using `main.py`.  
    If no arguments are passed to `main.py`, it will train a model using a sample dataset with default hyperparameters.
    ```sh
    python main.py
    ```
    The commands with suggested hyperparameters to train models on the different datasets are as follows:
    ```sh
    python main.py --dataset-dir datasets/diginetica --embedding-dim 32 --num-layers 4
    python main.py --dataset-dir datasets/gowalla --embedding-dim 64 --num-layers 4
    python main.py --dataset-dir datasets/lastfm --embedding-dim 128 --num-layers 4
    ```
    You can see the detailed usage of `main.py` by running the following command:
    ```sh
    python main.py -h
    ```

5. Use your own dataset.
    1. Create a subfolder in the `datasets/` folder.
    2. The subfolder should contain the following 3 files.
        - `num_items.txt`: This file contains a single integer which is the number of items in the dataset.
        - `train.txt`: This file contains all the training sessions.
        - `test.txt`: This file contains all the test sessions.
    3. Each line of `train.txt` and `test.txt` represents a session, which is a list of item IDs separated by commas. Note the item IDs must in the range of `[0, num_items)`.
    4. See the folder `datasets/sample` for an example of a dataset.

## Citation
If you use our code in your research, please cite our paper:
```
@inproceedings{chen2020lessr,
    title="Handling Information Loss of Graph Neural Networks for Session-based Recommendation",
    author="Tianwen {Chen} and Raymond Chi-Wing {Wong}",
    booktitle="Proceedings of the 26th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '20)",
    pages="1172-–1180",
    year="2020"
}
```
