# Recognition of hand motions molding clay (SHREC 2024)

For contest details see [https://www.shrec.net/SHREC-2024-hand-motion/](https://www.shrec.net/SHREC-2024-hand-motion/).


## Contents

This repository contains:
* A **description** of the proposed method in *Method Description/*
* The data of the contest in *Data Split/*
* An **executable** python file for training and producing the final results called *classifier.py*
* A pretrained model with the optimal parameters under *Pretrained/best.pkl*
* The classification **results** on the provided test set in *results.txt*
* A jupyter notebook showcasing the data parsing, manipulation and visualization capabilities named *data_exploration.ipynb*
* A jupyter notebook for testing different model parameters called *learning.ipynb*
* A jupyter notebook that was used to perform hyperparameter optimization called *tuning.ipynb*


## Setup

First clone the repository using:
```bash
git clone https://github.com/GavalasDev/shrec2024
cd shrec2024
```

Then, create a new virtual environment and install all dependencies by running:
```bash
python3 -m pip install -r requirements.txt
```

The *requirements.txt* file includes all necessary dependencies for executing the *classifier.py* script. Notebooks *data_exploration.ipynb* and *tuning.ipynb* have extra dependencies which are commented out by default.

The python version used was *3.11.8*.

## Classifying

The pretrained model can be immediately used to classify a set of labeled sequences (like *Data Split/Test-set*):

```bash
python3 classifier.py classify --labeled "Data Split/Test-set" > results.txt
```

or individual sequences:

```bash
python3 classifier.py classify "Data Split/Test-set/Centering/01.txt"
```

## Training

Models can be trained using the *train* subcommand. For example, the *best* model was trained like so:

```bash
python3 classifier.py train --name best --n_components 10 --n_mix 8 --n_iter 315 --downsample_step 7 "Data Split/Train-set/"
```

Similarly, a different model could be trained and subsequently used like so:

```bash
python3 classifier.py train --name new_model --n_components 2 --n_mix 2 --n_iter 50 --downsample_step 25 "Data Split/Train-set/"

python3 classifier.py classify --model new_model --labeled "Data Split/Test-set"
```
