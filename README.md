# Do Loyal Users Enjoy Better Recommendations?: Understanding Recommender Accuracy from a Time Perspective

Code to reproduce the experiments from the paper: Do Loyal Users Enjoy Better Recommendations?: Understanding Recommender Accuracy from a Time Perspective (ICTIR2022)

This repository has the implementations for four models:
1. BPR
2. NeuMF: we follow https://github.com/hexiangnan/neural_collaborative_filtering
3. LightGCN: we follow https://github.com/RUCAIBox/RecBole
4. SASRec: we follow https://github.com/pmixer/SASRec.pytorch
5. TiSASRec: we follow https://github.com/JiachengLi1995/TiSASRec


# Dataset
Dataset can be downloaded from https://drive.google.com/drive/folders/1TyQysstuaUo4IYb3WaVt4dVFOyZu3oN_.

# Environment Requirement
## BPR
- Tensorflow 1.14
- Python 3.6.9

## NeuMF
- Tensorflow 1.14
- Python 3.6.9
- Keras 2.3.0

## LightGCN
- Install RecBole package in https://github.com/RUCAIBox/RecBole

## SASRec
- PyTorch >= 1.6

## TiSASrec
- Tensorflow 1.14

# Data formart


User ID | Item Id | Rating | Timestamp | Year
--------|---------|--------|-----------|-----
...|...|...|...|...

# Examples to run the code

```
cd BPR/
python test.py --path data/ --data movielens --selected_year 5 --gpu 1
```


