# UHM ECE 496B Spring 2025 Assignment 1: Basics

This asignment is created from Assignment 1 of [CS336 at Stanford taught in Spring 2024](https://stanford-cs336.github.io/spring2024/). 
For the full description of the original assignment, see the assignment handout at
[cs336_spring2024_assignment1_basics.pdf](./cs336_spring2024_assignment1_basics.pdf)

Check out useful [lectures from CS336 at Stanford](https://github.com/stanford-cs336/spring2024-lectures).

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

0. Set up a conda environment and install packages:

``` sh
conda create -n ece496b_basics python=3.10 --yes
conda activate ece496b_basics
pip install -e .'[test]'
```

1. Run unit tests:

``` sh
pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

2. Download the TinyStories data and a subsample of OpenWebText:

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

## ECE491B Assignment instructions

Follow along the [CS336@Stanford handout](./cs336_spring2024_assignment1_basics.pdf) with small deviations:
1. If you are stuck with some implementation, just use the Huggingface/Pytorch implementation
    - Submit the report reflecting your attempts at implementation for partial credit
2. Skip Problem (unicode2) from section 2.2
3. Problems (learning_rate, batch_size_experiment, parallel_layers, layer_norm_ablation, pre_norm_ablation, main_experiment):
    - get a free T4 GPU at Colab
    - reduce the number of total tokens processed down to 33,000,000 or even lower for faster iteration
4. Problem (learning_rate):
    - validation loss can be anything
5. Skip Problem (leaderboard) from Section 7.5

