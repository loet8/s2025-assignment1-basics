# UHM ECE 496B Spring 2025 Assignment 1: Basics

This asignment is created from Assignment 1 of [CS336 at Stanford taught in Spring 2024](https://stanford-cs336.github.io/spring2024/). 
For the full description of the original assignment, see the assignment handout at
[cs336_spring2024_assignment1_basics.pdf](./cs336_spring2024_assignment1_basics.pdf)

Check out useful [lectures from CS336 at Stanford](https://github.com/stanford-cs336/spring2024-lectures).

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix. Any improvements of the existing codebase
(including adaptations from Stanford to UHM workflows, modifications of PDF, etc) will be rewarded with extra points.

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

### [Click here](https://colab.research.google.com/drive/1_zYjI4dqgPLwBL1vrp3adA4sfW_qk-1t?usp=sharing) for an example setup at Colab

Caution! The free GPU runtimes are very limited! Make sure to disconnect and delete your runtime when you spend time writing code or switch to another task. Using colab GPU runtimes for too long might result in losing access to them  (inceased wait times and/or short session durations).

If any of this happens to you, please consult with the professor.

## ECE491B Assignment instructions

Follow along the [CS336@Stanford handout](./cs336_spring2024_assignment1_basics.pdf) with small deviations:
1. What the code looks like: clone https://github.com/igormolybog/s2025-assignment1-basics.git
2. What you can use: Implementation from scratch is preferred, but experiments are essential. If you are stuck with some implementation, just use the Huggingface/Pytorch implementation and proceed to the experiments
    - Submit the report reflecting your attempts at implementation for partial credit
3. How to submit: You will submit the report on the assignment to [Assignment Submission Form](https://forms.gle/CSRweWjuBxvYbb9MA). The code does not have to be attached as long as you include links to the main GitHub branch where your code lives and links to all of the Colab notebooks if applicable.
    - You don't need to submit to leaderboard.
4. Skip Problem (unicode2) from section 2.2
5. Problems (learning_rate, batch_size_experiment, parallel_layers, layer_norm_ablation, pre_norm_ablation, main_experiment):
    - get a free T4 GPU at Colab
    - reduce the number of total tokens processed down to 33,000,000 or even lower for faster iteration. Keep the number of tokens consistant across your experiments.
6. Problem (learning_rate):
    - validation loss can be anything
7. Skip Problem (leaderboard) from Section 7.5

