# Polling Simulator

This is a lightweight Python tool for simulating political polls, based
on my understanding of how pollsters handle real data. It comes with
several methods for sampling electorates and aggregating polling
data built-in, but is designed in a way to easily allow for customization
as necessary.

# Installation
_Note: currently the only documented way to install this tool 
is by cloning the repo and installing an environment via `conda`.
It is, however, written as a Python package, so making it `pip`-installable
via PyPI would not be a heavy lift. If you'd like to see that, please
feel free to post an issue on the repo (or better yet, make a pull request);
if there's enough interest I'm happy to consider doing it._

## Assumptions:
* You are using a Linux-like OS (e.g. Ubuntu, Mac OSX)
* You are familiar with the command line
* You have [`conda`](https://docs.conda.io/en/latest/) installed

```bash
$ git clone https://github.com/AndrewRook/polling_simulator.git
$ cd polling_simulator
$ conda env create -f environment.yml
$ conda activate polling_simulator
```

# Usage:
For usage examples, check out the notebooks in this repository. There
are also docstrings for core functions if you want general guidance.