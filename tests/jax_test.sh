#!/bin/sh

wget https://raw.githubusercontent.com/jax-ml/jax/refs/heads/main/examples/mnist_classifier.py
wget https://raw.githubusercontent.com/jax-ml/jax/refs/heads/main/examples/datasets.py
mkdir -p examples
mv datasets.py examples/.
python mnist_classifier.py

