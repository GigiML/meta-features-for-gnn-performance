# Using ideas from meta-learning to understand Graph Neural Network performance

Code and resources associated with the paper **Using ideas from meta-learning to understand Graph Neural Network performance**.

## Purpose

This project aims to better understand why some Graph Neural Network (GNN) architectures perform better than others on small graph datasets. The main idea is to use **meta-features** of datasets to relate their structural properties to observed GNN performance.

## Repository contents

- extraction of meta-features from graph datasets;
- extraction of frequent and closed subgraph patterns;
- empirical evaluation of several GNN architectures;
- construction of a meta-database;
- learning regression trees to explain performance.

## Method

The considered descriptors are organized into four families:

1. basic graph statistics;
2. structural and topological descriptors;
3. spectral properties;
4. dataset-level classification properties.

These features are combined with the performance of several GNN architectures to learn an explanatory model at the dataset level.

## Main idea

Instead of explaining only an individual prediction, this work aims to explain the global behavior of a GNN architecture across a collection of datasets.

## Suggested structure

- `data/` for datasets and derived files;
- `src/` for feature extraction, subgraph mining, and training;


## Citation

If this repository is used, please cite the corresponding paper.
