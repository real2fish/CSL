# Official Implementation of MAST, A Memory-Aware Activation Checkpointing Planner for Shapelet Transformer.

## Requirements
* Python3.x
* Pytorch
* Numpy
* Sklearn
* tslearn
* tsaug
* Gurobi（linux）

## Datasets
We use the UEA datasets to test the planner.

* [UEA Archive](http://www.timeseriesclassification.com/)


The datasets should be in the "Multivariate_ts/" folder with the structure `Multivariate_ts/[dataset_name]/[dataset_name]_TRAIN.ts` and `Multivariate_ts/[dataset_name]/[dataset_name]_TEST.ts`.


## Usage

To evaluate the planner with a user-defined memory budget (on classification tasks by default):

`python UEA_MAST.py Cricket --checkpoint True --budget 2 -b=8`

Use `-h` or `--help` option for the detailed messages of the other options, such as the hyper-parameters and the random seed.
