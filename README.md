<div align="center">
  
# CSL
## Contrastive Shapelet Learning for General-purpose Unsupervised (Self-supervised) Multivariate Time Series Representation Learning

</div>

## References
Please consider citing our related work if you find them helpful:
* Zhiyu Liang, Jianfeng Zhang, Chen Liang, Hongzhi Wang, Zheng Liang, and Lujia Pan. 2023. A Shapelet-Based Framework for Unsupervised Multivariate Time Series Representation Learning. Proc. VLDB Endow. 17, 3, 386–399. https://doi.org/10.14778/3632093.3632103
* Zhiyu Liang, Xiongfei Li, Chen Liang, Yuchen Hu, Yutong Zhao, Zheng Liang, Jianfeng Zhang, Hongzhi Wang, and Lujia Pan. 2026. Towards GPU memory-aware efficient contrastive shapelet learning for unsupervised representation learning in multivariate time series. The VLDB Journal 35, 5. https://doi.org/10.1007/s00778-026-00996-2
* Zhiyu Liang, Chen Liang, Zheng Liang, Hongzhi Wang, and Bo Zheng. 2024. TimeCSL: Unsupervised Contrastive Learning of General Shapelets for Explorable Time Series Analysis. Proc. VLDB Endow. 17, 12, 4489–4492. https://doi.org/10.14778/3685800.3685907

## Requirements
* Python3.x
* Pytorch
* Numpy
* Sklearn
* tslearn
* tsaug

## Datasets
We use the 30 datasets from UEA archive and four anomaly detection datasets in this study.

* [UEA Archive](http://www.timeseriesclassification.com/)

* [Soil Moisture Active Passive satellite (SMAP)
and Mars Science Laboratory rover (SML) datasets](https://dl.acm.org/doi/10.1145/3219819.3219845)

* [Server Machine
Dataset, SMD](https://dl.acm.org/doi/10.1145/3292500.3330672)

* [Application Server Dataset, ASD](https://dl.acm.org/doi/10.1145/3447548.3467075)

The UEA datasets should be in the "Multivariate_ts/" folder with the structure `Multivariate_ts/[dataset_name]/[dataset_name]_TRAIN.ts` and `Multivariate_ts/[dataset_name]/[dataset_name]_TEST.ts`.

For SMAP and MSL datasets, create a folder named `SMAP&MSL` under 'AD_data/', and put the `.npy` data files into `AD_data/SMAP&MSL/`.

Similarly, to test SMD and ASD datasets, create a folder named `SMD&ASD` under 'AD_data/' then put the data files of `.pkl` into the folder `AD_data/SMD&ASD/`.


## Usage

### Commands for CSL

To evaluate the UEA datasets using the commands:

Classification:

`python UEA.py [dataset_name]`

Clustering:

`python UEA.py [dataset_name] --task clustering`

For anomaly detection, use the following command:

`python CSL_AD.py [dataset_name] --window-size [window-size]`

Use `-h` or `--help` option for the detailed messages of the other options, such as the hyper-parameters and the random seed.

### New: Efficient and memory-aware activation checkpointing is now supported using our MAST planner 

* [MAST](https://github.com/real2fish/CSL/tree/main/MAST)
