# DiagSet: a dataset for prostate cancer histopathological image classification

## Description

The dataset consists of three different partitions: DiagSet-A, containing over 2.6 million tissue patches extracted from 430 fully annotated scans; DiagSet-B, containing 4675 scans with assigned binary diagnosis; and DiagSet-C, containing 46 scans with diagnosis given independently by a group of histopathologists.

![DiagSet-A samples](samples.png)
*Samples from DiagSet-A dataset, containing patches from different tissue classes extracted at different magnifications.*

## Access

The data is publicly available and can be accessed by anyone (after registration) at <https://ai-econsilio.diag.pl>. Note that after registration site administrator will need to activate your account, which should typically take less than 24 hours.

In case of any problems with accessing the website you can either email the [website administrator](mailto:pawel.wasowicz@diag.pl) directly or, if that fails, create an issue here.

## DiagSet-A container

For easier access to the data we prepared a Python container for loading the patches from raw data files.

## Publication

More detailed description about the dataset and the conducted experiments can be found at <https://arxiv.org/abs/2105.04014>. To cite the dataset, you can use:
```bibtex
@article{koziarski2021diagset,
  title={DiagSet: a dataset for prostate cancer histopathological image classification},
  author={Koziarski, Micha{\l} and Cyganek, Bogus{\l}aw and Olborski, Bogus{\l}aw and Antosz, Zbigniew and {\.Z}ydak, Marcin and Kwolek, Bogdan and W{\k{a}}sowicz, Pawe{\l} and Buka{\l}a, Andrzej and Swad{\'z}ba, Jakub and Sitkowski, Piotr},
  journal={arXiv preprint arXiv:2105.04014},
  year={2021}
}
```