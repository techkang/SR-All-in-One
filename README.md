# SR All in One
This repo has many implementation of state of the art single image super resolution method. It is user friendly and more extendable than most of the others implementation.

## paper list

|model name| paper | link |
|---|---|---|
| DBPN/DDPBN| Deep Back-Projection Networks for Super-Resolution| https://arxiv.org/abs/1803.02735|
| DPDNN| Denoising Prior Driven Deep Neural Network for Image Restoration| https://arxiv.org/abs/1801.06756|
| EDSR| Enhanced Deep Residual Networks for Single Image Super-Resolution| https://arxiv.org/abs/1707.02921|
| MIRNet| Learning Enriched Features for Real Image Restoration and Enhancement| https://arxiv.org/abs/2003.06792|
| RCAN| Image Super-Resolution Using Very Deep Residual Channel Attention Networks| https://arxiv.org/abs/1807.02758|
| RDN| Residual Dense Network for Image Super-Resolution| https://arxiv.org/abs/1802.08797|
| SRFBN| Feedback Network for Image Super-Resolution| https://arxiv.org/abs/1903.09814|


## data prepare
Train dataset: [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)  
Test dataset: Set5, Set14, Urban100, BSD100, Manga109, [download](http://vllab.ucmerced.edu/wlai24/LapSRN/results/SR_testing_datasets.zip)  
put train set and test set in `dataset` folder like this.

    .
    ├── DIV2K
    │   ├── train_HR
    │   └── valid_HR
    └── SRbenchmark
        ├── BSD100
        ├── Manga109
        ├── Set14
        ├── Set5
        └── Urban100

## train
```shell script
python train.py --config-file experiment/rdn/base.yaml
```
## test
```shell script
python train.py --config-file experiment/rdn/base.yaml --eval-only model.weights path/to/model
```
Please refer to [README](config/README.md) about config for further details.
## note

 - I don't guarantee performance of models.
 - The architecture of this repo and some code comes from (detectron2)[https://github.com/facebookresearch/detectron2].

## contribution
All kinds of contributions are appreciated, include but not limited to add new model, report bugs, discuss details or provide pretrained models.
