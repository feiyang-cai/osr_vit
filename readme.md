<div id="top"></div>

# Open Set Recognition using Vision Transformer
This repository provides the implementation of an open set recognition method using vision transformer.

## Citation
If you use our method, please cite our paper.

_Open Set Recognition using Vision Transformer with an Additional Detection Head_<br>Feiyang Cai, Zhenkai Zhang, Jie Liu, and Xenofon Koutsoukos
[[PDF](https://arxiv.org/pdf/2203.08441.pdf)]
```
@article{cai2022open,
       author = {Cai, Feiyang and Zhang, Zhenkai and Liu, Jie and Koutsoukos, Xenofon},
        title = {Open Set Recognition using Vision Transformer with an Additional Detection Head},
      journal = {arXiv preprint arXiv:2203.08441},
         year = 2022,
}
```
<p align="right">(<a href="#top">back to top</a>)</p>


## Getting Started
This code repository includes the detailed instructions for reproducing experiments of MNIST, SVHN, CIFAR10, CIFAR+N, TinyImageNet, and CUB reported in the paper.

This repository is still updating. We will provide our trained models in this repository later.

All our experiments are performed in an 80-core Ubuntu Linux virtual machine with 128GB RAM and 4 Tesla V100 GPUs. 

### Prerequisites

Create an conda environment and install the dependencies
```bash
conda env create -f environment.yml 
```

Activate the environment
```bash
conda activate osr
```

### Pretrained ViT model
Download pretrained ViT-B/16 from this [[link](https://drive.google.com/file/d/1gEcyb4HUDzIvu7lQWTOyDC1X00YzCxFx/view?usp=sharing)]
and put it in the folder of "./pretrained_model/"

### Datasets
Download the TinyImageNet dataset from this [[link](https://drive.google.com/file/d/1oJe95WxPqEIWiEo8BI_zwfXDo40tEuYa/view)] and extract it in the folder of "./data/"

The other datasets will be downloaded automatically if they are not existed in the "./data/" folder.

<p align="right">(<a href="#top">back to top</a>)</p>

## Train
<span style="color:green">
"scripts.txt" contains the complete scripts of training and evaluating for each experiment.
</span>

### Stage 1 (Closed set training)
Train only 1 random "known/unknown" split trial 

```python
python train_classifier.py --exp-name osrclassifier --n-gpu 4 --tensorboard --image-size 224 --batch-size 256 --num-workers 16 --train-steps 4590 --lr 0.01 --wd 1e-5 --dataset <known_dataset> --num-classes <num_of_known_classes> --random-seed <random_seed>  --checkpoint-path ./pretrained_model/imagenet21k+imagenet2012_ViT-B_16.pth
```

Train 5 random "known/unknown" split trials
```python
python ./train_classifier_batch.py --num-classes <num_of_known_classes> --checkpoint-path ./pretrained_model/imagenet21k+imagenet2012_ViT-B_16.pth --dataset <known_dataset>
```
where <known_dataset> can be MNIST, SVHN, CIFAR10, TinyImageNet, or CUB.

### Stage 2 (Open set training)
Train the detector using the trained model from training stage 1
```python
python train_detector.py --exp-name osrdetector --n-gpu 4 --tensorboard --image-size 224 --batch-size 256 --num-workers 16 --train-steps 4590 --lr 0.01 --wd 1e-5 --dataset <known_dataset> --num-classes <num_of_known_classes> --checkpoint-path <trained_model> --random-seed <random_seed>
```

Train detectors by loading all the models whose settings match <num_of_known_classes> and <known_dataset> in "./experiments/" folder 
```python
python ./train_detector_batch.py --num-classes <num_of_known_classes> --dataset <known_dataset>
```

<p align="right">(<a href="#top">back to top</a>)</p>

## Evaluation
Evaluate the trained model
```
python ./measure_osrdetector.py --exp-name osrdetector --in-dataset <known_dataset> --out-dataset <unknown_dataset> --in-num-classes <num_of_known_classes> --out-num-classes <num_of_unknown_classes>
```
<p align="right">(<a href="#top">back to top</a>)</p>

## Trained Models
We will provide our trained models in this repo later.

<p align="right">(<a href="#top">back to top</a>)</p>

## Acknowledgments
<p align="right">(<a href="#top">back to top</a>)</p>

## License
[MIT](https://choosealicense.com/licenses/mit/)
