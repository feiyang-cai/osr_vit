# Open Set Recognition using Vision Transformer with an Additional Detection Head

## Getting Started
This code repository includes the detailed instructions for reproducing experiments of MNIST, SVHN, CIFAR10, CIFAR+N, TinyImageNet, and CUB reported in the paper.
Becuase the size of trained weights are too huge and the shareable link to these weights may lead to a compromise in anoymity, we decide to not share the trained models at this moment.
We will share the code and weights in public once this paper is accepted.
You can easily train your own models by running the code in this repository.

All our experiments are performed in an 80-core Ubuntu Linux virtual machine with 128GB RAM and 4 Tesla V100 GPUs. 

### Environment and dependencies

Create an conda environment and install the dependencies
```
conda env create -f environment.yml 
```

Activate the environment
```
conda activate osr
```

### Pretrained ViT model
Download pretrained ViT-B/16 from this [[link](https://drive.google.com/file/d/1gEcyb4HUDzIvu7lQWTOyDC1X00YzCxFx/view?usp=sharing)] (this link will **not** lead to a compromise in anonymity)
and put it in the folder of "./pretrained_model/"

### Datasets
Download the TinyImageNet dataset from this [[link](https://drive.google.com/file/d/1oJe95WxPqEIWiEo8BI_zwfXDo40tEuYa/view)] (this link will **not** lead to a compromise in anonymity) and extract it in the folder of "./data/"

The other datasets will be downloaded automatically if they are not existed in the "./data/" folder.

## Train
<span style="color:green">
"note.txt" contains the complete scripts of training and evaluating for each experiment.
</span>

### Stage 1 (Closed set training)
Train only 1 random "known/unknown" split trial 

```
python train_classifier.py --exp-name osrclassifier --n-gpu 4 --tensorboard --image-size 224 --batch-size 256 --num-workers 16 --train-steps 4590 --lr 0.01 --wd 1e-5 --dataset <known_dataset> --num-classes <num_of_known_classes> --random-seed <random_seed>  --checkpoint-path ./pretrained_model/imagenet21k+imagenet2012_ViT-B_16.pth
```

Train 5 random "known/unknown" split trials
```
python ./train_classifier_batch.py --num-classes <num_of_known_classes> --checkpoint-path ./pretrained_model/imagenet21k+imagenet2012_ViT-B_16.pth --dataset <known_dataset>
```
where <known_dataset> can be MNIST, SVHN, CIFAR10, TinyImageNet, or CUB.

### Stage 2 (Open set training)
Train the detector using the trained model from training stage 1
```
python train_detector.py --exp-name osrdetector --n-gpu 4 --tensorboard --image-size 224 --batch-size 256 --num-workers 16 --train-steps 4590 --lr 0.01 --wd 1e-5 --dataset <known_dataset> --num-classes <num_of_known_classes> --checkpoint-path <trained_model> --random-seed <random_seed>
```

Train detectors by loading all the models whose settings match <num_of_known_classes> and <known_dataset> in "./experiments/" folder 
```
python ./train_detector_batch.py --num-classes <num_of_known_classes> --dataset <known_dataset>
```


## Evaluation
Evaluate the trained model
```
python ./measure_osrdetector.py --exp-name osrdetector --in-dataset <known_dataset> --out-dataset <unknown_dataset> --in-num-classes <num_of_known_classes> --out-num-classes <num_of_unknown_classes>
```
