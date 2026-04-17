# EnrichGAN: Exploiting Enriched Discriminator Representations for Training GANs under Limited Data
The official pytorch implementation of the paper "EnrichGAN: Exploiting Enriched Discriminator Representations for 
Training GANs under Limited Data".


## 1. Description
The code is structured as follows:
* models.py: all the models' structure definition.

* operation.py: the helper functions and data loading methods during training.

* train.py: the main entry of the code, execute this file to train the model, the intermediate results and checkpoints will be automatically saved periodically into a folder "train_results".

* eval.py: generates images from a trained generator into a folder, which can be used to calculate FID score.

* benchmarking: the functions we used to compute FID are located here, it automatically downloads the pytorch official inception model. 

* lpips: this folder contains the code to compute the LPIPS score, the inception model is also automatically download from official location.

## 2. How to run
Place all your training images in a folder, and simply call
```
python train.py --path /path/to/RGB-image-folder --output_path /path/to/the/output
```

Once finish training, you can generate images by:
```
cd ./train_results/name_of_your_training/
python eval.py --n_sample 5000 
```