# EnrichGAN: Exploiting Enriched Discriminator Representations for Training GANs under Limited Data

This repository contains the official PyTorch implementation of the paper **"EnrichGAN: Exploiting Enriched Discriminator Representations for Training GANs under Limited Data"**.

## 1. Repository Structure

*   **`models.py`**: Contains the architectural definitions of the generator, discriminator, and decoder.
*   **`models_SVA.py`**: Contains the definition of the SVA module.
*   **`operation.py`**: Provides helper functions and data-loading pipelines.
*   **`train.py`**: The main training script. Intermediate results and model checkpoints are automatically saved to the `train_results/` directory during execution.
*   **`eval.py`**: Synthesizes images using a pre-trained generator and saves them to a designated directory for subsequent quantitative evaluation.
*   **`benchmarking/`**: Scripts for computing quantitative evaluation metrics. (Note: The official PyTorch Inception model is downloaded automatically upon first use).
*   **`lpips/`**: Implementation of the perceptual loss. (The corresponding pre-trained network is downloaded automatically).

## 2. Usage

### Training

To train the model, place all your training images into a single directory and run the following command:

```bash
python train.py --path /path/to/RGB-image-folder
```

### Generation

Upon completing the training process, you can synthesize image samples using the saved checkpoint by running:

```bash
cd ./train_results/name_of_your_training/
python eval.py --n_sample 5000 
```

### Evaluation

After generating the images, you can compute the evaluation metrics via:

```bash
cd ./benchmarking/
python metrics_compute.py --path_a /path/to/RGB-image-folder --path_b ./train_results/name_of_your_training
```