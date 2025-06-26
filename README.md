# dl-generative-kittens
Exploration of generative models for kitten generation based on AnnikaV9 'Cat Dataset' (https://av9.dev/cat-dataset/).

## Project Overview
This project explores various generative models to create realistic images of kittens. It leverages deep learning techniques and architectures such as DCGAN, WGAN, and StyleGAN2-Ada.

## Data
The dataset used in this project is the AnnikaV9 'Cat Dataset'. You can download it using the following command:

```bash
curl -L -o ./data/cat-dataset.zip https://www.kaggle.com/api/v1/datasets/download/borhanitrash/cat-dataset
```

## Models
The project includes implementations and experiments with the following generative models:
- **DCGAN**: Deep Convolutional Generative Adversarial Network
- **WGAN**: Wasserstein GAN

## Results
Generated images and metrics are stored in the `checkpoints/` and `metrics/` directories. Plots summarizing the results can be found in the `plots/` directory.

## Code Structure
- `src/`: Contains the source code for data preprocessing, model training, and utilities.
- `data/`: Directory for storing the dataset.
- `checkpoints/`: Directory for saving model checkpoints and generated samples.
- `metrics/`: Directory for storing evaluation metrics.
- `mlruns/`: Directory for tracking experiments using MLflow.

## How to Run
1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download the dataset as mentioned above.
3. Train a model by running the corresponding script in the `src/` directory.

## License
This project is licensed under the terms specified in the `LICENSE` file.
