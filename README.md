# Image Captioning
This project aims to create a neural network architecture consisting of both CNNs (Encoder) and LSTMs (Decoder) to automatically generate captions from images. The models are trained on the [Flickr8k dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k) and on the [Flickr30k dataset](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset), and implemented based on the following papers: [paper](https://arxiv.org/pdf/1411.4555) and [paper](https://arxiv.org/pdf/1502.03044), using PyTorch. The image captioning model is designed to leverage the power of pre-trained convolutional neural networks (CNN) for image feature extraction and recurrent neural networks (RNN) for sequence modeling. We also explore attention mechanisms for improved performance.

## Dataset
- `The Flickr8k Dataset`: This dataset consists of 8,000 images collected from the Flickr website. Each image is accompanied by five different captions, providing a diverse range of descriptions. It is used in the initial model implementation

- `The Flickr30k Dataset`: This larger dataset consists of 30,000 images, each with multiple captions. It is used in the attention-based model implementation.

For this project, the datasets are processed into pairs of image tensors and their corresponding indexed captions. This preprocessing step involves transforming the images into tensors and indexing the captions using a vocabulary built from the training captions.

## Project Structure
The project is structured into several Python scripts:

- `Image_Captioning.ipynb`: Notebook for the initial model using Flickr8k dataset and ResNet101 with LSTM.
- `Image_Captioning_att.ipynb`: Notebook for the advanced model using Flickr30k dataset with attention mechanisms, dataset loaded by `dataloaders.py`.

The initial model workflow is also split into multiple functions to enhance modularity and reusability.
- `arhitecture.py`: Main script for training the initial image captioning model.
- `arhitecture_models.py`: Defines the CNN Encoder and LSTM Decoder models.
- `inference.py`: Script for loading the trained model and performing inference.
- `processing_utils.py`: Utility functions for data processing and transformation.
- `data_loader.py`: Functions for creating data loaders.
- `inference_utils.py`: Functions for model inference and evaluation.

## Requirements
  To run the project with attention mechanisms, ensure you have the following dependencies installed:
  ```
  pip install torch torchvision spacy
  python -m spacy download en_core_web_sm
  ```
  
## Running the Project
**Training the Model:**

The arhitecture.py script contains the main training loop. It initializes the models, loads the data, and trains the model. Run the script as follows:
```
python arhitecture.py
```

**Inference:**

After training, use the inference.py script to generate captions for test images and evaluate the model using BLEU scores.
```
python inference.py
```

## Model Architecture
**Initial model**

The core architecture consists of a CNN encoder and an RNN decoder. The CNN encoder uses a pre-trained ResNet101 model to extract image features, while the LSTM decoder generates captions based on these features.

- **CNN Encoder:** Extracts features from images using a pre-trained ResNet101 model.
- **LSTM Decoder:** Generates captions from the encoded image features.

**Attention Model**

- **CNN Encoder**: The CNN encoder extracts features from the image.
- **Attention Mechanism**: The attention layer computes attention weights and context vectors.
- **LSTM Decoder**: The LSTM decoder generates captions using the context vectors and word embeddings.


## Detailed Workflow

- `Image Input`: An image is provided as input.
- `Image Transformation`: The image is transformed into a tensor.
- `CNN (Convolutional Neural Network)`: The tensor is passed through a CNN to extract image features.
- `Attention Mechanism`: Computes attention weights and context vectors from the image features.
- `Linear Layer`: The extracted features are processed through a linear layer to obtain an embedded image feature vector.
- `LSTM (Long Short-Term Memory)`: The embedded image vector is fed into an LSTM for sequential processing.
- `Word Embeddings`: The LSTM outputs are combined with word embeddings to generate a sequence of words.

## Training Details
**Hyperparameters:**

- Learning Rate: `3e-4`
- Batch Size: `64`
- Number of Epochs: `30`
- Embedding Dimension: `512`
- Hidden Dimension: `512`
- Number of LSTM Layers: `1`
- Dropout Probability: `0.4/0.5`

**Optimization:**

- Optimizer: `Adam`
- Scheduler: `StepLR`

## BLEU Score Evaluation
The model's performance is evaluated using BLEU (Bilingual Evaluation Understudy) scores. In this context, BLEU scores are used to evaluate how well the generated captions match the reference captions in the test dataset.

The BLEU score calculation is implemented in the `inference_utils.py` script, and both BLEU-1 and BLEU-2 scores are reported:

- BLEU-1: Measures the precision of unigrams (individual words).
- BLEU-2: Measures the precision of bigrams (pairs of words).

To perform BLEU score evaluation, run the inference script:
```
python inference.py
```

The results with the initial model were:
- BLEU-1: 0.372567
- BLEU-2: 0.190270

The results with the improved attention model were:
- BLEU-1: 0.380143
- BLEU-2: 0.204234

## Future work
Steps for additional improvement would be exploring the hyperparameters, different architectures, and training with more epochs. Experimenting with [attention mechanisms](https://arxiv.org/pdf/1502.03044) and more complex RNN variants could also enhance performance.

## Contributors
- Manolache Vlad George
- Balan Petru
- Eremia Silvia

Xarxes Neuronals i Aprenentatge Profund
Grau de __Artificial Intelligence__, UAB, 2024
