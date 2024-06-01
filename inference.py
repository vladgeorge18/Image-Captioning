import torch
import pickle
import torch.nn as nn
from torchvision import transforms
from sklearn.model_selection import train_test_split
import torchvision.models as models
import processing_utils as utils
import data_loader
import inference_utils
import arhitecture_models


def load_model(model_save_path, e_model_save_path, d_model_save_path, vocab_path, device):
    # Load the vocabulary
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    vocab_dim = len(vocab.word2ind)
    cnn_output_size = 2048
    embedding_dim = 512
    hidden_dim = 512
    n_layers = 1
    
    # Initialize models
    resnet101 = models.resnet101(pretrained=False)
    resnet101.fc = nn.Identity()
    encoder = arhitecture_models.CNN_Encoder(resnet101, cnn_output_size, embedding_dim)
    decoder = arhitecture_models.LSTM_Deconder(vocab_dim, embedding_dim, hidden_dim, n_layers, drop_prob=0.25)
    model = arhitecture_models.CNNLSTMModel(encoder, decoder)
    
    # Load model weights
    model.load_state_dict(torch.load(model_save_path))
    encoder.load_state_dict(torch.load(e_model_save_path))
    decoder.load_state_dict(torch.load(d_model_save_path))
    
    model.to(device)
    encoder.to(device)
    decoder.to(device)
    
    return model, encoder, decoder, vocab

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Define paths
    model_save_path = "./models/rnnlstm_model-10.pth"
    e_model_save_path = "./models/encoder_model-10.pth"
    d_model_save_path = "./models/decoder_model-10.pth"
    
    vocab_path = "vocab.pkl"
    dataset_path = "flickr8_data/flickr8k"
    dataset_images_path = dataset_path + "/Images/"
    image_tensors_path = "image_tensors.pkl"
    
    # Load the model, encoder, decoder, and vocabulary
    model, encoder, decoder, vocab = load_model(model_save_path, e_model_save_path, d_model_save_path, vocab_path, device)
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to 224x224
        transforms.ToTensor(),          # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
    ])

    # Load or create image tensors dictionary
    image_tensors_dict = utils.load_or_create_image_tensors_dict(image_tensors_path, dataset_images_path, transform)

    # Get image-caption dictionary
    images_captions_dict = utils.get_image_caption_dict(dataset_path)

    # Split data
    validation_split = 0.2
    image_filenames = list(images_captions_dict.keys())
    _, image_filenames_test = train_test_split(image_filenames, test_size=validation_split, random_state=1)
    X_test, y_test_raw = utils.get_images_labels(image_filenames_test, image_tensors_dict, images_captions_dict)

    y_test_filtered = [utils.filter_text(y) for y in y_test_raw]
    y_test = [vocab.numericalize(y) for y in y_test_filtered]

    utils.pad_captions(vocab, y_test)

    test_loader = data_loader.create_dataloaders(X_test, y_test, batch_size=128, num_workers=4, shuffle=False)

    # Perform prediction and BLEU score evaluation
    inference_utils.predict(model, encoder, decoder, vocab, test_loader, device)
    inference_utils.bleu_score_evaluation(model, encoder, decoder, vocab, test_loader, device)

if __name__ == "__main__":
    main()
