import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms, models
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

import processing_utils as utils
import data_loader
import arhitecture_models
import train_model
import inference_utils

assert torch.cuda.is_available(), "GPU is not enabled"

# Use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

dataset_path = "flickr8_data/flickr8k"
dataset_images_path = dataset_path + "/Images/" 
image_tensors_path = "image_tensors.pkl"

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224
    transforms.ToTensor(),          # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

images_captions_dict = utils.get_image_caption_dict(dataset_path)
image_tensors_dict = utils.load_or_create_image_tensors_dict(image_tensors_path, dataset_images_path, transform)

validation_split = 0.2
image_filenames = list(images_captions_dict.keys())
image_filenames_train, image_filenames_test = train_test_split(image_filenames, test_size=validation_split, random_state=1)

X_train, y_train_raw = utils.get_images_labels(image_filenames_train, image_tensors_dict, images_captions_dict )
X_test, y_test_raw = utils.get_images_labels(image_filenames_test, image_tensors_dict, images_captions_dict)

X_train, y_train_raw = utils.get_shuffled_data(X_train, y_train_raw)
X_test, y_test_raw = utils.get_shuffled_data(X_test, y_test_raw)

y_train_filtered = [utils.filter_text(y) for y in y_train_raw] 
y_test_filtered = [utils.filter_text(y) for y in y_test_raw] 

vocab_path = "vocab.pkl"
# Delete vocab.pkl if changing treshold, otherwise change will not be applied
freq_threshold = 7
vocab = utils.load_or_create_vocab(vocab_path, y_train_filtered, freq_threshold)

y_train = [vocab.numericalize(y) for y in y_train_filtered]
y_test = [vocab.numericalize(y) for y in y_test_filtered]

utils.pad_captions(vocab, y_train)
utils.pad_captions(vocab, y_test)

print(y_train_filtered[2])
print(y_train[2])
print([vocab.ind2word[word] for word in y_train[2]])

train_loader = data_loader.create_dataloaders(X_train, y_train,batch_size=128, num_workers=4, shuffle=True)
test_loader = data_loader.create_dataloaders(X_test, y_test, batch_size=128, num_workers=4, shuffle=False)

cnn_output_size = 2048
embedding_dim = 512
vocab_dim = len(vocab.word2ind)
hidden_dim =512
n_layers = 1

inception_v3 = models.inception_v3(weights='DEFAULT')
resnet50 = models.resnet50(pretrained=True)
resnet101 = models.resnet101(pretrained=True)

resnet50.fc = nn.Identity()
resnet101.fc = nn.Identity()
inception_v3.fc = nn.Identity()

utils.freeze_model(inception_v3)
utils.freeze_model(resnet101)
utils.freeze_model(resnet50)

encoder = arhitecture_models.CNN_Encoder(resnet101,cnn_output_size,embedding_dim)
decoder = arhitecture_models.LSTM_Deconder(vocab_dim,embedding_dim,hidden_dim,n_layers,drop_prob=0.25)
model = arhitecture_models.CNNLSTMModel(encoder,decoder)
model.to(device)

params_to_update = []
for name, param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)

lr = 1e-3
optimizer = optim.Adam(params_to_update, lr=lr, weight_decay=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1, last_epoch=-1)
criterion = nn.CrossEntropyLoss()

num_epoch = 10

losses = []
for epoch in range(num_epoch):

    train_loss = train_model.train(epoch, criterion, model, optimizer, train_loader, device)
    scheduler.step()
    losses.append(train_loss)
    print(losses[-1])

     # Save the model every 10 epochs
    if (epoch + 1) % 10 == 0:
        model_save_path = f"./models/rnnlstm_model-epoch{epoch+1}.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model weights saved to {model_save_path}")

        e_model_save_path = f"./models/encoder_model-epoch{epoch+1}.pth"
        torch.save(encoder.state_id(), e_model_save_path)
        print(f"Encoder weights saved to {e_model_save_path}")

        d_model_save_path = f"./models/decoder_model-epoch{epoch+1}.pth"
        torch.save(decoder.state_dict(), d_model_save_path)
        print(f"Decoder weights saved to {d_model_save_path}")

#Load model
# model_save_path = "rnnlstm_model-10.pth"
# e_model_save_path = "encoder_model-10.pth"
# d_model_save_path = "decoder_model-10.pth"
# model.load_state_dict(torch.load(model_save_path))
# decoder.load_state_dict(torch.load(d_model_save_path))
# encoder.load_state_dict(torch.load(e_model_save_path))

# inference_utils.predict(model, encoder, decoder, vocab, test_loader, device)
# inference_utils.bleu_score_evaluation(model, encoder, decoder, vocab, test_loader, device)
