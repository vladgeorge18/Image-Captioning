import spacy
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import random
from matplotlib import pyplot as plt

spacy_eng = spacy.load('en_core_web_sm')

num_workers = 2
batch_first = True
pin_memory = True
shuffle = True

class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k,v in self.itos.items()}
    
    def __len__(self):
        return len(self.itos)
  
    @staticmethod
    def tokenize(text):
        if not isinstance(text, str):
            text = str(text)
        return [token.text.lower() for token in spacy_eng.tokenizer(text) if token.is_alpha]
  
    def build_vocab(self, sent_list):
        freqs = {}
        idx = 4
        for sent in sent_list:
            sent = str(sent)
            for word in self.tokenize(sent):
                if word not in freqs:
                    freqs[word] = 1
                else:
                    freqs[word] += 1

                if freqs[word] == self.freq_threshold:
                    self.itos[idx] = word
                    self.stoi[word] = idx
                    idx += 1

    def numericalize(self, sents):
        tokens = self.tokenize(sents)
        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] 
                for token in tokens]
        
# Custom dataset class for Flickr30k
class Flickr30kDataset(Dataset):
    def __init__(self, image_names, captions, vocab, root_dir,transform=None):
        self.root_dir = root_dir
        self.image_names = image_names
        self.captions = captions
        self.transform = transform        
        
        self.vocab = vocab

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        caption = self.captions[idx]
        
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])
        
        return image, torch.tensor(numericalized_caption)
    
    
# Padding the captions according to the largest caption in the batch
class CaptionPadder:
    def __init__(self, pad_seq, batch_first=False):
        self.pad_seq = pad_seq
        self.batch_first = batch_first
  
    def __call__(self, batch):
        imgs = [itm[0].unsqueeze(0) for itm in batch]
        imgs = torch.cat(imgs, dim=0)

        target_caps = [itm[1] for itm in batch]
        target_caps = pad_sequence(target_caps, batch_first=self.batch_first,
                                   padding_value=self.pad_seq)
        return imgs, target_caps
    
    
def get_loaders(batch_size, dataset='flickr8k', root_folder='flickr8k/'):
    image_names = set()
    
    # Get names
    if dataset == 'flickr30k':
        captions_file = os.path.join(root_folder, 'flickr30k_images/results.csv')
        root_folder = os.path.join(root_folder, 'flickr30k_images/flickr30k_images/')
    
        df = pd.read_csv(captions_file, delimiter='|')
    elif dataset == 'flickr8k':
        captions_file = os.path.join(root_folder, 'captions.txt')
        root_folder = os.path.join(root_folder, 'Images/')
    
        df = pd.read_csv(captions_file, delimiter=',')
        df.rename(columns={'image': 'image_name', 'caption': ' comment'}, inplace=True)
        
    for i in range(len(df)):
        image_names.add(df['image_name'][i])
                    
    # Convert the set to list
    image_names = list(image_names)    
                
    # Shuffle and split the dataset
    random.shuffle(image_names)
    train_size = int(0.8 * len(image_names))
    train_img_set = image_names[:train_size]
    test_img_set = image_names[train_size:]
    vocab = Vocabulary(5)
    # Get captions for the images
    train_captions = []
    test_captions = []

    # Boolean mask
    is_train = df['image_name'].isin(train_img_set)

    # Split the data
    train_data = df.loc[is_train, ['image_name', ' comment']]
    test_data = df.loc[~is_train, ['image_name', ' comment']]

    train_img_names = train_data['image_name'].tolist()
    train_captions = train_data[' comment'].tolist()

    test_img_names = test_data['image_name'].tolist()
    test_captions = test_data[' comment'].tolist()

    vocab.build_vocab(train_captions)
    # Images normalized
    transforms = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = Flickr30kDataset(train_img_names, train_captions,vocab, root_dir=root_folder, transform=transforms) #train_dataset.vocab
    test_dataset = Flickr30kDataset(test_img_names, test_captions,vocab, root_dir=root_folder,  transform=transforms)

    pad_idx = train_dataset.vocab.stoi["<PAD>"]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers, collate_fn=CaptionPadder(pad_idx, batch_first))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers, collate_fn=CaptionPadder(pad_idx, batch_first))
    
    return train_loader, test_loader, vocab
