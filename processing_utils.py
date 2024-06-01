import re
from PIL import Image
import os
import random
import pickle
import vocabulary

def get_preprocessed_caption(caption):    
    caption = re.sub(r'\s+', ' ', caption)
    caption = caption.strip()
    caption = "<start> " + caption + " <end>"
    return caption

def get_image_caption_dict(dataset_path):
    images_captions_dict = {}

    with open(dataset_path + "/captions.txt", "r") as dataset_info:
        next(dataset_info) # Omit header: image, caption

        for info_raw in list(dataset_info):
            info = info_raw.split(",")
            image_filename = info[0]
            inf = ''
            for i in info[1:]:
                inf +=i+" "
            caption = get_preprocessed_caption(inf)

            if image_filename not in images_captions_dict.keys():
                images_captions_dict[image_filename] = [caption]
            else:
                images_captions_dict[image_filename].append(caption)

    return images_captions_dict

def get_image_tensor_dict(dataset_images_path, transform):
    image_tensors = {}
# Iterate over all files in the folder
    for filename in os.listdir(dataset_images_path):
        if filename.endswith('.jpg'): 
        # Load the image
            image_path = os.path.join(dataset_images_path, filename)
            image = Image.open(image_path)
        # Apply the transformations
            image_tensor = transform(image)
        
            image_tensors[filename] = image_tensor
    return image_tensors

def load_or_create_image_tensors_dict(image_tensors_path, dataset_images_path, transform):
    if os.path.exists(image_tensors_path):
        # Load the image tensors dictionary from the file
        with open(image_tensors_path, 'rb') as f:
            image_tensors_dict = pickle.load(f)
        print(f"Image tensors dictionary loaded from {image_tensors_path}")
    else:
        # Create a new image tensors dictionary
        image_tensors_dict = get_image_tensor_dict(dataset_images_path, transform)
        # Save the image tensors dictionary to a file
        with open(image_tensors_path, 'wb') as f:
            pickle.dump(image_tensors_dict, f)
        print(f"Image tensors dictionary saved to {image_tensors_path}")
    
    return image_tensors_dict

def get_images_labels(image_filenames,image_tensors, images_captions_dict):
    images = []
    labels = []
    
    for image_filename in image_filenames:
        image = image_tensors[image_filename]
        captions = images_captions_dict[image_filename]

        # Add one instance per caption
        for caption in captions:
            images.append(image)
            labels.append(caption)
            
    return images, labels

def get_shuffled_data(X, y):
    temp = list(zip(X, y))
    random.shuffle(temp)
    res1, res2 = zip(*temp)
    # res1 and res2 come out as tuples, and so must be converted to lists
    X, y = list(res1), list(res2)
    return X,y

def filter_text(text):
    pattern = r'[!"#$%&()*+.,\-/:;=?@[\]^_`{|}~]'
    pattern_spaces = r'\s+'

    filtered_text = re.sub(pattern, '', text)
    filtered_text = re.sub(pattern_spaces, ' ', filtered_text)
    return filtered_text

def pad_captions(vocab, y_train):
    max_len = max(len(seq) for seq in y_train)
    for y in y_train:
        while len(y) < max_len:
            y.append(vocab.word2ind['<pad>'])

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

def load_or_create_vocab(vocab_path, y_data, freq_threshold):
    if os.path.exists(vocab_path):
        # Load the vocabulary from the file
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        print(f"Vocabulary loaded from {vocab_path}. Size: {len(vocab.word2ind)}")
    else:
        # Create a new vocabulary
        vocab = vocabulary.Vocabulary(freq_threshold)
        vocab.build_vocab(y_data)
        print(f"Vocab size with threshold {freq_threshold}:", len(vocab.word2ind))
        
        # Save the vocabulary to a file
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab, f)
        print(f"Vocabulary saved to {vocab_path}")
    
    return vocab