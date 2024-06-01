import torch
import matplotlib.pyplot as plt
import numpy as np
from nltk.translate.bleu_score import corpus_bleu


def predict(model, encoder, decoder, vocab, test_loader, device):
    model.eval()
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            for id in range(1):
                pred = []
                img = data[id].unsqueeze(0)
                features = encoder(img)
                decoder_features=torch.reshape(features,(1,features.shape[0],-1))
                h = decoder_features
                c = decoder_features
                word = target[0][0].unsqueeze(0).unsqueeze(0)
                
                for i in range(len(target[0])-1):
                    y_pred, h, c = decoder(word,h,c)
                    word = torch.argmax(y_pred, dim=2)
                    pred.append(word.item())

                words_pred = []
                words_true = []
                for idx in pred:
                    words_pred.append(vocab.ind2word[idx])

                for idx in target[id]:
                    words_true.append(vocab.ind2word[idx.item()])

                #Print image
                img = img.squeeze(0)
                print(img.shape)
                img = img.permute(1, 2, 0).cpu().numpy()  # Convert from [C, H, W] to [H, W, C] and move to CPU

                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]

                # Denormalize
                img = std * img + mean  
                img = np.clip(img, 0, 1)  

                # Display the image 
                plt.imshow(img)
                plt.title("Visualized Image")
                plt.axis('off')  
                plt.show()

                pred =""
                for word in words_pred:
                    if word =="<pad>":
                        break
                    pred+=word+" "

                true =""
                for word in words_true:
                    if word =="<start>":
                        continue
                    if word =="<pad>":
                        break
                    true+=word+" "
                print("Prediction: ",pred)
                print("Truth :",true)


def bleu_score_evaluation(model, encoder, decoder, vocab, test_loader, device):
    model.eval()
    encoder.eval()
    decoder.eval()
    ct = 0
    with torch.no_grad():
        all_words_pred = []
        all_words_true = []
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            ct+=1
            for id in range(len(data)):
                pred = []
                img = data[id].unsqueeze(0)
                features = encoder(img)
                decoder_features=torch.reshape(features,(1,features.shape[0],-1))
                h = decoder_features
                c = decoder_features
                word = target[0][0].unsqueeze(0).unsqueeze(0)
                
                for i in range(len(target[0])-1):
                    y_pred, h, c = decoder(word,h,c)
                    word = torch.argmax(y_pred, dim=2)
                    pred.append(word.item())

                words_pred = []
                words_true = []
                for idx in pred:
                    if vocab.ind2word[idx] == '<pad>':
                        break
                    words_pred.append(vocab.ind2word[idx])
                all_words_pred.append([words_pred])

                for idx in target[id]:
                    if vocab.ind2word[idx.item()] == '<start>':
                        continue
                    if vocab.ind2word[idx.item()] == '<pad>':
                        break
                    words_true.append(vocab.ind2word[idx.item()])
                all_words_true.append(words_true)
            
        print(all_words_true)
        print(all_words_pred)
        print('BLEU-1: %f' % corpus_bleu(all_words_pred, all_words_true, weights=(1.0, 0, 0, 0)))
        print('BLEU-2: %f' % corpus_bleu(all_words_pred, all_words_true, weights=(0.5, 0.5, 0, 0))) 






















