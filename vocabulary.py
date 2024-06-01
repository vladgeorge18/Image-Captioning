from collections import Counter

class Vocabulary:
    def __init__(self,freq_threshold):
        #setting the pre-reserved tokens int to string tokens
        self.ind2word = {0:"<pad>",1:"<start>",2:"<end>",3:"<unk>"}
        
        #string to int tokens
        #its reverse dict self.itos
        self.word2ind = {v:k for k,v in self.ind2word.items()}
        
        self.freq_threshold = freq_threshold
        
    def __len__(self): return len(self.itos)
    
    @staticmethod
    def tokenize(text):
        return [token.lower() for token in text.split()]
    
    def build_vocab(self, sentence_list):
        frequencies = Counter()
        idx = 4
        
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1
                
                #add the word to the vocab if it reaches minimum frequecy threshold
                if frequencies[word] == self.freq_threshold and word not in self.word2ind:
                    self.word2ind[word] = idx
                    self.ind2word[idx] = word
                    idx += 1
    
    def numericalize(self,text):
        # change word with index
        tokenized_text = self.tokenize(text)
        return [ self.word2ind[token] if token in self.word2ind else self.word2ind["<unk>"] for token in tokenized_text ]  