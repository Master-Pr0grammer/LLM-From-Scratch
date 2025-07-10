import torch, random
from tokenizer import Tokenizer

class Data():
    def __init__(self, train_file_name:str, test_file_name:str, tokenizer:Tokenizer, sample_data=False):

        # get text data
        self.train_text = open(train_file_name, encoding='utf8').read().lower()
        self.test_text = open(test_file_name, encoding='utf8').read().lower()
        
        self.tokenizer = tokenizer
        self._encoded_train_data = self.tokenizer.encode(self.train_text)
        self._encoded_test_data = self.tokenizer.encode(self.test_text)

        if sample_data:
            print('character list:')
            print(f'\t{self.tokenizer.vocab_list}\n')
    
    #get sample batch from training set
    def get_random_train_sample(self, sample_len, num_samples):
        #get sample
        x=[]
        y=[]
        for _ in range(num_samples):
            index = random.randint(0, len(self.train_text) - sample_len - 1 - num_samples)
            x.append(self._encoded_train_data[index:sample_len + index])
            y.append(self._encoded_train_data[index + 1:sample_len + index + 1])
            index+=1

        y = torch.tensor(y)
        x = torch.tensor(x)

        return x, y
    
    def get_test_sample(self, index, sample_len, num_samples):
        #get sample

        x=[]
        y=[]
        for _ in range(num_samples):
            x.append(self._encoded_test_data[index:sample_len + index])
            y.append(self._encoded_test_data[index + 1:sample_len + index + 1])
            index+=1

        y = torch.tensor(y)
        x = torch.tensor(x)

        return x, y
    
    def get_random_test_sample(self, sample_len, num_samples):
        #get sample

        x=[]
        y=[]
        for _ in range(num_samples):
            index = random.randint(0, len(self.test_text) - sample_len - 1 - num_samples)
            x.append(self._encoded_test_data[index:sample_len + index])
            y.append(self._encoded_test_data[index + 1:sample_len + index + 1])
            index+=1

        y = torch.tensor(y, dtype=int)
        x = torch.tensor(x, dtype=int)

        return x, y
    
    def calc_accuracy(self, actual, target):
        print(actual.size(), target.size())
        acc=torch.mean((actual.argmax(-1)==target.argmax(-1)).float())
        return acc.item()
        


if __name__ == '__main__':
    import json
    file = open('vocab_chars.json', 'r')
    vocab = json.loads(file.read())
    file.close()
    
    tokenizer = Tokenizer(vocab)
    data = Data('tbbt_train.txt', 'tbbt_test.txt', tokenizer, 8, sample_data=True)

    #compress x to one vector
    a=torch.tensor([])
    print(data.num_char)

    x,y=data.get_train_sample(0, 2)
    print(x,'\n',y)
    print(x.shape)
    print(y.shape)
