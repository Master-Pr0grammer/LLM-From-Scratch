import torch, random
from tokenizer import Tokenizer

class Data():
    def __init__(self, train_file_name:str, test_file_name:str, tokenizer:Tokenizer, ctx_size:int, sample_data=False):

        # get text data
        self.train_text = open(train_file_name, encoding='utf8').read().lower()
        self.test_text = open(test_file_name, encoding='utf8').read().lower()
        
        self.tokenizer = tokenizer
        self._encoded_train_data = self.tokenizer.encode(self.train_text)
        self._encoded_test_data = self.tokenizer.encode(self.test_text)

        self.ctx_size = ctx_size

        if sample_data:
            print('character list:')
            print(f'\t{self.tokenizer.vocab_list}\n')

    #helper function used during initialization in order to convert raw text file to data
    def generate_data(self, text, sample_data=False):
        x = []
        y = []

        # take text, and get chunks of char, and then one hot encode all chars for one data point
        for i in range(len(text)-self.ctx_size-1):

            # get encoded data
            x.append(self.tokenizer.encode(text[i:self.ctx_size+i]))
            y.append(self.tokenizer.decode(text[i+self.ctx_size])[0])

        if sample_data:
            print('X and Y index encodings:')
            print('\tX:', x[0:3])
            print('\tY:', y[0:3])
            print()

        # one hot encode data
        x = torch.tensor(x)
        y = torch.tensor(y)

        if sample_data:
            print('X and Y one hot encodings:')
            print('\tX:', x[0:3])
            print('\tY:', y[0:3])
            print()

            print('X and Y one hot decodings:')
            for j in range(3):
                print(f'\tX[{j}]: "', ''.join([self.decode_dict[i.item()] for i in x[j].argmax(dim=-1)]).replace('\n', ' '), f'", Y[{j}]: "', self.decode_dict[y[j].argmax(-1).item()].replace('\n', ' '), '"', sep='')

        #save data
        return x, y

    def compress_x(self, x):
        x= x.view((-1, self.ctx_size * self.num_char))
        return x
    
    #get sample batch from training set
    def get_train_sample(self, index, num_samples):
        #get sample
        x=[]
        y=[]
        for _ in range(num_samples):
            x.append(self._encoded_train_data[index:self.ctx_size + index])
            y.append(self._encoded_train_data[index + 1:self.ctx_size + index + 1])
            index+=1

        y = torch.tensor(y)
        x = torch.tensor(x)

        return x, y
    
    def get_test_sample(self, index, num_samples):
        #get sample

        x=[]
        y=[]
        for _ in range(num_samples):
            x.append(self._encoded_test_data[index:self.ctx_size + index])
            y.append(self._encoded_test_data[index + 1:self.ctx_size + index + 1])
            index+=1

        y = torch.tensor(y)
        x = torch.tensor(x)

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
