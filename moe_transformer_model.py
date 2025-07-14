import torch
import torch.nn as nn

#Transformer network implementation
class self_attention_head(nn.Module):
    def __init__(self, embedding_dim, ctx_window_length, dropout_rate, head_size=16):
        super().__init__()
        T, C = ctx_window_length, embedding_dim
        # B = the number of samples in each batch (Batch Size: ctx_window_length)
        # T = the number of items in the time dependant series, (number of characters: num_char)
        # C = the number of channels in each time stamp (embedding dimension of time stamp: embedding_dim)
        self.head_size = head_size
        
        #key and query layers
        self.key = nn.Linear(C, head_size, bias = False)
        self.query = nn.Linear(C, head_size, bias = False)
        self.value = nn.Linear(C, head_size, bias = False)

        self.register_buffer('tril', tensor=torch.tril(torch.ones(T, T)))

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        #X = (B, T, C)
        B, T, C = x.shape

        #Get Keys and Qeuries
        k = self.key(x) #(B, T, head Size)
        q = self.query(x) #(B, T, head Size)

        #get averaged weight matrix
        #weights = q @ k.transpose(-2, -1) * C ** -0.5 # (B,T,head_size) @ (B,head_size,T) = (B,T,T)
        scale = self.head_size ** -0.5
        weights = q @ k.transpose(-2, -1) * scale
        weights = weights.masked_fill(self.tril == 0, float('-inf')) # optional, when eneabled, it prevents past nodes from accessing future nodes, EX w,o,r,d char w wont have info about o
        weights = nn.functional.softmax(weights, dim= -1)
        weights = self.dropout(weights)

        #get output
        v = self.value(x)
        out = weights @ v

        return out

class multi_head_attention(nn.Module):
    def __init__(self, num_heads, embedding_dim, ctx_window_length, dropout_rate, head_size=16):
        super().__init__()
        
        if num_heads * head_size != embedding_dim:
            raise ValueError("num_heads * head_size must equal embedding_dim")

        self.heads = nn.ModuleList([self_attention_head(embedding_dim, ctx_window_length, dropout_rate, head_size=head_size) for i in range(num_heads)])
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class Attention_Block(nn.Module):
    def __init__(self, embedding_dim, ctx_window_length, num_heads, dropout_rate):
        super().__init__()
        self.head_size = embedding_dim // num_heads
        if self.head_size * num_heads != embedding_dim:
            raise ValueError("embedding dimenstion needs to be divisible by number of heads")
        self.attention = multi_head_attention(num_heads, embedding_dim, ctx_window_length, dropout_rate, head_size=self.head_size)

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

        #feed forward section
        self.ffw = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x):
        x = x + self.attention(self.layer_norm1(x))
        x = x + self.ffw(self.layer_norm2(x))
        return x


class Router(nn.Module):
    def __init__(self, embedding_dim, num_experts):
        super().__init__()
        
        self.fc0 = nn.Linear(embedding_dim, num_experts)
        
    def forward(self, x):
        x = self.fc0(torch.sum(x, dim=1))
        x = nn.functional.softmax(x)
        x = torch.argmax(x, dim=1)
        
        return x



# Define the neural network model
class MOE_Transformer(nn.Module):
    def __init__(
        self, 
        vocab_size:int, 
        embedding_dim:int, 
        ctx_window_length:int, 
        num_attention_heads:int, 
        num_attention_blocks:int, 
        learning_rate:float, 
        num_experts:int,
        dropout_rate:float=0.0,
        ):
        
        super(MOE_Transformer, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.ctx_window_length = ctx_window_length
        self.num_attention_heads = num_attention_heads
        self.num_attention_blocks = num_attention_blocks
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        
        # meta data:
        self.training_state:dict = None
        

        self.char_emb = nn.Embedding(vocab_size, embedding_dim)
        self.pos_emb = nn.Embedding(self.ctx_window_length, embedding_dim)
        
        self.router = Router(embedding_dim, num_experts)

        #attention layers
        self.experts = [nn.Sequential(*[Attention_Block(embedding_dim, ctx_window_length, num_attention_heads, dropout_rate) for _ in range(num_attention_blocks)]) for _ in range(num_experts)]
        #self.attention_blocks = nn.Sequential(*[Attention_Block(embedding_dim, ctx_window_length, num_attention_heads, dropout_rate) for _ in range(num_attention_blocks)])
        self.layer_norm = nn.LayerNorm(embedding_dim)

        #Layers
        self.fc1 = nn.Linear(embedding_dim, vocab_size)

        #better weight initialization
        self.apply(self._init_weights)

        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer= torch.optim.AdamW(self.parameters(), lr=learning_rate)
        
        #Print parameters
        #print(sum(p.numel() for p in self.parameters())/1E6, 'Million Parameters')
        
    def save_checkpoint(self, filename='Checkpoint.pth', training_state:dict|None=None):
        checkpoint = {
            'model_state': self.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'hyperparams': {
                'vocab_size': self.vocab_size,
                'ctx_window_length': self.ctx_window_length,
                'embedding_dim': self.embedding_dim,
                'num_attention_blocks': self.num_attention_blocks,
                'num_attention_heads': self.num_attention_heads,
                'learning_rate': self.learning_rate,
                'dropout_rate': self.dropout_rate
            },
            'training_state': training_state # training meta data
        }
        
        torch.save(checkpoint, filename)

    def create_from_checkpoint(path, device='cpu'):
        checkpoint:dict = torch.load(path)
        h = checkpoint['hyperparams']
        model = MOE_Transformer(**h)
        model.load_state_dict(checkpoint['model_state'])
        
        if checkpoint.get('training_state'):
            model.training_state = checkpoint.get('training_state')
            
        model.to(device)

        # Recreate optimizer AFTER loading model
        model.optimizer = torch.optim.AdamW(model.parameters(), lr=h['learning_rate'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        return model
    
    def get_parameter_count(self):
        """Returns parameter count in millions"""
        return sum(p.numel() for p in self.parameters())/1E6

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        #run through embedding layer
        B, T = x.size()
        C = self.vocab_size
        
        char_emb = self.char_emb(x)
        pos_emb = self.pos_emb(torch.arange(self.ctx_window_length, device=next(self.parameters()).device)) # Use current model device
        x = char_emb + pos_emb
        
        experts = self.router(x)
        outputs = []
        
        i = 0
        for idx in experts:
            outputs.append(self.experts[0](x[i].unsqueeze(0)))
            i+=1
        x = torch.stack(outputs)
        #attention_blocks = self.experts[self.router(x)[:,-1]]

        #x = attention_blocks(x)
        x = self.layer_norm(x)
        #x = x.view((-1, chunk_size * embedding_dim))

        #run through reshaping layer and softmax
        logits = self.fc1(x)

        if targets is None:
            #logits = self.softmax(output)
            return logits
        else:
            logits_flat  = logits.view(B * T, C)     # (64*32, 55)
            targets_flat = targets.view(B * T)       # (64*32)
            loss = self.loss_func(logits_flat, targets_flat)
            
            return logits, loss
        
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.ctx_window_length:]
            # get the predictions
            logits = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            log_probs = nn.functional.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(log_probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx