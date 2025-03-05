import torch 
import torch.nn as nn
import math
import typing
from typing import List, Union
import pandas as pd
import torch.nn.functional as F
import numpy as np


def custom_dataloader(dataset, batch_size, context_len, device,):
        # Verifique o tamanho total do dataset
    dataset_size = len(dataset)
    # print('dataset is on ', dataset.data.get_device())
    max_iter = dataset_size // batch_size
    print(max_iter, 'expected iters with size', batch_size)
    def generator():
        for _ in range(max_iter): 
            # Gere n índices aleatórios no intervalo permitido
            ix = torch.randint(dataset_size - context_len -1, (batch_size,))
            x = torch.stack([dataset[i:i+context_len] for i in ix])
            y = torch.stack([dataset[i+1:i+context_len+1] for i in ix])

            # x, y = x.to(device), y.to(device)

            yield x, y
        
    return generator()

def one_hot_encoding(token_ids: torch.Tensor, vocab_size: int):
    """
    Cria uma matriz de one-hot encoding para múltiplos token_ids.

    Args:
        token_ids (torch.Tensor): Tensor 1D contendo os índices dos tokens a serem marcados com 1.
        vocab_size (int): Número total de posições possíveis.

    Returns:
        torch.Tensor: Matriz de shape (len(token_ids), vocab_size) onde cada linha é um vetor one-hot.
    """
    batch_size = token_ids.shape[0]
    one_hot_matrix = torch.zeros((batch_size, vocab_size), dtype=torch.float32)
    one_hot_matrix[torch.arange(batch_size), token_ids] = 1.0  # Define 1 nas posições dos token_ids
    return one_hot_matrix

class PositionStateTokenizer(nn.Module):
    def __init__(self, positions: List[str], vocab_size:int) -> None:
        super().__init__()
        self.positions = positions.copy()
        assert vocab_size == len(set(positions))
        self.vocab_size = len(set(positions))
        self.string_to_id = {p:i for i, p in enumerate(positions)}
        self.id_to_string = {i:p for i, p in enumerate(positions)}

    def forward(self, x: List[str], get_tensor: bool=True) -> Union[torch.tensor, list]:
        if get_tensor:
            return torch.tensor([self.string_to_id[p] for p in x], dtype=torch.long)
        else:
            return [self.string_to_id[p] for p in x]
    def reverse(self, x: List[int]) -> list:
        return [self.id_to_string[i] for i in x]

class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model:int,) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class UnScaledInputEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model:int,) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
    def forward(self, x):
        return self.embedding(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_length: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10_000)/d_model))

        pe[:, 0::2] = torch.sin(position * div_term) # even positions
        pe[:, 1::2] = torch.cos(position * div_term) # odd positions

        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x:torch.tensor) -> torch.tensor:
        return x + self.pe[:, :x.size(1)]
    
class AttentionHead(nn.Module):
    def __init__(self, head_dim: int) -> None:
        super().__init__()
        self.head_dim = head_dim

        # k, q, v
        self.key = nn.Linear(head_dim, head_dim, bias=False)
        self.query = nn.Linear(head_dim, head_dim, bias=False)
        self.value = nn.Linear(head_dim, head_dim, bias=False)
    
    def linear_projection(self, keys, queries, values) -> Union[torch.tensor, torch.tensor, torch.tensor]:
        return self.key(keys), self.query(queries), self.value(values)
    
    def compute_attention(self, k, q, v, mask:Union[torch.tensor, None]) -> torch.tensor:
        attention_scores = q @ k.transpose(-2, -1) / (self.head_dim**0.5)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=-1)
        output = attention_weights @ v
        return output

    def forward(self, keys, queries, values, mask:Union[torch.tensor, None] = None) -> torch.tensor:
        k, q, v = self.linear_projection(keys, queries, )
        return self.compute_attention(k, q, v, mask)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.heads = nn.ModuleList(
            [AttentionHead(self.d_model) for _ in range(num_heads)]
        )
        self.output_layer = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, keys, queries, values, mask:Union[torch.tensor, None] = None) -> torch.tensor:
        output = []
        for i, h in enumerate(self.heads):
            k, q, v = h.linear_projection( keys, queries, values,)
            splited_k = k[:, :, i*self.head_dim:(i+1)*self.head_dim]
            splited_q = q[:, :, i*self.head_dim:(i+1)*self.head_dim]        
            splited_v = v[:, :, i*self.head_dim:(i+1)*self.head_dim]
            head_output = h.compute_attention(splited_k, splited_q, splited_v, mask)
            # print('i', i, splited_k.shape, head_output.shape)
            output.append(head_output)
        # lin_projs [b, h, seq, d_model]
        output = torch.cat([out for out in output], dim=-1)
        output = self.output_layer(output)
        return output

class FeedForwardLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation_layer = nn.ReLU()
    
    def forward(self, x) -> torch.tensor:
        x = self.linear1(x)
        x = self.activation_layer(x)
        x = self.linear2(x)
        return x

class EncoderTransformerBody(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.norm_layer1 = nn.LayerNorm(normalized_shape=d_model)
        self.norm_layer2 = nn.LayerNorm(normalized_shape=d_model)
        self.mhsa = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.ffwd = FeedForwardLayer(d_model=d_model, d_ff=d_ff)

    def forward(self, key, query, value,  mask: Union[torch.tensor, None] = None) -> torch.tensor:
        # sublayer mhsa
        att_weigths = self.mhsa(key, query, value, mask)
        att_weigths = self.dropout(att_weigths)
        # first norm
        out = self.norm_layer1(key + att_weigths)

        # sublayer ffwd
        out = self.ffwd(out)
        out = self.dropout(out)
        # second norm
        out = self.norm_layer2(out)

        return out
    
class DecoderTransformerBody(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, context_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.context_len = context_len
        self.dropout = nn.Dropout(dropout)
        self.norm_layer1 = nn.LayerNorm(normalized_shape=d_model)
        self.norm_layer2 = nn.LayerNorm(normalized_shape=d_model)
        self.mhsa = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.ffwd = FeedForwardLayer(d_model=d_model, d_ff=d_ff)

        # tril
        self.register_buffer('tril', torch.tril(torch.ones(self.context_len, self.context_len)))

    def forward(self, x) -> torch.tensor:
        # decoder mask 
        mask = self.tril[:self.context_len, :self.context_len]

        # sublayer mhsa
        att_weigths = self.mhsa(x, x , x, mask)
        att_weigths = self.dropout(att_weigths)
        # first norm
        out = self.norm_layer1(x + att_weigths)

        # sublayer ffwd
        out = self.ffwd(out)
        out = self.dropout(out)
        # second norm
        out = self.norm_layer2(out)

        return out
    
class CrossAttentionHead(nn.Module):
    def __init__(self, ):
        super().__init__()
    def forward(self, k, q, v, mask:Union[torch.tensor, None] = None) -> torch.tensor:
        infer_head_dim = k.size(-1)
        cross_attention_scores = q @ k.transpose(-2, -1) / (infer_head_dim**0.5)

        if mask is not None:
            cross_attention_scores = cross_attention_scores.masked_fill(mask == 0, float('-inf'))
        cross_attention_weights = F.softmax(cross_attention_scores, dim=-1)
        output = cross_attention_weights @ v
        return output

class MaskedMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, ):
        super().__init__()

class DecoderTransformerHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.linear_vocab = nn.Linear(d_model, vocab_size)

    def forward(self, x, generate) -> torch.tensor:
        x = self.linear_vocab(x) # logits
        if generate:
            return F.softmax(x, dim=-1)
        else:
            return x

class DecoderTransformer(nn.Module):
    def __init__(self
                 , blocks: int
                 , d_model: int
                 , num_heads: int
                 , d_ff: int
                 , context_len: int
                 , dropout:float
                 , vocab_size: int):
        super().__init__()
        self.blocks = blocks
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.context_len = context_len
        self.dropout = dropout
        self.vocab_size = vocab_size
        # toekn embedding
        self.embedding = InputEmbeddings(
            vocab_size=vocab_size, d_model=d_model
        )
        # positional encoding
        self.pos_enc = PositionalEncoding(
            d_model=d_model, max_seq_length=context_len
        )
        # net architecture
        self.decoder_blocks = nn.ModuleList([DecoderTransformerBody(
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            context_len=self.context_len,
            dropout=self.dropout
        ) for _ in range(self.blocks)])

        self.net = nn.Sequential(*self.decoder_blocks)
        self.decoder_head = DecoderTransformerHead(d_model=self.d_model, vocab_size=self.vocab_size)

    def forward(self, x, generate: bool ):
        # x = self.tokenizer(x)
        x = self.embedding(x)
        x = self.pos_enc(x)
        x = self.net(x)
        # probs in vocab_size space
        out = self.decoder_head(x, generate)
        return out

if __name__ == '__main__':
    df = pd.read_csv(r'df_trainset_v0.csv')
    positions = df['position_state'].values
    unique_posisionts = df['position_state'].unique()
    # print(len(positions))
    # print(len(set(positions)), len(unique_posisionts))



    # PARAMETERS
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device: ', DEVICE)
    VOCAB_SIZE = len(unique_posisionts)
    D_MODEL = 512
    MAX_SEQ_LEN = 16
    NUM_HEADS = 2
    D_FF = 2*D_MODEL
    DROPOUT = 0.
    NUM_DECODER_BLOCKS = 4
    BATCH_SIZE = 32
    NUM_EPOCHS = 200
    LEARNING_RATE = 1e-3



    # pos_tokenizer = PositionStateTokenizer(positions = unique_posisionts
    #                                    , vocab_size= VOCAB_SIZE)
    
    # df['tokenized_positions'] = [pos_tokenizer([pos]) for pos in df['position_state'].values]

    # print(df['tokenized_positions'].nunique(), len(df['tokenized_positions']))

    dataloader = custom_dataloader(
        dataset=torch.tensor(df['position_state_id'].values, dtype=torch.long)
        ,batch_size=BATCH_SIZE
        ,context_len=MAX_SEQ_LEN
        , device=DEVICE
        )
    
    for xb, yb in dataloader:
        break
    # print(xb[0, :5], yb[0, :5])

    model = DecoderTransformer(
        blocks=NUM_DECODER_BLOCKS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        context_len=MAX_SEQ_LEN,
        dropout=DROPOUT,
        vocab_size=VOCAB_SIZE
    )


    out = model(xb, generate=False)
    print(model.parameters)
    

    print(out.view(BATCH_SIZE*MAX_SEQ_LEN, VOCAB_SIZE).shape, yb.view(BATCH_SIZE*MAX_SEQ_LEN).shape)














    # pos_embedding = InputEmbeddings(vocab_size=VOCAB_SIZE,
    #                                 d_model=512)
    # pos_encoding = PositionalEncoding(d_model=D_MODEL,
    #                                   max_seq_length=MAX_SEQ_LEN)
    
    # tokenized_positions = pos_tokenizer(positions)
    # embedded_positions = pos_embedding(tokenized_positions)
    # encoded_positions_1 = pos_encoding(embedded_positions[:MAX_SEQ_LEN])
    # encoded_positions_2 = pos_encoding(embedded_positions[1:MAX_SEQ_LEN+1])
    # t = torch.vstack([encoded_positions_1, encoded_positions_2])
    # print(encoded_positions_1.shape)
    # print(encoded_positions_2.shape)
    # encoded_positions_both = pos_encoding(t)
    # head_ = AttentionHead(head_dim=D_MODEL)
    # k, q, v = head_.linear_projection(t, t, t)
    # print(k.shape)
    # multihead = MultiHeadAttention(d_model=D_MODEL, num_heads=NUM_HEADS)
    # out = multihead(encoded_positions_both, encoded_positions_both, encoded_positions_both)
    # ffwd = FeedForwardLayer(d_model=D_MODEL, d_ff=D_FF)
    # out = ffwd(out)
    # out = out + encoded_positions_both
    # print(out.shape)
    # body = TransformerBody(d_model=D_MODEL
    #                        , num_heads=NUM_HEADS
    #                        , d_ff=D_FF
    #                        , dropout=DROPOUT)
    
    # out = body(encoded_positions_both,encoded_positions_both,encoded_positions_both)
    # cross_head = CrossAttentionHead()
    # out = cross_head(out, out, out)
    # print(out.shape)

    # dec_body = DecoderTransformerBody(d_model=D_MODEL
    #                                   , num_heads=NUM_HEADS
    #                                   , d_ff=D_FF
    #                                   , context_len=MAX_SEQ_LEN
    #                                   , dropout=DROPOUT )
    # out = dec_body(encoded_positions_both)
    # print(out.shape)
    # dec_formers = DecoderTransformer(
    #         blocks=NUM_DECODER_BLOCKS,
    #         d_model=D_MODEL,
    #         num_heads=NUM_HEADS,
    #         d_ff=D_FF,
    #         context_len=MAX_SEQ_LEN,
    #         dropout=DROPOUT,
    #         vocab_size=VOCAB_SIZE
    # )
    # out = dec_formers(encoded_positions_both, False)
    # print(out.shape,)
    # out = out[:, -1, :]
    # print(out.shape)
    # print(tokenized_positions[17])
    # target = one_hot_encoding(token_ids=torch.tensor([17, 17]), vocab_size=VOCAB_SIZE)
    # print(target.shape)
    # print(F.cross_entropy(out, target))





    
    