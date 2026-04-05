# LayerIntegratedGradients with Transformer, Bert (Captum)  

## Overview
This repository applies Layer Integrated Gradients to Transformer and BERT models to interpret classification decisions through token-level attribution. This work is part of Explainable AI (XAI), which aims to make machine learning models more transparent and understandable. By highlighting the contribution of individual tokens, it offers insights into how models arrive at their predictions and helps improve trust and analysis of model behavior.

## Dataset
The experiments are performed on the IMDB dataset, which is commonly used for evaluating sentiment classification models.
```markdown
### Python

url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
dataset = tf.keras.utils.get_file('aclImdb_v1', url, untar=True, cache_dir='', cache_subdir='')
dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb') # '/tmp/.keras/aclImdb'

raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    '/tmp/.keras/aclImdb/train',
    batch_size = batch_size,
    shuffle=False
)
raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    '/tmp/.keras/aclImdb/test',
    shuffle=False,
    batch_size = batch_size)

```

## Model

```markdown
### Python

# 1
class Custom_tf_encoder(nn.Module):
    def __init__(self, used_word, embed_dim, nhead, encoder_layers, max_len=500):
        super().__init__()
        self.embedding_layer = nn.Embedding(used_word, embed_dim)
        self.encoders = nn.ModuleList([nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=(embed_dim*2), dropout=0.2, activation='gelu', batch_first=True) for _ in range(encoder_layers)])
        self.avgpool = nn.AvgPool1d(kernel_size=embed_dim)
        self.fc = nn.Linear(embed_dim, 2)
        self.maxpool = nn.MaxPool2d(kernel_size = (max_len,1))
    
    def mk_padding_mask(self, text):
        # <pad>: 2
        return torch.eq(text, 2)
        
    def forward(self, text):
        x = self.embedding_layer(text)
        padding_mask = self.mk_padding_mask(text).to(x.device)
        for layer in self.encoders:
            x = layer(x, src_key_padding_mask=padding_mask)
        x = self.maxpool(x) # (batch, 1, embed_dim)
        x = x.squeeze(1) # (batch, embed_dim)
        x = self.fc(x)
        
        return x

# 2
from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2
    )
```
