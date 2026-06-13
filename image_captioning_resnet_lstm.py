# ============================================================
# Image Captioning Resnet Lstm
# ============================================================

!wget "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr8k.zip"
!unzip -q flickr8k.zip -d ./flickr8k
!rm flickr8k.zip
--2026-05-01 18:17:53--  https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr8k.zip
Resolving github.com (github.com)... 140.82.116.4
Connecting to github.com (github.com)|140.82.116.4|:443... connected.
HTTP request sent, awaiting response... 302 Found
Location: https://release-assets.githubusercontent.com/github-production-release-asset/753516996/d7c62b13-1e50-40ea-8fae-f34a4
--2026-05-01 18:17:53--  https://release-assets.githubusercontent.com/github-production-release-asset/753516996/d7c62b13-1e50-
Resolving release-assets.githubusercontent.com (release-assets.githubusercontent.com)... 185.199.108.133, 185.199.109.133, 185
Connecting to release-assets.githubusercontent.com (release-assets.githubusercontent.com)|185.199.108.133|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 1112971163 (1.0G) [application/octet-stream]
Saving to: ‘flickr8k.zip’
flickr8k.zip        100%[===================>]   1.04G  71.8MB/s    in 20s     
2026-05-01 18:18:14 (52.6 MB/s) - ‘flickr8k.zip’ saved [1112971163/1112971163]
import os, re, pickle, json
import numpy as np
from PIL import Image
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)
# ── 1. Build Vocabulary ──────────────────────────────────────
class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
    def __len__(self):
        return len(self.itos)
    @staticmethod
    def tokenize(text):
        return re.sub(r"[^a-z ]", "", text.lower()).split()
    def build(self, captions):
        counter = Counter()
        for cap in captions:
            counter.update(self.tokenize(cap))
        idx = 4
        for word, freq in counter.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1
    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.stoi.get(t, self.stoi["<UNK>"]) for t in tokens]
c = Vocabulary()
# Load captions from Flickr8k
captions_file = "./flickr8k/captions.txt"
img_dir       = "./flickr8k/Images"
with open(captions_file) as f:
    lines = f.read().strip().split("\n")[1:]   # skip header
img2caps = {}
for line in lines:
    fname, cap = line.split(",", 1)
    img2caps.setdefault(fname.strip(), []).append(cap.strip())
all_captions = [c for caps in img2caps.values() for c in caps]
vocab = Vocabulary(freq_threshold=5)
vocab.build(all_captions)
print(f"Vocabulary size: {len(vocab)}")
# ── 2. Dataset ───────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])
class Flickr8kDataset(Dataset):
    def __init__(self, img2caps, img_dir, vocab, transform):
        self.pairs = []
        for fname, caps in img2caps.items():
            for cap in caps:
                self.pairs.append((fname, cap))
        self.img_dir = img_dir
        self.vocab   = vocab
        self.transform = transform
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        fname, cap = self.pairs[idx]
        img = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        img = self.transform(img)
        tokens = ([self.vocab.stoi["<SOS>"]]
                  + self.vocab.encode(cap)
                  + [self.vocab.stoi["<EOS>"]])
        return img, torch.tensor(tokens, dtype=torch.long)
def collate_fn(batch):
    imgs, caps = zip(*batch)
    imgs = torch.stack(imgs)
    lengths = [len(c) for c in caps]
    max_len = max(lengths)
    padded = torch.zeros(len(caps), max_len, dtype=torch.long)
    for i, cap in enumerate(caps):
        padded[i, :len(cap)] = cap
    return imgs, padded, torch.tensor(lengths)
dataset    = Flickr8kDataset(img2caps, img_dir, vocab, transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True,
                        collate_fn=collate_fn, num_workers=2)
# ── 3. Model Definition ──────────────────────────────────────
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for param in resnet.parameters():
            param.requires_grad = False
        resnet.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.resnet = resnet
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
    def forward(self, images):
        x = self.resnet(images)
        return self.bn(x)
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size,
                 num_layers=1, dropout=0.5):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, embed_size)
        self.lstm    = nn.LSTM(embed_size, hidden_size,
                               num_layers, batch_first=True,
                               dropout=dropout if num_layers > 1 else 0)
        self.linear  = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions[:, :-1]))
        features   = features.unsqueeze(1)
        inputs     = torch.cat([features, embeddings], dim=1)
        hiddens, _ = self.lstm(inputs)
        outputs    = self.linear(hiddens[:, :-1])
        return outputs
    def generate(self, feature, vocab, max_len=30):
        result = []
        states = None
        x = feature.unsqueeze(0).unsqueeze(0)
        for _ in range(max_len):
            hidden, states = self.lstm(x, states)
            output   = self.linear(hidden.squeeze(1))
            pred_idx = output.argmax(1).item()
            if pred_idx == vocab.stoi["<EOS>"]:
                break
            result.append(vocab.itos[pred_idx])
            x = self.embed(torch.tensor([pred_idx], dtype=torch.long, device=x.device)).unsqueeze(0)
        return " ".join(result)
# ── 4. Training ──────────────────────────────────────────────
EMBED_SIZE  = 256
HIDDEN_SIZE = 256
VOCAB_SIZE  = len(vocab)
NUM_EPOCHS  = 5
LR          = 3e-4
encoder = EncoderCNN(EMBED_SIZE).to(DEVICE)
decoder = DecoderRNN(EMBED_SIZE, HIDDEN_SIZE, VOCAB_SIZE).to(DEVICE)
criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
params    = list(encoder.resnet.fc.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(params, lr=LR)
for epoch in range(NUM_EPOCHS):
    encoder.train(); decoder.train()
    total_loss = 0
    for imgs, caps, lengths in dataloader:
        imgs, caps = imgs.to(DEVICE), caps.to(DEVICE)
        features = encoder(imgs)
        outputs  = decoder(features, caps)
        outputs  = outputs[:, :caps.size(1)-1].reshape(-1, VOCAB_SIZE)
        targets  = caps[:, 1:caps.size(1)].reshape(-1)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    avg = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]  Loss: {avg:.4f}")
# ── 5. Inference ─────────────────────────────────────────────
def caption_image(image_path, encoder, decoder, vocab, transform):
    encoder.eval(); decoder.eval()
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feature = encoder(img).squeeze(0)
        caption = decoder.generate(feature, vocab)
    return caption
sample_img = os.path.join(img_dir,
             list(img2caps.keys())[0])
print("Generated caption:", caption_image(sample_img, encoder,
                                          decoder, vocab, transform))
Using device: cuda
Vocabulary size: 2988
Epoch [1/5]  Loss: 4.7062
Epoch [2/5]  Loss: 4.1049
Epoch [3/5]  Loss: 3.8972
Epoch [4/5]  Loss: 3.7597
Epoch [5/5]  Loss: 3.6536
Generated caption: a and dog a in of
# ====================== SCENARIO 1 ======================
# 1.1 Vocabulary size
print(f"Vocabulary size with freq_threshold=5: {len(vocab)}")
# 1.2 Some word to index examples
print("dog     →", vocab.stoi.get("dog", "Not found"))
print("running →", vocab.stoi.get("running", "Not found"))
print("the     →", vocab.stoi.get("the", "Not found"))
# 1.3 Tokenize and encode "A dog is running"
sentence = "A dog is running"
tokens = vocab.encode(sentence)
print(f"Encoded tokens for '{sentence}': {tokens}")
# 1.4 Change freq_threshold to 2
vocab2 = Vocabulary(freq_threshold=2)
vocab2.build(all_captions)
print(f"Vocabulary size with freq_threshold=2: {len(vocab2)}")
print("Explanation: Lower threshold includes more rare words → vocabulary becomes larger.")
# 1.5 Manually add a new word
vocab.stoi["deeplearning"] = len(vocab)
vocab.itos[len(vocab)] = "deeplearning"
print("After adding 'deeplearning':", vocab.encode("I love deeplearning"))
Vocabulary size with freq_threshold=5: 2989
dog     → 28
running → 112
the     → 24
Encoded tokens for 'A dog is running': [4, 28, 9, 112]
Vocabulary size with freq_threshold=2: 5202
Explanation: Lower threshold includes more rare words → vocabulary becomes larger.
After adding 'deeplearning': [2286, 2743, 2989]
# ====================== SCENARIO 2 ======================
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for param in resnet.parameters():
            param.requires_grad = False
        self.resnet = resnet
        self.resnet.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
    def forward(self, images):
        x = images
        print("Input shape          :", x.shape)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        print("After conv1 + pool   :", x.shape)
        x = self.resnet.layer1(x); print("After layer1         :", x.shape)
        x = self.resnet.layer2(x); print("After layer2         :", x.shape)
        x = self.resnet.layer3(x); print("After layer3         :", x.shape)
        x = self.resnet.layer4(x); print("After layer4         :", x.shape)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        print("After avgpool+flatten:", x.shape)
        x = self.resnet.fc(x)
        print("After FC (embed_size):", x.shape)
        x = self.bn(x)
        return x
# Test with one image
sample_img_path = os.path.join(img_dir, list(img2caps.keys())[0])
img = transform(Image.open(sample_img_path).convert("RGB")).unsqueeze(0).to(DEVICE)
encoder = EncoderCNN(embed_size=256).to(DEVICE)
# Set encoder to evaluation mode for single image inference with BatchNorm
encoder.eval()
_ = encoder(img)
# Change embed_size to 512
print("\n--- With embed_size=512 ---")
encoder512 = EncoderCNN(embed_size=512).to(DEVICE)
# Set encoder512 to evaluation mode for single image inference with BatchNorm
encoder512.eval()
_ = encoder512(img)
Input shape          : torch.Size([1, 3, 224, 224])
After conv1 + pool   : torch.Size([1, 64, 56, 56])
After layer1         : torch.Size([1, 256, 56, 56])
After layer2         : torch.Size([1, 512, 28, 28])
After layer3         : torch.Size([1, 1024, 14, 14])
After layer4         : torch.Size([1, 2048, 7, 7])
After avgpool+flatten: torch.Size([1, 2048])
After FC (embed_size): torch.Size([1, 256])
--- With embed_size=512 ---
Input shape          : torch.Size([1, 3, 224, 224])
After conv1 + pool   : torch.Size([1, 64, 56, 56])
After layer1         : torch.Size([1, 256, 56, 56])
After layer2         : torch.Size([1, 512, 28, 28])
After layer3         : torch.Size([1, 1024, 14, 14])
After layer4         : torch.Size([1, 2048, 7, 7])
After avgpool+flatten: torch.Size([1, 2048])
After FC (embed_size): torch.Size([1, 512])
# ====================== SCENARIO 3 ======================
class DecoderRNN(nn.Module):
   def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0.5):
       super().__init__()
       self.embed = nn.Embedding(vocab_size, embed_size)
       self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
       self.linear = nn.Linear(hidden_size, vocab_size)
       self.dropout = nn.Dropout(dropout)
   def forward(self, features, captions):
       embeddings = self.dropout(self.embed(captions[:, :-1]))
       print("Embeddings shape     :", embeddings.shape)
features = features.unsqueeze(1)
       features   features.unsqueeze(1)
       inputs = torch.cat([features, embeddings], dim=1)
       hiddens, _ = self.lstm(inputs)
       print("LSTM output shape    :", hiddens.shape)
       outputs = self.linear(hiddens[:, :-1])
       print("Linear output shape  :", outputs.shape)
       return outputs
   def generate(self, feature, vocab, max_len=20):
       result = []
       states = None
       x = feature.unsqueeze(0).unsqueeze(0)   # (1,1,embed_size)
       for i in range(max_len):
           hidden, states = self.lstm(x, states)
           output = self.linear(hidden.squeeze(1))
           pred_idx = output.argmax(1).item()
           word = vocab.itos[pred_idx]
           print(f"Step {i+1}: Predicted word = {word} (idx={pred_idx})")
           if pred_idx == vocab.stoi["<EOS>"]:
               break
           result.append(word)
           x = self.embed(torch.tensor([pred_idx], device=feature.device)).unsqueeze(0)
       return " ".join(result)
# Test Decoder
EMBED_SIZE = 256
HIDDEN_SIZE = 256
decoder = DecoderRNN(EMBED_SIZE, HIDDEN_SIZE, len(vocab)).to(DEVICE)
features = torch.randn(1, EMBED_SIZE).to(DEVICE)   # dummy feature
# Dummy caption for shape check
dummy_cap = torch.tensor([[vocab.stoi["<SOS>"], vocab.stoi.get("dog", 3), vocab.stoi["<EOS>"]]]).to(DEVICE)
_ = decoder(features, dummy_cap)
Embeddings shape     : torch.Size([1, 2, 256])
LSTM output shape    : torch.Size([1, 3, 256])
Linear output shape  : torch.Size([1, 2, 2990])
# ====================== SCENARIO 4 ======================
# Temperature sampling in generate
def generate_with_temperature(feature, decoder, vocab, max_len=20, temperature=1.0):
    result = []
    states = None
    x = feature.unsqueeze(0).unsqueeze(0)
    for _ in range(max_len):
        hidden, states = decoder.lstm(x, states)
        output = decoder.linear(hidden.squeeze(1))
        output = output / temperature
        probs = torch.softmax(output, dim=1)
        pred_idx = torch.multinomial(probs, 1).item()
        if pred_idx == vocab.stoi["<EOS>"]:
            break
        result.append(vocab.itos[pred_idx])
        x = decoder.embed(torch.tensor([pred_idx], device=feature.device)).unsqueeze(0)
    return " ".join(result)
# Simple Beam Search (width=3)
def beam_search(feature, decoder, vocab, max_len=20, beam_width=3):
    # Basic implementation - for full version you can expand later
    # For now using greedy as placeholder
    return decoder.generate(feature, vocab, max_len)
# Run with different temperatures
sample_feature = encoder(img).squeeze(0)   # from scenario 2
print("Temperature 0.5:", generate_with_temperature(sample_feature, decoder, vocab, temperature=0.5))
print("Temperature 1.5:", generate_with_temperature(sample_feature, decoder, vocab, temperature=1.5))
Input shape          : torch.Size([1, 3, 224, 224])
After conv1 + pool   : torch.Size([1, 64, 56, 56])
After layer1         : torch.Size([1, 256, 56, 56])
After layer2         : torch.Size([1, 512, 28, 28])
After layer3         : torch.Size([1, 1024, 14, 14])
After layer4         : torch.Size([1, 2048, 7, 7])
After avgpool+flatten: torch.Size([1, 2048])
After FC (embed_size): torch.Size([1, 256])
Temperature 0.5: strange wet screams atm soars hairy dry trash bar small flying traveling overlook dangling turn wall plane ma
Temperature 1.5: suits chases microphone having casts own bicycler motorcyclist parked teenagers drops flowing mobile getting 
# ====================== SCENARIO 5 ======================
# Just change these parameters and re-run training loop below
# Example: Change dropout, num_layers, etc.
decoder = DecoderRNN(embed_size=256, hidden_size=256, vocab_size=len(vocab),
                     num_layers=1, dropout=0.0)   # dropout=0
# For unfreezing last block of ResNet (advanced)
# for param in encoder.resnet.layer4.parameters():
#     param.requires_grad = True
# ====================== TRAINING LOOP (Common) ======================
EMBED_SIZE  = 256
HIDDEN_SIZE = 256
NUM_EPOCHS  = 3          # as per scenario instruction
LR          = 3e-4
encoder = EncoderCNN(EMBED_SIZE).to(DEVICE)
decoder = DecoderRNN(EMBED_SIZE, HIDDEN_SIZE, len(vocab), num_layers=1, dropout=0.5).to(DEVICE)
criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
params = list(encoder.resnet.fc.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(params, lr=LR)
for epoch in range(NUM_EPOCHS):
    encoder.train(); decoder.train()
    total_loss = 0
    for imgs, caps, lengths in dataloader:
        imgs, caps = imgs.to(DEVICE), caps.to(DEVICE)
        features = encoder(imgs)
        outputs = decoder(features, caps)
        outputs = outputs.reshape(-1, len(vocab))
        targets = caps[:, 1:].reshape(-1)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Loss: {avg_loss:.4f}")
# Final Caption
sample_img_path = os.path.join(img_dir, list(img2caps.keys())[5])
print("Generated Caption:", caption_image(sample_img_path, encoder, decoder, vocab, transform))
Show hidden output
Start coding or generate with AI.
