# Lineage to follow for CNNs

- [x] Basic FCN
- [x] Basic CNN
- [x] Lenet5
- [x] AlexNet
- [x] VGGNet
- [x] Inception/GoogleNet
- [x] ResNet
- [ ] SENet
- [ ] DenseNet
- [ ] MobileNet
- [ ] EfficientNet
- [ ] UNet
- [ ] ViT
- [ ] Swim Transformer
- [ ] ConvNeXt

# Lineage for text

- [ ] Embedding + RNN (vanilla)
- [ ] LSTM
- [ ] GRU
- [ ] Seq2Seq (encoder-decoder with RNN)
- [ ] Bahdanau Attention
- [ ] Transformer (Vaswani 2017)
- [ ] BERT
- [ ] GPT
- [ ] T5

# How to execute shell command

## Defaults (cnn, mnist)

`python3 torch_portfolio/src/torch_portfolio/cnn_lineage/train.py`

## Override anything

`python3 torch_portfolio/src/torch_portfolio/cnn_lineage/train.py model=fcn dataset=cifar10 train.epochs=20 train.lr=3e-4`

## Multirun — sweep over LRs automatically

`python3 torch_portfolio/src/torch_portfolio/cnn_lineage/train.py --multirun train.lr=1e-3,3e-4,1e-4`

## Multirun over model × dataset grid

`python3 torch_portfolio/src/torch_portfolio/cnn_lineage/train.py --multirun model=cnn,fcn dataset=mnist,cifar10`

### Basic custom training

`python3 src/torch_portfolio/cnn_lineage/train.py model=alexnet dataset=cifar10`
