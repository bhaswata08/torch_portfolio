# Lineage to follow for CNNs

- [x] Basic FCN
- [x] Basic CNN
- [ ] Lenet5
- [ ] AlexNet
- [ ] VGGNet
- [ ] Inception/GoogleNet
- [ ] ResNet
- [ ] DenseNet
- [ ] SENet
- [ ] EfficientNet
- [ ] UNet
- [ ] MobileNet
- [ ] ViT
- [ ] Swim Transformer
- [ ] ConvNeXt

# How to execute shell command

## Defaults (cnn, mnist)

python3 torch_portfolio/src/torch_portfolio/cnn_lineage/train.py

## Override anything

python3 torch_portfolio/src/torch_portfolio/cnn_lineage/train.py model=fcn dataset=cifar10 train.epochs=20 train.lr=3e-4

## Multirun — sweep over LRs automatically

python3 torch_portfolio/src/torch_portfolio/cnn_lineage/train.py --multirun train.lr=1e-3,3e-4,1e-4

## Multirun over model × dataset grid

python3 torch_portfolio/src/torch_portfolio/cnn_lineage/train.py --multirun model=cnn,fcn dataset=mnist,cifar10
