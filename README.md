# How to register a new model

## CNN Lineage

- In filepath: src/torch_portfolio/cnn_lineage/conf/model
- create new_model.yaml with the content

```yaml
name: new_model
```

- at: `torch_portfolio/cnn_lineage/models/__init__.py`
- Add the new model to the registry

```python
_REGISTRY: dict[str, ModelFactory] = {
    "new_model": lambda c, n: NewModel(in_channels=c, num_classes=n),
}
```

**CNN Lineage** (current project)

```
-> Basic FCN
-> Basic CNN
-> Lenet5
-> AlexNet
-> VGGNet
-> Inception/GoogleNet
-> ResNet
-> SENet
-> DenseNet
-> MobileNet
-> EfficientNet
-> UNet
-> ViT
-> Swim Transformer
-> ConvNeXt
```

---

**Text Lineage** (`text_lineage/`)

```
→ Tokenization (BPE — implement a toy version)
→ Embedding + RNN
→ LSTM
→ GRU
→ Seq2Seq
→ Bahdanau Attention
→ Positional Encoding variants (sinusoidal / learned / RoPE / ALiBi)
→ Transformer (Vaswani 2017)
→ BERT
    - RoBERTa
    - ALBERT
    - DeBERTa
    - ConvBERT
    - BigBird
    - ELECTRA
→ XLNet
→ GPT
→ T5
→ MoE (sparse routing, toy implementation)
Distillation
```

---

**Track 2: Training Concepts** (papers + derivations + notes, no model files)

```
→ Tokenization theory (BPE math, vocabulary size decisions)
→ Pretraining objectives (MLM vs CLM vs span corruption)
→ Scaling laws (Chinchilla — read + derive compute-optimal formula)
→ Weight tying + embedding geometry
→ Pre-LN vs Post-LN (training stability at scale)
→ Distributed training (ZeRO stages, FSDP, tensor/pipeline parallelism)
→ DPO (read paper mathematically — KL + Bradley-Terry)
→ Policy gradients → REINFORCE derivation
→ Trust region intuition → PPO
→ GRPO (DeepSeek-R1)
→ Process reward models vs outcome reward models
→ Rejection sampling fine-tuning
```

---

Order: finish CNN lineage → text lineage → Track 2 interleaved with text lineage (e.g. do scaling laws alongside GPT, do GRPO alongside MoE). Paper runs in parallel, doesn't wait for any of this.
