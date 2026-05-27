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
