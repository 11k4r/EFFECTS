# EFFECTS

EFFECTS is a Python library for explaianle feature extraction for time series data. EFFECTS provide both feature extractor and feature explorer to maximize the performance and the explainability of the model.


## Usage

```python
import effects

# read data
X_train, X_test, y_train, y_test = effects.ts_format_data_loader('datasets/BasicMotions/')

# crate feature extractor
extractor = effects.Extractor()

# extract features
train_features = extractor.fit_transform(X_train, y_train)
test_features = extractor.transform(X_test, y_test)
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

