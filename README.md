# Ridge_Pytorch2.0
An implementation of Ridge Regression using Pytorch 2.0.

 This code was Imspired by https://gist.github.com/myazdani/3d8a00cf7c9793e9fead1c89c1398f12, which was written in Pytorhch 1.0 Framework.

For test:
```python
if __name__ == "__main__":
    ## demo
    torch.manual_seed(42)
    X = torch.randn(30, 18)
    y = torch.randn(30, 1)  # supports only single outputs

    model = Ridge(alpha = 1e-3, fit_intercept = True)
    model.fit(X, y)

    predictions = model.predict(X)
    loss = model.mse_loss(y, predictions)
    print(f'MSE Loss: {loss.item()}')
 
