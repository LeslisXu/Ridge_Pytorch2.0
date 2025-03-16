import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

class Ridge:
    def __init__(self, alpha = 0, fit_intercept = True,):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        
    def fit(self, X: torch.tensor, y: torch.tensor) -> None:
        X = X.rename(None)
        y = y.rename(None).view(-1,1)
        assert X.shape[0] == y.shape[0], "Number of X and y rows don't match"
        if self.fit_intercept:
            X = torch.cat([torch.ones(X.shape[0], 1), X], dim = 1)
        # Solving X*w = y with Normal equations:
        # X^{T}*X*w = X^{T}*y 
        lhs = X.T @ X 
        rhs = X.T @ y
        if self.alpha == 0:
            self.w, _ = torch.lstsq(rhs, lhs)
            print(f'self.w.shape = {self.w.shape}')
        else:
            ridge = self.alpha*torch.eye(lhs.shape[0])
            self.w = torch.linalg.lstsq( lhs + ridge, rhs ).solution
            print(f'self.w.shape = {self.w.shape}')
            
    def predict(self, X: torch.tensor) -> None:
        X = X.rename(None)
        if self.fit_intercept:
            X = torch.cat([torch.ones(X.shape[0], 1), X], dim = 1)
        print(f'X.shape = {X.shape}\tself.w.shape = {self.w.shape}')
   
        return X @ self.w
    
    def mse_loss(self, y_true: torch.tensor, y_pred: torch.tensor) -> torch.tensor:

        return torch.mean((y_true - y_pred) ** 2)
    
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
