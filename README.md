# Invertible neural networks in PyTorch

This repository implements invertible neural networks in pytorch with very limited architecture assumptions.

# Example code

This shows how an invertible neural network can be defined

```python
import invertible_layers as il

# Construct forward model
model = il.Sequential(il.PixelShuffle(4),
                      il.Conv2d(16, 3),
                      il.LeakyReLU(0.5),
                      il.Conv2d(16, 3),
                      il.LeakyReLU(0.5),
                      il.Conv2d(16, 3),
                      il.PixelUnShuffle(4))
                      
# Get the inverse model
inverse = model.inverse
```
