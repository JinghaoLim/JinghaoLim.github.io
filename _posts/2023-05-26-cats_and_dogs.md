#  Running Cats and Dogs Example on Google Colab

I personally believe that Google Colab is better than Github Codespaces.

Table of contents:

1. TOC
{:toc}

## Code
```python
# Install dependencies
pip install fastbook

# Import dependencies
from fastbook import *
from fastai.vision.all import *

path = untar_data(URLs.PETS)/'images'

def is_cat(x): return x[0].isupper()
dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat, item_tfms=Resize(224))
    
# Train model
learn = vision_learner(dls, resnet50, metrics=accuracy)
learn.fine_tune(3)
```

## Modifications
- Implement ResNet-50 rather than ResNet-34.
- Present the final result in accuracy rather than in error rate.

## Results
![Cats and Dogs](/images/cats_and_dogs.jpg)

The model achieved an accuracy of **0.995940** after three epochs.
