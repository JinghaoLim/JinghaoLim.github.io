# Jennie or Lisa? An Adaptation based on "Is it a bird?" from fast.ai

I created this adaptation to test the accuracy of model on a more specific case. Surprisingly, it gave an almost perfect prediction with an accuracy of **0.9985**.

Table of contents:
1. TOC
{:toc}

## Download test image
```python
# Install dependencies
pip install fastbook

# Search Jennie image
from fastbook import search_images_ddg
urls = search_images_ddg('Jennie', max_images=1)
urls[0]

# Load Jennie image
from fastdownload import download_url
dest = 'jennie.jpg'
download_url(urls[0], dest, show_progress=False)

from fastai.vision.all import *
im = Image.open(dest)
im.to_thumb(256,256)
```
![Jennie image](/images/jennie_img.png)

```python
# Search and load Lisa image
download_url(search_images_ddg('Lisa', max_images=1)[0], 'lisa.jpg', show_progress=False)
Image.open('lisa.jpg').to_thumb(256,256)
```
![Lisa image](/images/lisa_img.png)

## Generate training and validation sets
```python
# Search and generate image datasets
searches = 'Jennie','Lisa'
path = Path('Jennie_or_not')
from time import sleep

for o in searches:
    dest = (path/o)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images_ddg(f'{o} photo'))
    sleep(10)  # Pause between searches to avoid over-loading server
    download_images(dest, urls=search_images_ddg(f'{o} glasses photo'))
    sleep(10)
    download_images(dest, urls=search_images_ddg(f'{o} mask photo'))
    sleep(10)
    resize_images(path/o, max_size=400, dest=path/o)

# Remove failed downloads
failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
len(failed)

# Generate training set and validation set
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path, bs=32)

dls.show_batch(max_n=6)
```
![Jennie Lisa image](/images/jennie_lisa_img.png)

## Train model and predict results
```python
# Train model
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(10)
```
![Jennie or Lisa image](/images/jennie_or_lisa.jpg)

```python
# Predict results
is_jennie,_,probs = learn.predict(PILImage.create('jennie.jpg'))
print(f"She is: {is_jennie}.")
print(f"Probability she is Jennie: {probs[0]:.4f}")
```
![Jennie or Lisa image](/images/jennie_or_lisa(2).jpg)

## More about Jennie and Lisa
Want to know more about Jennie? [Click here](https://en.wikipedia.org/wiki/Jennie_(singer)).

Want to know more about Lisa? [Click here](https://en.wikipedia.org/wiki/Lisa_(rapper)).
