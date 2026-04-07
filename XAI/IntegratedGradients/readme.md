# IntegratedGradients with Resnet (Captum)  

## Overview
attribution for visual inputs. As part of Explainable AI (XAI), it aims to make machine learning models more transparent and interpretable. By highlighting the contribution of individual visual tokens (e.g., image regions or patches), it provides insights into how models arrive at their predictions, thereby improving trust and facilitating analysis of model behavior.

## Dataset
The experiments are performed on the Oxford-IIIT Pet Dataset dataset, which is commonly used for evaluating image classification models.

```markdown
### Python

from torchvision.datasets import OxfordIIITPet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# should set download=True the first time 
train = OxfordIIITPet(
    root="./data",
    split="trainval",
    target_types="category",
    download=False,
    transform=transform
)

test = OxfordIIITPet(
    root="./data",
    split="test",
    target_types="category",
    download=False,
    transform=transform
)
```

## Model

```markdown
### Python

from torchvision import models
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

```
