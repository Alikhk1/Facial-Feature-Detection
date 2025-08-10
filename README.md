# 🧠 MultiTask ResNet18: Gender, Eyeglasses & Shirt Color Classifier

This project uses a **ResNet-18** model trained on **1750 images** taken from **FairFace Dataset**, **250 from each race with 50/50 gender split** to predict:
- 👕 Shirt color (11 classes)
- 🕶 Eyeglasses (Yes/No)
- 🚻 Gender (Male/Female)

**The Shirt Color and Eyeglasses Labels were manually annotated.**  
**Achieved accuracies:** **93.4%** eyeglasses, **80.9%** gender, **52.6%** shirt color.

---

## 📦 How to run

> ⚡ **Step 1:** Open a new [Google Colab](https://colab.research.google.com) notebook and copy-paste the following code blocks.

---

### ✅ **1️⃣ Download the pretrained model**

```python
# Install gdown for downloading model from Google Drive
!pip install -q gdown

# Download model from public Google Drive link
!gdown 'https://drive.google.com/uc?id=1UFDFWbFkQ7nOJSYmEzxrGo0B9J3qbtZu' -O model_multitask_resnet.pth
```
### 🧰 **2️⃣ Run inference on your image**

```python
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from google.colab import files
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Define label encoder (use same order as training)
shirt_color_encoder = LabelEncoder()
shirt_color_encoder.classes_ = np.array(['black', 'blue', 'brown', 'gray', 'green', 
                                        'other', 'pink', 'purple', 'red', 'white', 'yellow'])

# Define model
class MultiTaskResNet(nn.Module):
    def __init__(self, num_colors):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        self.gender_head = nn.Linear(512, 1)
        self.eyeglasses_head = nn.Linear(512, 1)
        self.shirt_color_head = nn.Linear(512, num_colors)
    def forward(self, x):
        features = self.backbone(x)
        gender = torch.sigmoid(self.gender_head(features))
        eyeglasses = torch.sigmoid(self.eyeglasses_head(features))
        shirt_color = self.shirt_color_head(features)
        return gender, eyeglasses, shirt_color

# Load model
model = MultiTaskResNet(num_colors=len(shirt_color_encoder.classes_))
model.load_state_dict(torch.load('model_multitask_resnet.pth', map_location='cpu'))
model.eval()

# Upload image
uploaded = files.upload()
img_path = list(uploaded.keys())[0]

# Preprocess
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])
image = Image.open(img_path).convert('RGB')
image = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    pred_gender, pred_eyeglasses, pred_shirt_color = model(image)
    gender = "Male" if pred_gender.item()<0.5 else "Female"   # dataset uses 0=male
    eyeglasses = "Yes" if pred_eyeglasses.item()>0.5 else "No"
    shirt_color_idx = torch.argmax(pred_shirt_color, dim=1).item()
    shirt_color = shirt_color_encoder.inverse_transform([shirt_color_idx])[0]

print(f"✅ Prediction: Gender: {gender}, Eyeglasses: {eyeglasses}, Shirt color: {shirt_color}")
```


