import torch
import requests as req
from PIL import Image # convert image
from torchvision import transforms
import os




# response = req.get('/Users/stan/Desktop/deep learning/assignment1/archive/wikiart/0-ravenna-cappella-arcivescovile-166.jpg', stream=True)
# os.system(f'xdg-open "{image_path}"')


# image = Image.open('/Users/stan/Desktop/deep learning/assignment1/archive/wikiart/0-ravenna-cappella-arcivescovile-166.jpg').convert('RGB')  # 确保图像为RGB格式
preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  # convert image size to 224x224, its a normal convert way
            transforms.ToTensor(),  # change image to pytorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # normalization image
        ])
# image_1 = preprocess(image
image_paths = "0-ravenna-cappella-arcivescovile-166.jpg"
path = "/Users/stan/Desktop/deep_learning/assignment1/archive/wikiart/wikiart/" + image_paths

image = Image.open("/Users/stan/Desktop/deep_learning/assignment1/archive/wikiart/wikiart/0-ravenna-cappella-arcivescovile-166.jpg").convert('RGB')
# if os.path.exists('/Users/stan/Desktop/deep_learning/assignment1/archive/wikiart/wikiart/0-ravenna-cappella-arcivescovile-166.jpg'):
#     print(123)
#     image = Image.open("/Users/stan/Desktop/deep_learning/assignment1/archive/wikiart/0-ravenna-cappella-arcivescovile-166.jpg").convert('RGB')
# else:
#     print(222)

image_1 = preprocess(image)
print(image_1)
