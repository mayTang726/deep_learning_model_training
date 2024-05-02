import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset # an abstract class of pytorch,create the costom dataset
from PIL import Image # convert image
from torchvision import transforms
from sklearn import preprocessing
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# resolve dataset 
class WikiArtDataset(Dataset):
    def __init__(self, data_dir, file_name, label_name, flatten=True, device=any, local_image_path=''):
        """
        data_dir (str): Path to data containing data and labels. 
        X_filename (str): Name of file containing input data. 
        y_filename (str): Name of file containing labels.
        """
        self.device = device
        local_image_path = local_image_path
        df = pd.read_csv(os.path.join(data_dir, file_name)) #get dataset
        labels = pd.read_csv(os.path.join(data_dir, label_name))
        # label_list = [list(map(int, label.split(','))) for label in labels['label']]
        image_paths = df['file_name'].tolist() # image feature 
        labels = labels['label'].tolist()
        
        # image pretreatment convert
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  # convert image size to 224x224, its a normal convert way
            transforms.ToTensor(),  # change image to pytorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # normalization image
        ])

        self.data = []
        # download image and convert
        def load_and_preprocess_image(image_path, preprocess):
            '''
                when i directly request url ,the server aleways response [Errno 54] Connection reset by peer
                so i changed image reading method to read local image

                if you wanna run this codes, you should change this path to your own local image path
            ''' 
            path = local_image_path + image_path
            image = Image.open(path).convert('RGB')  # 确保图像为RGB格式
            image = preprocess(image)
            return image

        for i in range(len(image_paths)):
            image = load_and_preprocess_image(image_paths[i], preprocess)
            self.data.append(image)

        # change label to tensor
        self.labels = torch.tensor(labels,dtype=torch.long).to(self.device)

    def __getitem__(self, index):
        X = self.data[index].float().to(self.device)
        y = self.labels[index]
        return X, y

    def __len__(self):
        return len(self.data)
