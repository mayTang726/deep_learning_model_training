import os
import numpy as np
import pandas as pd
from PIL import Image # convert image
from torchvision import transforms
# from sklearn import preprocessing
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def change_dataset(params, X_train, y_train, X_val, y_val, X_test, y_test):


    data_array = ['X_train', 'X_val', 'X_test']
    # label_array = ['y_train', 'y_val', 'y_test']

    X_train_ = []
    X_val_ = []
    X_test_ = []
    # y_train_ = []
    # y_val_ = []
    # y_test_ = []


    preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  # convert image size to 224x224, its a normal convert way
            transforms.ToTensor(),  # change image to pytorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # normalization image
        ])
    
    def load_and_preprocess_image(image_path, preprocess):
        '''
            when i directly request url ,the server aleways response [Errno 54] Connection reset by peer
            so i changed image reading method to read local image

            if you wanna run this codes, you should change this path to your own local image path
        ''' 
        path = "/Users/stan/Desktop/deep_learning/assignment1/archive/wikiart/wikiart/" + image_path
        image = Image.open(path).convert('RGB')  # 确保图像为RGB格式
        image = preprocess(image)
        return image
    # convert image
    for element in data_array:
        if element == 'X_train':
            image_paths = X_train['file_name'].tolist() # image feature 
        elif element == 'X_val':
            image_paths = X_val['file_name'].tolist() # image feature 
        else:
            image_paths = X_test['file_name'].tolist() # image feature 
        
        for i in range(len(image_paths)):
            image = load_and_preprocess_image(image_paths[i], preprocess)
            if element == 'X_train':
                X_train_.append(image)
            elif element == 'X_val':
                X_val_.append(image)
            else:
                X_test_.append(image)
    # convert label to list 
    # for elemtnt in label_array:
    #     if elemtnt == 'y_train':
    #         y_train_ = y_train['label'].tolist() 
    #     elif elemtnt == 'y_val':
    #         y_val_ = y_val['label'].tolist() 
    #     else:
    #         y_test_ = y_test['label'].tolist()
        
    X_train_ = pd.DataFrame(X_train_)
    X_test_ = pd.DataFrame(X_test_)
    X_val_ = pd.DataFrame(X_val_)

    
    # save dataset to responding folder
    X_train_.to_csv(os.path.join(params.data_dir, "X_train.csv"), index=False)  
    X_test_.to_csv(os.path.join(params.data_dir, "X_test.csv"), index=False) 
    X_val_.to_csv(os.path.join(params.data_dir, "X_val.csv"), index=False) 
    y_train.to_csv(os.path.join(params.data_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(params.data_dir, "y_test.csv"), index=False) 
    y_val.to_csv(os.path.join(params.data_dir, "y_val.csv"), index=False) 

