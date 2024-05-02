import torch
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

def train(device, train_loader, val_loader):

    train_data, train_labels = extract_features_and_labels(device, train_loader)
    val_data, val_labels = extract_features_and_labels(device, val_loader)

    svm_pipeline = make_pipeline(StandardScaler(), SVC())

    parameters = {
        'svc__C': [0.1, 1, 10, 100],  # C value
        'svc__kernel': ['rbf','linear'],  # kernel 
        'svc__gamma': [0.001, 0.0001, 'scale']  # scal control gamma value baed on feature number
    }

    grid_search = GridSearchCV(svm_pipeline, parameters, cv=3)
    grid_search.fit(train_data, train_labels)

    print("Best parameters set found on development set:")
    print(grid_search.best_params_)
    print("Grid scores on development set:")
    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    # 使用最佳参数的模型进行验证集评估
    val_predictions = grid_search.predict(val_data)
    val_acc = np.mean(val_predictions == val_labels)

    print("label details:")
    print(classification_report(val_labels, val_predictions))
    print(f"SVM val acc: {val_acc}")

    return grid_search.best_estimator_  # 返回训练好的最佳模型

def val(svm_classifier, device, test_loader):
    test_data, test_labels = extract_features_and_labels(device, test_loader)
    
    if test_data is None:
        raise ValueError("Data extraction failed.")
    
    test_predictions = svm_classifier.predict(test_data)
    test_acc = np.mean(test_predictions == test_labels)
    print(f"SVM Test Accuracy: {test_acc}")

def extract_features_and_labels(device, loader):
    features = []
    labels = []
    with torch.no_grad():
        for data, target in loader:
            # 确保数据已正确加载并可以被处理
            if data is not None and target is not None:
                # 将数据移到 CPU 上
                data = data.cpu()
                target = target.cpu()
                # 将数据展平，使其适合 SVM
                flattened_data = data.reshape(data.size(0), -1)
                labels.append(target)
                features.append(flattened_data)
            else:
                return None, None
    # 将列表转换为 NumPy 数组
    return torch.cat(features).numpy(), torch.cat(labels).numpy()
