import os
import torch
import pandas as pd
import cxr_dataset as CXR
import seaborn as sns
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision import transforms, datasets
import sklearn
import sklearn.metrics as sklm
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable
import numpy as np

def make_pred_multilabel(data_transforms, model, PATH_TO_IMAGES):

    # calc preds in batches of 16, can reduce if your GPU has less RAM
    BATCH_SIZE = 16

    # set model to eval mode; required for proper predictions given use of batchnorm
    model.eval()

    # create dataloader
    dataset = CXR.CXRDataset(
        path_to_images=PATH_TO_IMAGES,
        fold="test",
        transform=data_transforms['val'])
    dataloader = torch.utils.data.DataLoader(
        dataset, BATCH_SIZE, shuffle=False, num_workers=8)
    size = len(dataloader.dataset)

    # create empty dfs
    pred_df = pd.DataFrame(columns=["Image Index"])
    true_df = pd.DataFrame(columns=["Image Index"])
    
    # create empty confusion matrix
    conf_matrix = np.zeros((len(dataset.PRED_LABEL), len(dataset.PRED_LABEL)))

    # iterate over dataloader
    for i, data in enumerate(dataloader):

        inputs, labels, _ = data
        inputs, labels = inputs.cuda(), labels.cuda()

        true_labels = labels.cpu().data.numpy()
        batch_size = true_labels.shape[0]

        outputs = model(inputs)
        probs = outputs.cpu().data.numpy()

        # get predictions and true values for each item in batch
        for j in range(batch_size):
            thisrow = {}
            truerow = {}
            thisrow["Image Index"] = dataset.df.index[BATCH_SIZE * i + j]
            truerow["Image Index"] = dataset.df.index[BATCH_SIZE * i + j]

            # iterate over each entry in prediction vector; each corresponds to
            # individual label
            for k in range(len(dataset.PRED_LABEL)):
                thisrow["prob_" + dataset.PRED_LABEL[k]] = probs[j, k]
                truerow[dataset.PRED_LABEL[k]] = true_labels[j, k]

            thisrow_df = pd.DataFrame.from_dict(thisrow, orient='index').T
            truerow_df = pd.DataFrame(truerow, index=[0])
            pred_df = pd.concat([pred_df, thisrow_df], ignore_index=True)
            true_df = pd.concat([true_df, truerow_df], ignore_index=True)
            
            # update confusion matrix
            pred_label = np.argmax(probs[j, :])
            true_label = np.argmax(true_labels[j, :])
            conf_matrix[true_label, pred_label] += 1


        if(i % 10 == 0):
            print(str(i * BATCH_SIZE))
            
    # print confusion matrix
    # display confusion matrix as table
    conf_matrix_table = pd.DataFrame(conf_matrix, columns=dataset.PRED_LABEL, index=dataset.PRED_LABEL)
    print("Confusion matrix:")
    print(conf_matrix_table)
    
    auc_df = pd.DataFrame(columns=["label", "auc"])
    
    # calculate accuracy
    accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
    print(f"Test set accuracy: {accuracy}")
    
    # calculate AUCs
    auc_dict = {}
    for label in dataset.PRED_LABEL:
        if label == "No Finding":  # skip the "No Finding" label
            continue
        true_labels = true_df[label].values.astype(int)
        pred_probs = pred_df["prob_" + label].values
        try:
            auc = sklm.roc_auc_score(true_labels, pred_probs)
            auc_dict[label] = auc
        except ValueError as e:
            print(f"Can't calculate AUC for {label}: {e}")

    # print AUCs
    for label, auc in auc_dict.items():
        print(f"{label} AUC: {auc:.4f}")
        
    # create the 'results' directory if it doesn't exist
    os.makedirs("results", exist_ok=True)   
        
    pred_df.to_csv("results/preds.csv", index=False)
    auc_df.to_csv("results/aucs.csv", index=False)


    # save the model to a file
    filename = "my_model.pkl"
    torch.save(model, filename)
    
    # create a DenseNet instance
    densenet = model.densenet121(pretrained=True)

    # get the state dictionary from the DenseNet instance
    state_dict = densenet.state_dict()

    # save the state dictionary to a file
    torch.save(state_dict, 'densenet121_state_dict.pth')

    return pred_df, auc_df
