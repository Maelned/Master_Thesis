import numpy as np
import matplotlib.pyplot as plt
import pickle
import itertools
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from operator import truediv, add
from sklearn.metrics import confusion_matrix

Dataset = "E:\\NTNU\\TTM4905 Communication Technology, Master's Thesis\\Code\\Dataset\\"
Test_dir = Dataset + "ISIC2018V2\\Test\\"

# model = load_model("./Saves/Models/InceptionV3_v3.h5")
cm_plot_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

def print_table(confusion_matrix):
    classes = ["akiec", "bcc", "bkl", "df", "mel","nv","vasc"]
    healthy = [0,0,1,1,0,0,1]

    HH = 0
    H_as_H = 0
    H_as_C = 0
    CC = 0
    C_as_H = 0
    C_as_C = 0
    tot = 0

    for line in range(len(confusion_matrix)): #fixe le TRUE LABEL
        bool_healthy = (healthy[line]==1)

        for column in range(len(confusion_matrix)): #parcours les PREDICTED LABELS
            val = confusion_matrix[line,column]
            if bool_healthy:
                if line == column:
                    HH+= val
                elif healthy[column] == 1: #classified as healthy
                    H_as_H+= val
                else:
                    H_as_C+= val
            elif line == column:
                CC += val
            elif healthy[column] == 1:
                C_as_H += val
            else:
                C_as_C += val
            tot += val

    tot_H = HH + H_as_C + H_as_H
    tot_C = CC + C_as_C + C_as_H

    print("Line 1 ", HH,(100*HH/tot_H), H_as_H,(100*H_as_H/tot_H), H_as_C, (100*H_as_C/tot_H))
    print("Line 2 ", CC,(100*CC/tot_C), C_as_H,(100*C_as_H/tot_C), C_as_C, (100*C_as_C/tot_C))
    print("tot = ", tot, "tot FN = ", tot - HH - CC)
    C_as_H = np.around(100*C_as_H/tot_C, 4)
    return C_as_H


def plot_curves(history):
    loss_train = history['loss']
    loss_val = history['val_loss']
    acc_train = history['categorical_accuracy']
    acc_val = history['val_categorical_accuracy']
    values = [[loss_train, loss_val], [acc_train, acc_val]]
    epochs = range(len(loss_train))

    for i in range(2):
        for j in range(2):
            if j:
                plt.plot(epochs, values[i][j], color='b', label="Validation")
            else:
                plt.plot(epochs, values[i][j], color='g', label="Training")
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
        plt.show()

    # plt.plot(epochs, acc_train, 'g', label='Training acc')
    # plt.plot(epochs, acc_val, 'b', label='validation accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('accuracy')
    # plt.legend()
    # plt.show()


def plot_graph(multiple_cm, title, experiment):
    plt.xticks(fontsize=9)
    precision_tot, recall_tot, F1_tot, Specificity_tot, Accuracy_tot, Fooling_rate = [], [], [], [], [] , []
    for i in multiple_cm:
        plot_metrics(i, "No title", False, True, True)
        macro_avg_precision, macro_avg_recall, macro_avg_F1, Specificity, Accuracy = model_evaluation(i)
        precision_tot.append(macro_avg_precision)
        recall_tot.append(macro_avg_recall)
        F1_tot.append(macro_avg_F1)
        Specificity_tot.append(Specificity)
        Accuracy_tot.append(Accuracy)
        Fooling_rate.append(1-Accuracy)
    plt.title(title)

    plt.plot(experiment, Accuracy_tot, 'g', label='Accuracy')
    plt.plot(experiment, recall_tot, 'b', label='Recall')
    plt.plot(experiment, Specificity_tot, 'r', label='Specificity')

    # plt.plot(experiment, Fooling_rate, label="Fooling_rate")
    plt.xlabel('Different experiments')
    plt.ylabel('Percentage')
    plt.legend()
    plt.show()


def plot_metrics(cm, title, plot_cm, verbose, Attack):
    if plot_cm:
        plot_confusion_matrix(cm, cm_plot_labels, title)
    macro_avg_precision, macro_avg_recall, macro_avg_F1, Specificity, Accuracy = model_evaluation(cm)
    if verbose:
        print("Title : ", title)
        print("Accuracy :", Accuracy)
        print("Macro average recall : ", macro_avg_recall)
        print("Specificity : ", Specificity)
        print("Macro average precision : ", macro_avg_precision)
        print("Macro average F1 : ", macro_avg_F1)
        print("\n")
    if Attack:
        Fooling_rate = 1 - Accuracy
        print("Fooling rate : ", Fooling_rate)


def true_negative(confusion_matrix):
    TN = []
    for label in range(len(confusion_matrix)):
        row = confusion_matrix[label, :]
        col = confusion_matrix[:, label]
        FN = row.sum()
        FP = col.sum()
        TN.append(confusion_matrix.sum() - FN - FP + confusion_matrix[label, label])
    return TN


def true_positive(confusion_matrix):
    TP = []
    for label in range(len(confusion_matrix)):
        TP.append(confusion_matrix[label, label])
    return TP


def false_negative(confusion_matrix):
    FN = []
    for label in range(len(confusion_matrix)):
        FN.append(confusion_matrix[label, :].sum() - confusion_matrix[label, label])
    return FN


def false_positive(confusion_matrix):
    FP = []
    for label in range(len(confusion_matrix)):
        FP.append(confusion_matrix[:, label].sum() - confusion_matrix[label, label])
    return FP


def model_evaluation(confusion_matrix):
    FP = false_positive(confusion_matrix)
    FN = false_negative(confusion_matrix)
    TP = true_positive(confusion_matrix)
    TN = true_negative(confusion_matrix)

    TotalFP = sum(FP)
    TotalFN = sum(FN)
    TotalTP = sum(TP)
    TotalTN = sum(TN)

    specificity_list = list(map(truediv, TN, list(map(add, TN, FP))))

    Precision_list = list(map(truediv, TP, list(map(add, TP, FP))))

    Recall_list = list(map(truediv, TP, list(map(add, TP, FN))))

    specificity_avg = np.around(sum(specificity_list) / len(specificity_list), 4)

    macro_avg_precision = np.around(sum(Precision_list) / len(Precision_list), 4)

    macro_avg_recall = np.around(sum(Recall_list) / len(Recall_list), 4)

    macro_avg_F1 = np.around(
        2 * ((macro_avg_precision * macro_avg_precision) / (macro_avg_precision + macro_avg_recall)), 4)

    Accuracy = np.around(confusion_matrix.trace() / confusion_matrix.sum(), 4)

    return macro_avg_precision, macro_avg_recall, macro_avg_F1, specificity_avg, Accuracy


def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # plt.title(title)
    # plt.colorbar()
    plt.figure(figsize=(5, 5))
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        cm = np.around(cm, 1)
        # print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


history = np.load('Saves/Hitsory/history_InceptionV3_v3.npy', allow_pickle='TRUE').item()
plot_curves(history)

with open("./Saves/ConfusionMatrixes/ConfusionMatrix_InceptionV3.pkl", "rb") as f:
    cm_InceptionV3= pickle.load(f)

with open("./Saves/ConfusionMatrixes/ConfusionMatrix_InceptionV3_FGSM.pkl", "rb") as f:
    cm_InceptionV3_FGSM = pickle.load(f)

with open("./Saves/ConfusionMatrixes/ConfusionMatrix_InceptionV3_Retrained_FGSM.pkl", "rb") as f:
    cm_InceptionV3_Retrained_FGSM = pickle.load(f)

with open("./Saves/ConfusionMatrixes/ConfusionMatrix_InceptionV3_FGSM_Compressed_Flipped.pkl", "rb") as f:
    cm_InceptionV3_FGSM_Flipped_Compressed = pickle.load(f)

with open("./Saves/ConfusionMatrixes/ConfusionMatrix_InceptionV3_Retrained_model_FGSM_Compressed_Flipped.pkl", "rb") as f:
    cm_InceptionV3_Retrained_FGSM_Flipped_Compressed = pickle.load(f)

with open("./Saves/ConfusionMatrixes/ConfusionMatrix_InceptionV3_NonTargetedUAP.pkl", "rb") as f:
    cm_InceptionV3_NonTargetedUAP = pickle.load(f)

with open("./Saves/ConfusionMatrixes/ConfusionMatrix_InceptionV3_Retrained_NonTargetedUAP.pkl", "rb") as f:
    cm_InceptionV3_Retrained_NonTargetedUAP = pickle.load(f)

with open("./Saves/ConfusionMatrixes/CM_RTInception_UAP_Compressed_Flipped.pkl", "rb") as f:
    cm_InceptionV3_Retrained_UAP_Flipped_Compressed = pickle.load(f)

with open("./Saves/ConfusionMatrixes/CM_Inception_UAP_Compressed_Flipped.pkl", "rb") as f:
    cm_InceptionV3_UAP_Flipped_Compressed = pickle.load(f)

plot_metrics(cm_InceptionV3,"Matrix without attack nor defenses",True,True,False)
plot_metrics(cm_InceptionV3_FGSM,"Inception V3 FGSM",True,True,True)
plot_metrics(cm_InceptionV3_Retrained_FGSM,"Inception V3 FGSM afte Retraining",True,True,True)
plot_metrics(cm_InceptionV3_FGSM_Flipped_Compressed,"Inception V3 FGSM after Flipping and compressing",True,True,True)
plot_metrics(cm_InceptionV3_Retrained_FGSM_Flipped_Compressed,"Inception V3 FGSM after Retraining + Flipping and compressing",True,True,True)
plot_metrics(cm_InceptionV3_NonTargetedUAP,"Inception V3 Non Targeted UAP",True,True,True)

experiences = ["InceptionV3", "FGSM attack", "Retraining defense","LLT Defense", "LLT+Retrain Defense"]
List_FGSM = [cm_InceptionV3,
                   cm_InceptionV3_FGSM,
                   cm_InceptionV3_Retrained_FGSM,
                   cm_InceptionV3_FGSM_Flipped_Compressed,
                   cm_InceptionV3_Retrained_FGSM_Flipped_Compressed]

plot_graph(List_FGSM,"Results",experiences)

experiences = ["InceptionV3", "UAP attack", "Retraining defense","LLT Defense", "LLT+Retrain Defense"]
List_UAP = [cm_InceptionV3,
                   cm_InceptionV3_NonTargetedUAP,
                   cm_InceptionV3_Retrained_NonTargetedUAP,
                   cm_InceptionV3_UAP_Flipped_Compressed,
                   cm_InceptionV3_Retrained_UAP_Flipped_Compressed]
plot_graph(List_UAP,"Results",experiences)

C_as_H_InceptionV3 = print_table(cm_InceptionV3)
C_as_H_FGSM = print_table(cm_InceptionV3_FGSM)
C_as_H_UAP = print_table(cm_InceptionV3_NonTargetedUAP)
C_as_H_FGSM_RT = print_table(cm_InceptionV3_Retrained_FGSM)
C_as_H_UAP_RT = print_table(cm_InceptionV3_Retrained_NonTargetedUAP)
C_as_H_LLT_FGSM = print_table(cm_InceptionV3_FGSM_Flipped_Compressed)
C_as_H_LLT_UAP = print_table(cm_InceptionV3_UAP_Flipped_Compressed)
C_as_H_FGSM_RT_Flipped = print_table(cm_InceptionV3_Retrained_FGSM_Flipped_Compressed)
C_as_H_UAP_RT_Flipped = print_table(cm_InceptionV3_Retrained_UAP_Flipped_Compressed)
experiment = ["base model","Attack method","Retraining defense","LLT defense", "LLT+Retrain Defense"]
List_C_as_H_FGSM = [C_as_H_InceptionV3,C_as_H_FGSM,C_as_H_FGSM_RT,C_as_H_LLT_FGSM,C_as_H_FGSM_RT_Flipped]

List_C_as_H_UAP = [C_as_H_InceptionV3,C_as_H_UAP,C_as_H_UAP_RT,C_as_H_LLT_UAP,C_as_H_UAP_RT_Flipped]
plt.xticks(fontsize=9)
plt.title("Cancerous as Healthy evolution")
plt.plot(experiment, List_C_as_H_FGSM, 'g', label='FGSM')
plt.plot(experiment, List_C_as_H_UAP, 'b', label='UAP')

plt.xlabel('Different experiments')
plt.ylabel('Percentage')
plt.legend()
plt.show()

plot_confusion_matrix(cm_InceptionV3_UAP_Flipped_Compressed,cm_plot_labels)