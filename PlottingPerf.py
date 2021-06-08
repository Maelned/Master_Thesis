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
cm_plot_labels_healthy = ["Healthy", "Cancerous"]


def modif_cm(confusion_matrix):
    Healthy_or_not = [0, 0, 1, 1, 0, 0, 1]
    new_confusion_matrix = []

    Healthy = 0
    Healthy_as_Cancerous = 0
    Cancerous = 0
    Cancerous_as_Healthy = 0

    for label in range(len(confusion_matrix)):
        row = confusion_matrix[label, :]
        if Healthy_or_not[label]:
            for i in range(len(row)):
                if i == label:
                    Healthy += row[i]
                else:
                    if Healthy_or_not[i]:
                        Healthy += row[i]
                    else:
                        Healthy_as_Cancerous += row[i]
        else:
            for i in range(len(row)):
                if i == label:
                    Cancerous += row[i]
                else:
                    if Healthy_or_not[i]:
                        Cancerous_as_Healthy += row[i]
                    else:
                        Cancerous += row[i]
        List_Healthy = [Healthy, Healthy_as_Cancerous]
        List_Cancerous = [Cancerous_as_Healthy, Cancerous]
        new_confusion_matrix = [List_Healthy, List_Cancerous]
        new_confusion_matrix = np.array(new_confusion_matrix)
    return new_confusion_matrix


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

def plot_graph_healthy(multiple_cm, title, experiment):
    plt.xticks(fontsize=9)
    recall_tot,NPV_tot, Accuracy_tot, FNR_tot = [], [], [], []
    for i in multiple_cm:
        plot_metrics(i, "No title", False, True, True)
        Accuracy,recall, NPV, FNR = metrics_healthy(i)
        recall_tot.append(recall)
        FNR_tot.append(FNR)
        NPV_tot.append(NPV)
        Accuracy_tot.append(Accuracy)
    plt.title(title)

    plt.plot(experiment, Accuracy_tot, 'g', label='Accuracy')
    plt.plot(experiment, recall_tot, 'b', label='Recall')
    plt.plot(experiment, NPV_tot, 'r', label='NPV')
    plt.plot(experiment, FNR_tot, 'o', label='FNR')

    # plt.plot(experiment, Fooling_rate, label="Fooling_rate")
    plt.xlabel('Different experiments')
    plt.ylabel('Percentage')
    plt.legend()
    plt.show()


def plot_graph(multiple_cm, title, experiment):
    plt.xticks(fontsize=9)
    precision_tot, recall_tot, F1_tot, Specificity_tot, Accuracy_tot, Fooling_rate = [], [], [], [], [], []
    for i in multiple_cm:
        plot_metrics(i, "No title", False, True, True)
        macro_avg_recall, Specificity, Accuracy = model_evaluation(i)
        recall_tot.append(macro_avg_recall)
        Specificity_tot.append(Specificity)
        Accuracy_tot.append(Accuracy)
        Fooling_rate.append(1 - Accuracy)
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
    if len(cm) == 2:
        if plot_cm:
            plot_confusion_matrix(cm, cm_plot_labels_healthy, title)
        Accuracy, Recall, NPV,FNR = metrics_healthy(cm)

        if verbose:
            print("Title : ", title)
            print("Accuracy :", Accuracy)
            print("Recall : ", Recall)
            print("NPV : ", NPV)
            print("FNR: ", FNR)
            print("\n")
    else:
        if plot_cm:
            plot_confusion_matrix(cm, cm_plot_labels, title)

        macro_avg_recall, Specificity, Accuracy = model_evaluation(cm)
        if verbose:
            print("Title : ", title)
            print("Accuracy :", Accuracy)
            print("Macro average recall : ", macro_avg_recall)
            print("Specificity : ", Specificity)

        if Attack:
            Fooling_rate = 1 - Accuracy
            print("Fooling rate : ", Fooling_rate)
        print("\n")


def metrics_healthy(confusion_matrix):
    TN = confusion_matrix[0, 0]
    FP = confusion_matrix[0, 1]
    FN = confusion_matrix[1, 0]
    TP = confusion_matrix[1, 1]

    Accuracy = np.around((TP + TN) / (TN + FP + FN + TP), 4)
    Recall = np.around((TP / (TP + FN)), 4)
    NPV = np.around(TN / (TN + FN), 4)
    FNR = 1 - Recall

    return Accuracy, Recall, NPV, FNR


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
    Recall_list = list(map(truediv, TP, list(map(add, TP, FN))))
    specificity_avg = np.around(sum(specificity_list) / len(specificity_list), 4)

    macro_avg_recall = np.around(sum(Recall_list) / len(Recall_list), 4)

    Accuracy = np.around(confusion_matrix.trace() / confusion_matrix.sum(), 4)

    return macro_avg_recall, specificity_avg, Accuracy


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
# plot_curves(history)

with open("./Saves/ConfusionMatrixes/ConfusionMatrix_InceptionV3.pkl", "rb") as f:
    cm_InceptionV3 = pickle.load(f)

with open("./Saves/ConfusionMatrixes/ConfusionMatrix_InceptionV3_FGSM.pkl", "rb") as f:
    cm_InceptionV3_FGSM = pickle.load(f)

with open("./Saves/ConfusionMatrixes/ConfusionMatrix_InceptionV3_Retrained_FGSM.pkl", "rb") as f:
    cm_InceptionV3_FGSM_RT = pickle.load(f)

with open("./Saves/ConfusionMatrixes/ConfusionMatrix_InceptionV3_FGSM_Compressed_Flipped.pkl", "rb") as f:
    cm_InceptionV3_FGSM_LLT = pickle.load(f)

with open("./Saves/ConfusionMatrixes/ConfusionMatrix_InceptionV3_Retrained_model_FGSM_Compressed_Flipped.pkl",
          "rb") as f:
    cm_InceptionV3_FGSM_RT_LLT = pickle.load(f)

with open("./Saves/ConfusionMatrixes/ConfusionMatrix_InceptionV3_NonTargetedUAP.pkl", "rb") as f:
    cm_InceptionV3_UAP = pickle.load(f)

with open("./Saves/ConfusionMatrixes/ConfusionMatrix_InceptionV3_Retrained_NonTargetedUAP.pkl", "rb") as f:
    cm_InceptionV3_UAP_RT = pickle.load(f)

with open("./Saves/ConfusionMatrixes/CM_RTInception_UAP_Compressed_Flipped.pkl", "rb") as f:
    cm_InceptionV3_UAP_RT_LLT = pickle.load(f)

with open("./Saves/ConfusionMatrixes/CM_Inception_UAP_Compressed_Flipped.pkl", "rb") as f:
    cm_InceptionV3_UAP_LLT = pickle.load(f)

# plot_metrics(cm_InceptionV3,"Matrix without attack nor defenses",True,True,False)
# plot_metrics(cm_InceptionV3_FGSM,"Inception V3 FGSM",True,True,True)
# plot_metrics(cm_InceptionV3_FGSM_RT,"Inception V3 FGSM afte Retraining",True,True,True)
# plot_metrics(cm_InceptionV3_FGSM_LLT,"Inception V3 FGSM after Flipping and compressing",True,True,True)
# plot_metrics(cm_InceptionV3_FGSM_RT_LLT,"Inception V3 FGSM after Retraining + Flipping and compressing",True,True,True)
# plot_metrics(cm_InceptionV3_UAP,"Inception V3 Non Targeted UAP",True,True,True)
# plot_metrics(cm_InceptionV3_UAP_RT,"Inception V3 Non Targeted UAP",True,True,True)
# plot_metrics(cm_InceptionV3_UAP_LLT,"Inception V3 Non Targeted UAP",True,True,True)
# plot_metrics(cm_InceptionV3_UAP_RT_LLT,"Inception V3 Non Targeted UAP",True,True,True)



experiences = ["InceptionV3", "FGSM attack", "Retraining defense", "LLT Defense", "LLT+Retrain Defense"]
List_FGSM = [cm_InceptionV3,
             cm_InceptionV3_FGSM,
             cm_InceptionV3_FGSM_RT,
             cm_InceptionV3_FGSM_LLT,
             cm_InceptionV3_FGSM_RT_LLT]
# plot_graph(List_FGSM,"",experiences)

# plot_metrics(cm_InceptionV3,"Inception V3",True,True,False)
experiences_UAP = ["InceptionV3", "UAP attack", "Retraining defense", "LLT Defense", "LLT+Retrain Defense"]
List_UAP = [cm_InceptionV3,
            cm_InceptionV3_UAP,
            cm_InceptionV3_UAP_RT,
            cm_InceptionV3_UAP_LLT,
            cm_InceptionV3_UAP_RT_LLT]
# plot_graph(List_UAP,"",experiences)


experiences_Healthy = ["InceptionV3", "UAP attack", "Retraining defense", "LLT Defense", "LLT+Retrain Defense"]

# plot_metrics(cm_InceptionV3, "Inception V3", True, True, False)
cm_InceptionV3_Health = modif_cm(cm_InceptionV3)

# plot_metrics(cm_InceptionV3_Health, "Inception V3 Health oriented", True, True, False)


cm_InceptionV3_FGSM_Health = modif_cm(cm_InceptionV3_FGSM)
cm_InceptionV3_FGSM_RT_Health = modif_cm(cm_InceptionV3_FGSM_RT)
cm_InceptionV3_FGSM_LLT_Health = modif_cm(cm_InceptionV3_FGSM_LLT)
cm_InceptionV3_FGSM_RT_LLT_Health = modif_cm(cm_InceptionV3_FGSM_RT_LLT)

List_FGSM_Health = [cm_InceptionV3_Health,
                    cm_InceptionV3_FGSM_Health,
                    cm_InceptionV3_FGSM_RT_Health,
                    cm_InceptionV3_FGSM_LLT_Health,
                    cm_InceptionV3_FGSM_RT_LLT_Health]
# plot_graph_healthy(List_FGSM_Health,"",experiences_Healthy)
cm_InceptionV3_UAP_Health = modif_cm(cm_InceptionV3_UAP)
cm_InceptionV3_UAP_RT_Health = modif_cm(cm_InceptionV3_UAP_RT)
cm_InceptionV3_UAP_LLT_Health = modif_cm(cm_InceptionV3_UAP_LLT)
cm_InceptionV3_UAP_RT_LLT_Health = modif_cm(cm_InceptionV3_UAP_RT_LLT)

List_UAP_Health = [cm_InceptionV3_Health,
                   cm_InceptionV3_UAP_Health,
                   cm_InceptionV3_UAP_RT_Health,
                   cm_InceptionV3_UAP_LLT_Health,
                   cm_InceptionV3_UAP_RT_LLT_Health]

a = 0
for i in List_UAP_Health:
    plot_metrics(i,experiences_Healthy[a],True,True,True)
    a+=1

# plot_graph_healthy(List_UAP_Health,"",experiences_Healthy)
# ************************************** FGSM **********************************************
# plot_metrics(cm_InceptionV3_FGSM, "Inception V3 FGSM", True, True, True)
# new_cm = modif_cm(cm_InceptionV3_FGSM)
# plot_metrics(new_cm, "Inception V3 FGSM Health oriented", True, True, True)
#
# plot_metrics(cm_InceptionV3_FGSM_RT, "Inception V3 FGSM Retrained", True, True, True)
# new_cm = modif_cm(cm_InceptionV3_FGSM_RT)
# plot_metrics(new_cm, "Inception V3 FGSM Retrained Health oriented", True, True, True)
#
# plot_metrics(cm_InceptionV3_FGSM_LLT, "Inception V3 FGSM LLT", True, True, True)
new_cm = modif_cm(cm_InceptionV3_FGSM_LLT)
plot_metrics(new_cm, "Inception V3 FGSM LLT Health oriented", True, True, True)
#
# plot_metrics(cm_InceptionV3_FGSM_RT_LLT, "Inception V3 FGSM RT + LLT", True, True, True)
# new_cm = modif_cm(cm_InceptionV3_FGSM_RT_LLT)
# plot_metrics(new_cm, "Inception V3 FGSM RT + LLT Health oriented", True, True, True)

# ************************************** UAPs **********************************************


# # plot_metrics(cm_InceptionV3_UAP "Inception V3 UAP", True, True, True)
# new_cm = modif_cm(cm_InceptionV3_UAP)
# plot_metrics(new_cm, "Inception V3 UAP Health oriented", True, True, True)
#
# plot_metrics(cm_InceptionV3_UAP_RT, "Inception V3 UAP Retrained", True, True, True)
# new_cm = modif_cm(cm_InceptionV3_UAP_RT)
# plot_metrics(new_cm, "Inception V3 UAP Retrained Health oriented", True, True, True)
#
# plot_metrics(cm_InceptionV3_UAP_LLT, "Inception V3 UAP LLT", True, True, True)
# new_cm = modif_cm(cm_InceptionV3_UAP_LLT)
# plot_metrics(new_cm, "Inception V3 UAP LLT Health oriented", True, True, True)
#
# plot_metrics(cm_InceptionV3_UAP_RT_LLT, "Inception V3 UAP RT + LLT", True, True, True)
# new_cm = modif_cm(cm_InceptionV3_UAP_RT_LLT)
# plot_metrics(new_cm, "Inception V3 UAP RT + LLT Health oriented", True, True, True)
