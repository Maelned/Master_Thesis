import numpy as np
import matplotlib.pyplot as plt
import pickle
import itertools
from operator import truediv, add, mul


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

def balanced_accuracy(confusion_matrix):
    balanced_accuracy = 0
    for label in range(len(confusion_matrix)):
        balanced_accuracy += confusion_matrix[label,label] / confusion_matrix[label, :].sum()
    balanced_accuracy = balanced_accuracy / len(confusion_matrix)
    print(balanced_accuracy)
    return balanced_accuracy

def model_evaluation(confusion_matrix):
    FP = false_positive(confusion_matrix)
    FN = false_negative(confusion_matrix)
    TP = true_positive(confusion_matrix)
    TN = true_negative(confusion_matrix)
    TotalFP = sum(FP)
    TotalFN = sum(FN)
    TotalTP = sum(TP)
    TotalTN = sum(TN)

    AvgFP = np.mean(FP)
    AvgFN = np.mean(FN)
    AvgTP = np.mean(TP)
    AvgTN = np.mean(TN)

    Precision_list = list(map(truediv, TP, list(map(add, TP, FP))))

    Recall_list = list(map(truediv, TP, list(map(add, TP, FN))))

    macro_avg_precision = np.around(sum(Precision_list) / len(Precision_list),4)

    macro_avg_recall = sum(Recall_list) / len(Recall_list)

    micro_avg_precision = np.around(TotalTP / confusion_matrix.sum(),4)

    micro_avg_recall = np.around(TotalTP / confusion_matrix.sum(),4)


    macro_avg_F1 = 2 * ( (macro_avg_precision * macro_avg_precision) / (macro_avg_precision + macro_avg_recall))

    Balanced_accuracy = balanced_accuracy(confusion_matrix)

    # Sensitivity, hit rate, recall, or true positive rate
    Avg_Sensitivity = np.around(AvgTP / (AvgTP + AvgFN), 2)
    # Specificity or true negative rate
    Avg_Specificity = np.around(AvgTN / (AvgTN + AvgFP), 2)
    # Precision or positive predictive value
    Avg_Precision = np.around(AvgTP / (AvgTP + AvgFP), 2)
    # Negative predictive value
    Avg_NegativePredictiveValue = np.around(AvgTN / (AvgTN + AvgFN), 2)
    # Fall out or false positive rate
    Avg_FallOut = np.around(AvgFP / (AvgFP + AvgTN), 2)
    # False negative rate
    Avg_FalseNegativeRate = np.around(AvgFN / (AvgTP + AvgFN), 2)
    # False discovery rate
    Avg_FalseDiscoveryRate = np.around(AvgFP / (AvgTP + AvgFP), 2)

    # Sensitivity, hit rate, recall, or true positive rate
    Sensitivity = np.around(TotalTP / (TotalTP + TotalFN), 2)
    # Specificity or true negative rate
    Specificity = np.around(TotalTN / (TotalTN + TotalFP), 2)
    # Precision or positive predictive value
    Precision = np.around(TotalTP / (TotalTP + TotalFP), 2)
    # Negative predictive value
    NegativePredictiveValue = np.around(TotalTN / (TotalTN + TotalFN), 2)
    # Fall out or false positive rate
    FallOut = np.around(TotalFP / (TotalFP + TotalTN), 2)
    # False negative rate
    FalseNegativeRate = np.around(TotalFN / (TotalTP + TotalFN), 2)
    # False discovery rate
    FalseDiscoveryRate = np.around(TotalFP / (TotalTP + TotalFP), 2)

    ErrorRate = np.around((TotalFP + TotalFN) / confusion_matrix.sum(), 2)
    Accuracy = np.around(confusion_matrix.trace() / confusion_matrix.sum(), 2)

    # return Sensitivity,Specificity,Precision,NegativePredictiveValue,FallOut,FalseNegativeRate,FalseDiscoveryRate,ErrorRate,Accuracy,Avg_Sensitivity,Avg_Specificity,Avg_Precision,Avg_NegativePredictiveValue,Avg_FallOut,Avg_FalseNegativeRate,Avg_FalseDiscoveryRate, micro_avg_precision, macro_avg_precision, micro_avg_recall, macro_avg_recall
    return micro_avg_precision, macro_avg_precision, micro_avg_recall, macro_avg_recall, macro_avg_F1, Specificity, Sensitivity,FalseNegativeRate,Accuracy,Balanced_accuracy


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
        print("Normalized confusion matrix")
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


history = np.load('./Saves/Hitsory/history_InceptionV3_2.npy', allow_pickle='TRUE').item()

with open("./Saves/ConfusionMatrixes/ConfusionMatrix_InceptionV3_2.pkl", "rb") as f:
    cm_inception2 = pickle.load(f)

with open("./Saves/ConfusionMatrixes/ConfusionMatrix_InceptionV3_2_AttackedModel_5%25.pkl", "rb") as f:
    cm_attacked5 = pickle.load(f)

with open("./Saves/ConfusionMatrixes/ConfusionMatrix_InceptionV3_2_AttackedModel_10%25.pkl", "rb") as f:
    cm_attacked10 = pickle.load(f)

with open("./Saves/ConfusionMatrixes/ConfusionMatrix_InceptionV3_2_AttackedModel_15%25.pkl", "rb") as f:
    cm_attacked15 = pickle.load(f)

with open("./Saves/ConfusionMatrixes/ConfusionMatrix_InceptionV3_2_AttackedModel_20%25.pkl", "rb") as f:
    cm_attacked20 = pickle.load(f)

with open("./Saves/ConfusionMatrixes/ConfusionMatrix_InceptionV3_2_AttackedModel_30%25.pkl", "rb") as f:
    cm_attacked30 = pickle.load(f)

with open("./Saves/ConfusionMatrixes/ConfusionMatrix_InceptionV3_2_AttackedModel_45%25.pkl", "rb") as f:
    cm_attacked45 = pickle.load(f)

with open("./Saves/ConfusionMatrixes/ConfusionMatrix_InceptionV3_2_AttackedModel_60%25.pkl", "rb") as f:
    cm_attacked60 = pickle.load(f)

with open("./Saves/ConfusionMatrixes/ConfusionMatrix_BeforeFGSM.pkl", "rb") as f:
    cm_Before_FGSM = pickle.load(f)

with open("./Saves/ConfusionMatrixes/ConfusionMatrix_AfterFGSM.pkl", "rb") as f:
    cm_After_FGSM = pickle.load(f)

cm_plot_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# plot_confusion_matrix(cm=cm_inception2, classes=cm_plot_labels, title='Confusion Matrix Inception V3')
# plot_confusion_matrix(cm=cm_attacked5, classes=cm_plot_labels, title='Confusion Matrix Inception V3 with 5% labels modified')
# plot_confusion_matrix(cm=cm_attacked10, classes=cm_plot_labels, title='Confusion Matrix Inception V3 with 10% labels modified')
# plot_confusion_matrix(cm=cm_attacked15, classes=cm_plot_labels, title='Confusion Matrix Inception V3 with 15% labels modified')
# plot_confusion_matrix(cm=cm_attacked20, classes=cm_plot_labels, title='Confusion Matrix Inception V3 with 20% labels modified')
plot_confusion_matrix(cm=cm_attacked30, classes=cm_plot_labels,
                      title='Confusion Matrix Inception V3 with 30% labels modified')
plot_confusion_matrix(cm=cm_attacked45, classes=cm_plot_labels,
                      title='Confusion Matrix Inception V3 with 45% labels modified')
plot_confusion_matrix(cm=cm_attacked60, classes=cm_plot_labels,
                      title='Confusion Matrix Inception V3 with 60% labels modified')
# plot_confusion_matrix(cm=cm_Before_FGSM, classes=cm_plot_labels, title='Confusion Matrix Inception V3 before FGSM attack')
# plot_confusion_matrix(cm=cm_After_FGSM, classes=cm_plot_labels, title='Confusion Matrix Inception V3 after FGSM attack')


loss_train = history['loss']
loss_val = history['val_loss']

acc_train = history['categorical_accuracy']
acc_val = history['val_categorical_accuracy']

epochs = range(0, 50)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss Inception V3')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, acc_train, 'g', label='Training acc')
plt.plot(epochs, acc_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy Inception V3')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

micro_avg_precision, macro_avg_precision, micro_avg_recall, macro_avg_recall, macro_avg_F1,Specificity, Sensitivity,FalseNegativeRate,Accuracy, Balanced_accuracy = model_evaluation(cm_inception2)

print("Metrics for cm_inceptionV3 :")
print("Micro average precision : ",micro_avg_precision)
print("Micro average recall : ",micro_avg_recall)

print("Macro average recall : ",macro_avg_precision)
print("Macro average recall : ",macro_avg_recall)

print("Macro average F1 : ", macro_avg_F1)

print("Specificity : ", Specificity)
print("Sensitivity : ", Sensitivity)
print("FalseNegativeRate : ", FalseNegativeRate)

print("Balanced_accuracy :", Balanced_accuracy)


micro_avg_precision, macro_avg_precision, micro_avg_recall, macro_avg_recall, macro_avg_F1,Specificity, Sensitivity,FalseNegativeRate, Accuracy,Balanced_accuracy = model_evaluation(cm_attacked5)
print("metrics for the same model with 5% modified labels :")
print("Micro average precision : ",micro_avg_precision)
print("Micro average recall : ",micro_avg_recall)

print("Macro average recall : ",macro_avg_precision)
print("Macro average recall : ",macro_avg_recall)

print("Macro average F1 : ", macro_avg_F1)

print("Specificity : ", Specificity)
print("Sensitivity : ", Sensitivity)
print("FalseNegativeRate : ", FalseNegativeRate)

print("Accuracy :", Accuracy)
print("Balanced_accuracy :", Balanced_accuracy)