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

    macro_avg_recall = np.around(sum(Recall_list) / len(Recall_list),4)

    macro_avg_F1 = np.around(2 * ((macro_avg_precision * macro_avg_precision) / (macro_avg_precision + macro_avg_recall)),4)

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

with open("./Saves/ConfusionMatrixes/ConfusionMatrix_InceptionV3_2_AttackedModel_75%25.pkl", "rb") as f:
    cm_attacked75 = pickle.load(f)

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

macro_avg_precision, macro_avg_recall, macro_avg_F1, Specificity, Accuracy = model_evaluation(
    cm_inception2)
print("Metrics for cm_inceptionV3 :")
print("Macro average recall : ", macro_avg_precision)
print("Macro average recall : ", macro_avg_recall)
print("Macro average F1 : ", macro_avg_F1)
print("Specificity : ", Specificity)
print("Accuracy :", Accuracy)

macro_avg_precision, macro_avg_recall, macro_avg_F1, Specificity, Accuracy = model_evaluation(
    cm_attacked5)
print("metrics for the same model with 5% modified labels :")
print("Macro average recall : ", macro_avg_precision)
print("Macro average recall : ", macro_avg_recall)
print("Macro average F1 : ", macro_avg_F1)
print("Specificity : ", Specificity)
print("Accuracy :", Accuracy)

macro_avg_precision, macro_avg_recall, macro_avg_F1, Specificity, Accuracy = model_evaluation(
    cm_attacked10)
print("metrics for the same model with 10% modified labels :")
print("Macro average recall : ", macro_avg_precision)
print("Macro average recall : ", macro_avg_recall)
print("Macro average F1 : ", macro_avg_F1)
print("Specificity : ", Specificity)
print("Accuracy :", Accuracy)

macro_avg_precision, macro_avg_recall, macro_avg_F1, Specificity, Accuracy = model_evaluation(
    cm_attacked15)
print("metrics for the same model with 15% modified labels :")
print("Macro average recall : ", macro_avg_precision)
print("Macro average recall : ", macro_avg_recall)
print("Macro average F1 : ", macro_avg_F1)
print("Specificity : ", Specificity)
print("Accuracy :", Accuracy)

macro_avg_precision, macro_avg_recall, macro_avg_F1, Specificity, Accuracy = model_evaluation(
    cm_attacked20)
print("metrics for the same model with 20% modified labels :")
print("Macro average recall : ", macro_avg_precision)
print("Macro average recall : ", macro_avg_recall)
print("Macro average F1 : ", macro_avg_F1)
print("Specificity : ", Specificity)
print("Accuracy :", Accuracy)

macro_avg_precision, macro_avg_recall, macro_avg_F1, Specificity, Accuracy = model_evaluation(
    cm_attacked30)
print("metrics for the same model with 30% modified labels :")
print("Macro average recall : ", macro_avg_precision)
print("Macro average recall : ", macro_avg_recall)
print("Macro average F1 : ", macro_avg_F1)
print("Specificity : ", Specificity)
print("Accuracy :", Accuracy)

macro_avg_precision, macro_avg_recall, macro_avg_F1, Specificity, Accuracy = model_evaluation(
    cm_attacked45)
print("metrics for the same model with 45% modified labels :")
print("Macro average recall : ", macro_avg_precision)
print("Macro average recall : ", macro_avg_recall)
print("Macro average F1 : ", macro_avg_F1)
print("Specificity : ", Specificity)
print("Accuracy :", Accuracy)

macro_avg_precision, macro_avg_recall, macro_avg_F1, Specificity, Accuracy = model_evaluation(
    cm_attacked60)
print("metrics for the same model with 60% modified labels :")
print("Macro average recall : ", macro_avg_precision)
print("Macro average recall : ", macro_avg_recall)
print("Macro average F1 : ", macro_avg_F1)
print("Specificity : ", Specificity)
print("Accuracy :", Accuracy)

macro_avg_precision, macro_avg_recall, macro_avg_F1, Specificity, Accuracy = model_evaluation(
    cm_attacked75)
print("metrics for the same model with 75% modified labels :")
print("Macro average recall : ", macro_avg_precision)
print("Macro average recall : ", macro_avg_recall)
print("Macro average F1 : ", macro_avg_F1)
print("Specificity : ", Specificity)
print("Accuracy :", Accuracy)

