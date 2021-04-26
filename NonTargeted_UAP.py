import argparse
import numpy as np
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from FGSM_attack import create_adversarial_pattern

Validation_set = "E:\\DataSet\\ISIC2018\\ISIC_Validation\\"
model = load_model("./Saves/Models/InceptionV3_Model_2.h5")

val_datagen = ImageDataGenerator(
    rescale=1. / 255.,
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
)

val_ds = val_datagen.flow_from_directory(
    Validation_set,
    target_size=(224, 224),
    color_mode="rgb",
    classes=None,
    class_mode="categorical",
    batch_size=1,
    shuffle=False,
    seed=False,
    interpolation="bilinear",
    follow_links=False)


def calculate_data(dataset, norm=False, ):
    if norm:
        mean_l2_train = 0
        mean_inf_train = 0
        for e in range(len(dataset)):
            i = next(dataset)
            image = i[0]
            mean_l2_train += np.linalg.norm(image[:, :, 0].flatten(), ord=2)
            mean_inf_train += np.abs(image[:, :, 0].flatten()).max()

        mean_l2_train /= len(dataset)
        mean_inf_train /= len(dataset)

        if norm:
            return mean_l2_train, mean_inf_train


def UniversalPerturbation(classifier, attacker, delta, attacker_targeted, attacker_eps, max_iter, eps, norm):
    params = [classifier, attacker, delta, attacker_target]
    noise = 0
    fooling_rate = 0.0
    nb_instances = len(x)
    pred_y = classifier.predict(Validation_set, batch_size=1)
    pred_y_max = np.argmax(pred_y, axis=1)
    nb_iter = 0

    while fooling_rate < 1. - delta and nb_iter < max_iter:
        # Go through all the examples randomly
        rnd_idx = random.sample(range(nb_instances), nb_instances)

        # Go through the data set and compute the perturbation increments sequentially
        for j, ex in enumerate(x[rnd_idx]):
            x_i = ex[None, ...]

            current_label = np.argmax(self.classifier.predict(x_i + noise)[0])
            original_label = np.argmax(pred_y[rnd_idx][j])

            if current_label == original_label:
                # Compute adversarial perturbation
                adv_xi = attacker.generate(x_i + noise)
                new_label = np.argmax(self.classifier.predict(adv_xi)[0])

                # If the class has changed, update v
                if current_label != new_label:
                    noise = adv_xi - x_i

                    # Project on L_p ball
                    noise = projection(noise, self.eps, self.norm)
        nb_iter += 1

        # Apply attack and clip
        x_adv = x + noise
        if hasattr(self.classifier, 'clip_values') and self.classifier.clip_values is not None:
            clip_min, clip_max = self.classifier.clip_values
            x_adv = np.clip(x_adv, clip_min, clip_max)

        # Compute the error rate
        y_adv = np.argmax(self.classifier.predict(x_adv, batch_size=1), axis=1)
        fooling_rate = np.sum(pred_y_max != y_adv) / nb_instances
    return 0


def generate(Inputs, Labels, classifier):
    # Init universal perturbation
    noise = 0
    fooling_rate = 0.0
    nb_instances = len(Inputs)

    # Instantiate the middle attacker and get the predicted labels
    pred_y = classifier.predict(Inputs, batch_size=1)
    pred_y_max = np.argmax(pred_y, axis=1)

    # Start to generate the adversarial examples
    nb_iter = 0
    while fooling_rate < 1. - delta and nb_iter < max_iter:
        # Go through all the examples randomly
        rnd_idx = random.sample(range(nb_instances), nb_instances)

        # Go through the data set and compute the perturbation increments sequentially
        for j, ex in enumerate(x[rnd_idx]):
            x_i = ex[None, ...]

            current_label = np.argmax(self.classifier.predict(x_i + noise)[0])
            original_label = np.argmax(pred_y[rnd_idx][j])

            if current_label == original_label:
                # Compute adversarial perturbation
                adv_xi = attacker.generate(x_i + noise)
                new_label = np.argmax(self.classifier.predict(adv_xi)[0])

                # If the class has changed, update v
                if current_label != new_label:
                    noise = adv_xi - x_i

                    # Project on L_p ball
                    noise = projection(noise, self.eps, self.norm)
        nb_iter += 1

        # Apply attack and clip
        x_adv = x + noise
        if hasattr(self.classifier, 'clip_values') and self.classifier.clip_values is not None:
            clip_min, clip_max = self.classifier.clip_values
            x_adv = np.clip(x_adv, clip_min, clip_max)

        # Compute the error rate
        y_adv = np.argmax(self.classifier.predict(x_adv, batch_size=1), axis=1)
        fooling_rate = np.sum(pred_y_max != y_adv) / nb_instances

    self.fooling_rate = fooling_rate
    self.converged = nb_iter < self.max_iter
    self.noise = noise

    return x_adv


def set_art(model, norm_str, eps, mean_l2_train, mean_linf_train):
    classifier = KerasClassifier(model=model)
    if norm_str == 'l2':
        norm = 2
        scaled_eps = mean_l2_train / 128.0 * eps
    elif norm_str == 'linf':
        norm = np.inf
        scaled_eps = mean_linf_train / 128.0 * eps
    return classifier, norm, scaled_eps


mean_l2_train, mean_inf_train = calculate_data(val_ds, norm=True)
norm = "l2"
eps = 0.04


classifier, norm, eps = set_art(
    model=model,
    norm_str=norm,
    eps = eps,
    mean_l2_train=mean_l2_train,
    mean_linf_train=mean_inf_train)

adv_crafter = UniversalPerturbation(
    model,
    attacker='fgsm',
    delta=0.000001,
    attacker_params={'targeted': False, 'eps': 0.0024},
    max_iter=15,
    eps=eps,
    norm=norm)

while fooling_rate < 1. - self.delta and nb_iter < self.max_iter:
    # Go through all the examples randomly
    rnd_idx = random.sample(range(nb_instances), nb_instances)

    # Go through the data set and compute the perturbation increments sequentially
    for j, ex in enumerate(x[rnd_idx]):
        x_i = ex[None, ...]

        current_label = np.argmax(self.classifier.predict(x_i + noise)[0])
        original_label = np.argmax(pred_y[rnd_idx][j])

        if current_label == original_label:
            # Compute adversarial perturbation
            adv_xi = attacker.generate(x_i + noise)
            new_label = np.argmax(self.classifier.predict(adv_xi)[0])

            # If the class has changed, update v
            if current_label != new_label:
                noise = adv_xi - x_i

                # Project on L_p ball
                noise = projection(noise, self.eps, self.norm)
    nb_iter += 1

    # Apply attack and clip
    x_adv = x + noise
    if hasattr(self.classifier, 'clip_values') and self.classifier.clip_values is not None:
        clip_min, clip_max = self.classifier.clip_values
        x_adv = np.clip(x_adv, clip_min, clip_max)

    # Compute the error rate
    y_adv = np.argmax(self.classifier.predict(x_adv, batch_size=1), axis=1)
    fooling_rate = np.sum(pred_y_max != y_adv) / nb_instances

self.fooling_rate = fooling_rate
self.converged = nb_iter < self.max_iter
self.noise = noise
logger.info('Success rate of universal perturbation attack: %.2f%%', fooling_rate)

return x_adv