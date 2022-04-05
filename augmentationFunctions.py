import imgaug.augmenters as iaa
from matplotlib.pyplot import axis
import numpy as np
import imgaug as ia


def augment(num_iter, train_images, train_masks, SEED):

    ia.seed(SEED)
    # augmentation pipeline
    augmentation = iaa.Sequential(
        [
            iaa.Fliplr(0.5),  # horizontal flips
            iaa.Affine(
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                rotate=(-15, 15),
            ),
        ]
    )

    for i in range(num_iter):  # apply augmentation
        augmented_train_imgs_i, augmented_train_masks_i = augmentation(
            images=train_images, segmentation_maps=train_masks
        )
        if i == 0:
            augmented_train_imgs = augmented_train_imgs_i
            augmented_train_masks = augmented_train_masks_i
        else:
            augmented_train_imgs = np.append(
                augmented_train_imgs, augmented_train_imgs_i, axis=0
            )
            augmented_train_masks = np.append(
                augmented_train_masks, augmented_train_masks_i, axis=0
            )

    return augmented_train_imgs, augmented_train_masks


# iaa.Sometimes( 0.5, iaa.GaussianBlur(sigma=(0, 0.5)), iaa.LinearContrast((0.75, 1.5))),
