import imgaug.augmenters as iaa
import numpy as np

def augment(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS, num_iter, train_images, train_masks): 

    #augmentaatio
    augmentation = iaa.Sequential([
        # 1. Flip
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        # 2. Affine
        iaa.Affine(translate_percent={"x": (-0.10, 0.10), "y": (-0.10, 0.10)},
                rotate=(-20, 20),
                ),
    ])

    augmented_train_imgs = [] 
    augmented_train_masks = []
    for i in range(num_iter): 
        augmented_train_imgs_i, augmented_train_masks_i = augmentation(images=train_images,segmentation_maps=train_masks)
        augmented_train_imgs.append(augmented_train_imgs_i)
        augmented_train_masks.append(augmented_train_masks_i)

    augmented_train_imgs = np.array(augmented_train_imgs)
    augmented_train_masks = np.array(augmented_train_masks)

    new_length = len(augmented_train_imgs)*len(augmented_train_imgs[0]) #Uuden matriisin kuvamäärä

    augmented_train_imgs = augmented_train_imgs.reshape(new_length,IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)
    augmented_train_masks = augmented_train_masks.reshape(new_length,IMG_HEIGHT,IMG_WIDTH,1)

    return augmented_train_imgs, augmented_train_masks