import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import imageFunctions
import random
import augmentationFunctions
from dataToExcel import loadToExcel
import evaluationFunctions
import matplotlib.pyplot as plt
import matplotlib
from models import create_unet
from sklearn.utils import shuffle
import segmentation_models as sm
from cv2 import resize, INTER_CUBIC
from glob import glob
import nibabel as nib

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 2
SEED = 1
random.seed(SEED) # use seed when reproducible testing is desired

# Get patients with cancer

pos_mri_path = r"C:\Users\joona\Documents\Tohtorikoulu\kuvadata 2_artikkeli\positiiviset\*[0-9]*\*mri*\*.img"
pos_mri_paths = glob(pos_mri_path, recursive=True)
pos_mri = []
for path in pos_mri_paths:
    pos_mri.append(
        resize(
            nib.load(path).get_fdata(),
            (IMG_HEIGHT, IMG_WIDTH),
            interpolation=INTER_CUBIC,
        )
    )


pos_pet_path = r"C:\Users\joona\Documents\Tohtorikoulu\kuvadata 2_artikkeli\positiiviset\*[0-9]*\*pet*\*.img"
pos_pet_paths = glob(pos_pet_path, recursive=True)
pos_pet = []
for path in pos_pet_paths:
    pos_pet.append(
        resize(
            nib.load(path).get_fdata(),
            (IMG_HEIGHT, IMG_WIDTH),
            interpolation=INTER_CUBIC,
        )
    )


pos_mask_path = r"C:\Users\joona\Documents\Tohtorikoulu\kuvadata 2_artikkeli\positiiviset\*[0-9]*\*mask*\*.img"
pos_mask_paths = glob(pos_mask_path, recursive=True)
pos_masks = []
for path in pos_mask_paths:
    pos_masks.append(
        resize(
            nib.load(path).get_fdata(),
            (IMG_HEIGHT, IMG_WIDTH),
            interpolation=INTER_CUBIC,
        )
    )

# Fuse positives
fused_patients_pos = imageFunctions.fuseImages(pos_mri, pos_pet)
pos_mri.clear()
pos_pet.clear()

# Masks to binary
for i, mask in enumerate(pos_masks):
    pos_masks[i] = np.array(mask, dtype=bool)

# remove images without cancer from patient stacks
inds = []

for i in range(len(fused_patients_pos)):
    for j in range(fused_patients_pos[i].shape[2]):
        if np.max(pos_masks[i][:, :, j]) == 0:
            inds.append(j)
    fused_patients_pos[i] = np.delete(fused_patients_pos[i], obj=inds, axis=2)
    pos_masks[i] = np.delete(pos_masks[i], obj=inds, axis=2)
    inds.clear()


# Get patients without cancer


neg_mri_path = r"C:\Users\joona\Documents\Tohtorikoulu\kuvadata 2_artikkeli\negatiiviset\*[0-9]*\*mri*\*.img"
neg_mri_paths = glob(neg_mri_path, recursive=True)
neg_mri = []
for path in neg_mri_paths:
    neg_mri.append(
        resize(
            nib.load(path).get_fdata(),
            (IMG_HEIGHT, IMG_WIDTH),
            interpolation=INTER_CUBIC,
        )
    )


neg_pet_path = r"C:\Users\joona\Documents\Tohtorikoulu\kuvadata 2_artikkeli\negatiiviset\*[0-9]*\*pet*\*.img"
neg_pet_paths = glob(neg_pet_path, recursive=True)
neg_pet = []
for path in neg_pet_paths:
    neg_pet.append(
        resize(
            nib.load(path).get_fdata(),
            (IMG_HEIGHT, IMG_WIDTH),
            interpolation=INTER_CUBIC,
        )
    )


# Fuse negatives
fused_patients_neg = imageFunctions.fuseImages(neg_mri, neg_pet)
neg_mri.clear()
neg_pet.clear()

# Merge fused
all_fused = fused_patients_pos + fused_patients_neg


# negative masks
neg_masks = []
for i, img in enumerate(fused_patients_neg):
    neg_mask = np.zeros(shape=(IMG_HEIGHT, IMG_WIDTH, img.shape[2]))
    neg_masks.append(neg_mask)


all_masks = pos_masks + neg_masks

# Function for splitting and preprocessing the image data
def preprocess(all_fused, all_masks):

    # Divide patientwise
    stratify_val = [0 if mask.any() else 1 for mask in all_masks]

    train_ids, test_ids, train_labels, test_labels = train_test_split(
        all_fused, all_masks, test_size=0.20, random_state=SEED, stratify=stratify_val
    )

    # Separate positives and negatives and random sample negatives to match the amount of positives

    neg_train_ids = []
    neg_train_labels = []
    pos_train_ids = []
    pos_train_labels = []
    for i, img in enumerate(train_ids):
        if np.max(train_labels[i]) == 0:
            neg_train_ids.append(train_ids[i])
            neg_train_labels.append(train_labels[i])
        else:
            pos_train_ids.append(train_ids[i])
            pos_train_labels.append(train_labels[i])
    neg_train_slices = imageFunctions.getSlices(neg_train_ids)
    pos_train_slices = imageFunctions.getSlices(pos_train_ids)
    pos_train_mask_slices = imageFunctions.getMaskSlices(pos_train_labels)
    neg_train_mask_slices = imageFunctions.getMaskSlices(neg_train_labels)

    # Random sample negative slices to match the number of positive slices
    neg_train_slices = random.sample(neg_train_slices, len(pos_train_slices))
    neg_train_mask_slices = random.sample(neg_train_mask_slices, len(pos_train_mask_slices))

    # merge
    train_slices = pos_train_slices + neg_train_slices
    train_mask_slices = pos_train_mask_slices + neg_train_mask_slices


    neg_test_ids = []
    neg_test_labels = []
    pos_test_ids = []
    pos_test_labels = []
    for i, img in enumerate(test_ids):
        if np.max(test_labels[i]) == 0:
            neg_test_ids.append(test_ids[i])
            neg_test_labels.append(test_labels[i])
        else:
            pos_test_ids.append(test_ids[i])
            pos_test_labels.append(test_labels[i])
    neg_test_slices = imageFunctions.getSlices(neg_test_ids)
    pos_test_slices = imageFunctions.getSlices(pos_test_ids)
    pos_test_mask_slices = imageFunctions.getMaskSlices(pos_test_labels)
    neg_test_mask_slices = imageFunctions.getMaskSlices(neg_test_labels)


    neg_test_slices = random.sample(neg_test_slices, len(pos_test_slices))
    neg_test_mask_slices = random.sample(neg_test_mask_slices, len(pos_test_mask_slices))


    test_slices = pos_test_slices + neg_test_slices
    test_mask_slices = pos_test_mask_slices + neg_test_mask_slices


    # shuffle slices
    train_slices, train_mask_slices = shuffle(
        train_slices, train_mask_slices, random_state=SEED
    )
    test_slices, test_mask_slices = shuffle(
        test_slices, test_mask_slices, random_state=SEED
    )
    # Normalize training set

    mri_min, mri_max, pet_min, pet_max = imageFunctions.getMinMax(
        train_slices
    )  # Get min and max values

    print(f"MRI-MIN: {mri_min}\nMRI-MAX: {mri_max}\nPET-MIN: {pet_min}\nPET-MAX: {pet_max}")
    imageFunctions.normalize(train_slices, mri_min, mri_max, pet_min, pet_max)

    X_train = np.array(train_slices, dtype=np.float32)
    Y_train = np.array(train_mask_slices, dtype=bool)

    # X_train = np.expand_dims(X_train, axis=3)
    Y_train = np.expand_dims(Y_train, axis=3)


    imageFunctions.normalize(test_slices, mri_min, mri_max, pet_min, pet_max)

    X_test = np.array(test_slices, dtype=np.float32)
    Y_test = np.array(test_mask_slices, dtype=bool)
    # X_test = np.expand_dims(X_test, axis=3)
    Y_test = np.expand_dims(Y_test, axis=3)

    return X_train, Y_train, X_test, Y_test



X_train, Y_train, X_test, Y_test = preprocess(all_fused, all_masks)


# Image augmentation
# X_train_aug, Y_train_aug = augmentationFunctions.augment(5, X_train, Y_train, SEED)

# # Try  adding sample weights
# X_train, Y_train, sample_weights = imageFunctions.add_sample_weights(
#     X_train, Y_train, 1.0, 3.0
# )


# Y_train = np.array(Y_train, dtype=np.float32)
# Y_test = np.array(Y_test, dtype=np.float32)




iou_scores = []
dice_scores = []

def train(X_train, Y_train, X_test, Y_test):


    model = create_unet(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    model.compile(
        "Adam",
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[sm.metrics.iou_score],
    )
    cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_iou_score", mode="max", patience=50, restore_best_weights=True
    )

    results = model.fit(
        x=X_train,
        y=Y_train,
        epochs=200,
        batch_size=32,
        validation_split=0.15,
        shuffle=True,
        callbacks=[cb],
    )

    acc = results.history["iou_score"]
    val_acc = results.history["val_iou_score"]
    epochs = range(1, len(acc) + 1)

    evaluationFunctions.plotTrainingAcc(acc, val_acc, epochs)

    print("Predicting for test set")
    preds = model.predict(X_test)
    preds_t = preds > 0.5

    evaluationFunctions.plotMasks(preds_t)
    evaluationFunctions.plotMasks(Y_test)

    iou_score = evaluationFunctions.iouScore(preds_t, Y_test)
    dice_score = evaluationFunctions.diceScore(preds_t, Y_test)
    iou_scores.append(iou_score)
    dice_scores.append(dice_score)
    model.save(
        r"C:\Users\joona\Documents\Tohtorikoulu\Uudet ajot\aug_fuusio_280322\unet_"
        + str(i + 1)
    )


print("Average IoU-score: ", np.average(iou_scores))
print("Average Dice-score: ", np.average(dice_scores))
loadToExcel("AUG_FUSED_50x_280322.xlsx", iou_scores, dice_scores)

# Brute forece method for gaining thresholds
def get_tholds(preds, Y_test):
    n = 0
    tholds = []
    scores = []
    for j in range(1000):
        n += 0.001
        temp_preds_t = preds >= n
        d = evaluationFunctions.diceScore(temp_preds_t, Y_test)
        scores.append(d)
        tholds.append(n)
    best_thold = tholds[scores.index(max(scores))]
    preds_t = preds >= best_thold

    return preds_t



med_model = tf.keras.models.load_model(r'C:\Users\joona\Documents\Tohtorikoulu\Uudet ajot\fuusio220222\unet_14', custom_objects={'iou_score': sm.metrics.iou_score})

#slicewise predictions
slice_dice_scores = []
slice_iou_scores = []
for i in range(len(X_test)):
    img = np.expand_dims(X_test[i], axis=0)
    pred = med_model.predict(img)
    pred_t = pred > 0.5
    dsc = evaluationFunctions.diceScore(pred_t, Y_test[i])
    iou = evaluationFunctions.iouScore(pred_t, Y_test[i])
    slice_dice_scores.append(dsc)
    slice_iou_scores.append(iou)

"""
Plot the images and masks of where real segmentation has happened

"""

# Set the thresholds for Dice scores
lower_threshold = 0
upper_threshold = 1

# Calculate the number of images that meet the criteria
num_images = sum(
    [lower_threshold < dice_score < upper_threshold for dice_score in dice_scores]
)

# Set the number of columns for the grid and calculate the number of rows
num_columns = 4
num_rows = int(np.ceil(num_images / num_columns))

# Create a figure
fig = plt.figure(figsize=(num_columns * 5, num_rows * 5))

plot_count = 1
for index, dice_score in enumerate(dice_scores):
    if lower_threshold < dice_score < upper_threshold:
        # Add a subplot to the grid
        ax = fig.add_subplot(num_rows, num_columns, plot_count)

        # Plot the image with channel 0 having higher alpha value
        ax.imshow(X_test[index, :, :, 0], alpha=0.99, cmap="gray")
        ax.imshow(X_test[index, :, :, 1], alpha=0.6, cmap="viridis")

        # Plot the mask using plt.contour
        plt.contour(Y_test[index, :, :, 0], levels=[0.5], colors="b")
        plt.contour(preds_t[index, :, :, 0], levels=[0.5], colors="r")

        # Set the title for each plot
        ax.set_title(f"Test Case: {index}, Dice Score: {dice_score:.2f}")

        # Create a custom legend for the contour label
        pred_legend = matplotlib.patches.Patch(color="red", label="Prediction")
        gt_legend = matplotlib.patches.Patch(color="blue", label="Ground truth")
        ax.legend(handles=[pred_legend, gt_legend], loc="upper right")

        # Increment the plot_count
        plot_count += 1

# Display the figure
plt.show()
