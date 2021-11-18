import numpy as np
import matplotlib.pyplot as plt
import cv2

def iouScore(predictions, test_masks): 
    intersection = np.logical_and(test_masks, predictions)
    union = np.logical_or(test_masks, predictions)
    iou_score = np.sum(intersection) / np.sum(union)
    print("IoU-score: ", iou_score)

    return iou_score
    

def diceScore(predictions, test_masks):
    intersection = np.logical_and(test_masks, predictions)
    img_sum = np.sum(predictions) + np.sum(test_masks)
    if img_sum == 0: 
        return 1        #1 vai 0? testaa
    dice_score = 2*np.sum(intersection) / img_sum
    print("Dice-score: ", dice_score)

    return dice_score


def plotTrainingAcc(acc, val_acc, epochs): 
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.show()

def plotMasks(masks): 
    plt.figure(figsize=(10,10))
    for i in range(20):
        plt.subplot(4,5,i+1)
        plt.grid(False)
        plt.imshow(masks[i])
    plt.show()

#returns number of annotated pixels in given masks, method from: 
#https://github.com/hmhellstrom/ai-cancer-project/blob/python/image_functions.py
def countPosPixels(masks): 
    num_pixels = []
    for ind, mask in enumerate(masks):
        for slice in range(mask.shape[2]):
            num_pixels.append(sum([pixel > 0 for pixel in mask[:, :, slice].flatten()]))
    return num_pixels

#Visualize fused image with corresponding mask
def plotImgAndMask(img, mask): 
    #img = cv2.resize(img, (512,512),interpolation=cv2.INTER_AREA)
    #mask = cv2.resize(np.array(mask,dtype=np.float32), (512,512), interpolation=cv2.INTER_AREA)
    plt.imshow(img[:,:,0], cmap=plt.cm.gray, alpha=0.99) #0 refers to mri in stacked img
    plt.imshow(img[:,:,1], cmap=plt.cm.viridis, alpha=0.6) #1 refers to pet in stacked img
    plt.contour(mask[:,:,0],colors='red', linewidths=0.1)
    plt.axis("off")
    plt.show()


'''img49 = cv2.resize(X_test[49], (512,512), interpolation=cv2.INTER_CUBIC)
gt49 = cv2.resize(np.array(Y_test[49], dtype=np.int16), (512,512), interpolation=cv2.INTER_CUBIC)'''