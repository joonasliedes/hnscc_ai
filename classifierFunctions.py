import numpy as np

def classifyPixelwise(predictions, groundtruths):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(predictions)): 
        if np.max(predictions[i])==True and np.max(groundtruths[i])==True: 
            tp += 1
        elif np.max(predictions[i])==True and np.max(groundtruths[i])==False: 
            fp += 1
            print("False positive at index: ",i)
        elif np.max(predictions[i])==False and np.max(groundtruths[i])==False: 
            tn += 1
        elif np.max(predictions[i])==False and np.max(groundtruths[i])==True: 
            fn += 1
            print("False negative at index: ",i)
    
    print("TP: {}\nFP: {}\nTN: {}\nFN: {}".format(tp, fp, tn, fn))

    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    acc = (tp + tn) / (tp + tn + fp + fp)

    print("Sensitivity is: {}".format(round(sens, 3)))
    print("Specificity is: {}".format(round(spec, 3)))
    print("Accuracy is: {}".format(round(acc, 3)))

#return number of true/false positives/negatives
#9 pixels set for the limit of reliable classification
def classify(predictions, groundtruths):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    predictions = np.array(predictions, dtype=np.int0)
    groundtruths = np.array(groundtruths, dtype=np.int0)
    fp_i = []
    fn_i = []
    for i in range(len(predictions)): 
        if np.sum(predictions[i]) > 8 and np.sum(groundtruths[i]) > 8: 
            tp += 1
        elif np.sum(predictions[i]) > 8 and np.sum(groundtruths[i]) <= 8: 
            fp += 1
            print("False positive at index: ",i)
        elif np.sum(predictions[i]) <= 8 and np.sum(groundtruths[i]) <= 8: 
            tn += 1
        elif np.sum(predictions[i]) <= 8 and np.sum(groundtruths[i]) > 8: 
            fn += 1
            print("False negative at index: ",i)
    
    print("TP: {}\nFP: {}\nTN: {}\nFN: {}".format(tp, fp, tn, fn))

    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    acc = (tp + tn) / (tp + tn + fp + fp)

    print("Sensitivity is: {}".format(round(sens, 3)))
    print("Specificity is: {}".format(round(spec, 3)))
    print("Accuracy is: {}".format(round(acc, 3)))
