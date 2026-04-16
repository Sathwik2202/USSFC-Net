import numpy as np


def ConfusionMatrix(num_classes, pres, gts):
    def __get_hist(pre, gt):
        pre = pre.cpu().detach().numpy()
        gt = gt.cpu().detach().numpy()
        pre[pre >= 0.5], pre[pre < 0.5] = 1, 0
        gt[gt >= 0.5], gt[gt < 0.5] = 1, 0
        mask = (gt >= 0) & (gt < num_classes)
        label = num_classes * gt[mask].astype(int) + pre[mask].astype(int)
        return np.bincount(label, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    
    cm = np.zeros((num_classes, num_classes))
    for lt, lp in zip(gts, pres):
        cm += __get_hist(lt.flatten(), lp.flatten())
    return cm

def get_score(confusionMatrix):
    eps = np.finfo(np.float32).eps
    precision = np.diag(confusionMatrix) / (confusionMatrix.sum(axis=0) + eps)
    recall = np.diag(confusionMatrix) / (confusionMatrix.sum(axis=1) + eps)
    f1score = 2 * precision * recall / ((precision + recall) + eps)
    iou = np.diag(confusionMatrix) / (confusionMatrix.sum(axis=1) + confusionMatrix.sum(axis=0) - np.diag(confusionMatrix) + eps)
    po = np.diag(confusionMatrix).sum() / (confusionMatrix.sum() + eps)
    pe = (confusionMatrix[0].sum() * confusionMatrix[:,0].sum() + confusionMatrix[1].sum() * confusionMatrix[:,1].sum()) / confusionMatrix.sum() ** 2 + eps
    kc = (po - pe) / (1 - pe + eps)
    return precision, recall, f1score, iou, kc

def get_score_sum(confusionMatrix):
    p, r, f, i, k = get_score(confusionMatrix)
    return {'p':p, 'r':r, 'f1':f, 'iou':i, 'kc':k}
