from sklearn.metrics import roc_auc_score,f1_score, precision_score, recall_score, classification_report
import numpy as np
from scipy.sparse import csr_matrix
from functools import partial
from sklearn.preprocessing import MultiLabelBinarizer

def get_precision(y_true,y_pred,classes,top=5):
    mlb = MultiLabelBinarizer(classes=classes,sparse_output=True)
    mlb.fit(y_true)
    if not isinstance(y_true, csr_matrix):
        y_true = mlb.transform(y_true)
    y_pred = mlb.transform(y_pred[:,:top])
    return y_pred.multiply(y_true).sum() / (top * y_true.shape[0])

get_p_1 = partial(get_precision, top=1)
get_p_3 = partial(get_precision, top=3)
get_p_5 = partial(get_precision, top=5)

def get_ndcg(y_true, y_pred, classes,top=5):
    mlb = MultiLabelBinarizer(classes=classes,sparse_output=True)
    mlb.fit(y_true)
    if not isinstance(y_true, csr_matrix):
        y_true = mlb.transform(y_true)
    log = 1.0 / np.log2(np.arange(top) + 2)
    dcg = np.zeros((y_true.shape[0], 1))
    for i in range(top):
        p = mlb.transform(y_pred[:, i: i + 1])
        dcg += p.multiply(y_true).sum(axis=-1) * log[i]
    return np.average(dcg / log.cumsum()[np.minimum(y_true.sum(axis=-1), top) - 1])

get_n_1 = partial(get_ndcg, top=1)
get_n_3 = partial(get_ndcg, top=3)
get_n_5 = partial(get_ndcg, top=5)

def label_wise_prf1(y_true, y_pred):
    p = precision_score(y_true=y_true, y_pred=y_pred, average=None)
    r = recall_score(y_true=y_true, y_pred=y_pred, average=None)
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average=None)
    return p, r, f1

def base1_metric(y_true, y_pred,classes):
    p1, p3, p5 = get_p_1(y_true,y_pred, classes), get_p_3(y_true, y_pred, classes), get_p_5(y_true, y_pred, classes)
    n1, n3, n5 = get_n_1(y_true, y_pred, classes), get_n_3(y_true, y_pred, classes), get_n_5(y_true,y_pred, classes)
    return p1,p3,p5,n3,n5

def MicroF1(predict_labels, test_target):
    return f1_score(test_target,predict_labels,average='micro')

def MacroF1(predict_labels, test_target):
    f1 = f1_score(test_target,predict_labels,average='macro')
    return f1

def MicroPrecision(predict_labels, test_target):
    return precision_score(test_target,predict_labels,average='micro')

def MacroPrecision(predict_labels, test_target):
    return precision_score(test_target,predict_labels,average='macro')

def MicroRecall(predict_labels, test_target):
    return recall_score(test_target,predict_labels,average='micro')

def MacroRecall(predict_labels, test_target):
    return recall_score(test_target,predict_labels,average='macro')

def base2_metric(y_true, y_pred):
    ma_p, ma_r, ma_f1 = MacroPrecision(y_pred, y_true), MacroRecall(y_pred, y_true), MacroF1(y_pred, y_true)
    mi_p, mi_r, mi_f1 = MicroPrecision(y_pred, y_true), MicroRecall(y_pred, y_true), MicroF1(y_pred, y_true)
    return ma_p, ma_r, ma_f1, mi_p, mi_r, mi_f1