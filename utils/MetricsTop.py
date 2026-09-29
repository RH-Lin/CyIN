import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score

import warnings
warnings.filterwarnings("ignore")

__all__ = ['MetricsTop']

class MetricsTop():
    def __init__(self, train_mode):
        if train_mode == 'regression':
            self.metrics_dict = {
                'MOSI': self.__eval_mosi_regression,
                'MOSEI': self.__eval_mosei_regression,
            }
        elif train_mode == 'recognition':
            self.metrics_dict = {
                'IEMOCAP6': self.__eval_iemocap6_recognition,
                'MELD': self.__eval_meld_recognition,
            }

    def __eval_mosi_regression(self, y_pred, y_true, masks=None, exclude_zero=False):
        if masks is not None:
            # 获取 mask 为 1 的索引
            indices = np.nonzero(masks)[0]

            # 根据索引提取预测值和真实标签
            y_pred = y_pred[indices]
            y_true = y_true[indices]

        y_pred = y_pred.view(-1).cpu().detach().numpy()
        y_true = y_true.view(-1).cpu().detach().numpy()

        y_pred_7 = np.clip(y_pred, a_min=-3., a_max=3.)
        y_true_7 = np.clip(y_true, a_min=-3., a_max=3.)
        y_pred_5 = np.clip(y_pred, a_min=-2., a_max=2.)
        y_true_5 = np.clip(y_true, a_min=-2., a_max=2.)
        y_pred_3 = np.clip(y_pred, a_min=-1., a_max=1.)
        y_true_3 = np.clip(y_true, a_min=-1., a_max=1.)
        Mult_acc_7 = self.__multiclass_acc(y_pred_7, y_true_7)
        Mult_acc_5 = self.__multiclass_acc(y_pred_5, y_true_5)
        Mult_acc_3 = self.__multiclass_acc(y_pred_3, y_true_3)

        # with 0 (<=0 or >0)
        Has0_y_pred_2 = (y_pred >= 0) # binary True or False
        Has0_y_true_2 = (y_true >= 0)
        Has0_acc_2 = accuracy_score(Has0_y_true_2, Has0_y_pred_2) #if masks == None else accuracy_score(Has0_y_true_2, Has0_y_pred_2, sample_weight=masks)
        Has0_F1_score = f1_score(Has0_y_true_2, Has0_y_pred_2, average='weighted') #if masks == None else f1_score(Has0_y_true_2, Has0_y_pred_2, sample_weight=masks, average='weighted')

        # without 0 (<0 or >0)
        non_zeros = np.array([i for i, e in enumerate(y_true) if e != 0])
        Non0_y_pred_2 = (y_pred[non_zeros] > 0)
        Non0_y_true_2 = (y_true[non_zeros] > 0)
        Non0_acc_2 = accuracy_score(Non0_y_true_2, Non0_y_pred_2) #if masks == None else accuracy_score(Non0_y_true_2, Non0_y_pred_2, sample_weight=masks)
        Non0_F1_score = f1_score(Non0_y_true_2, Non0_y_pred_2, average='weighted') #if masks == None else f1_score(Non0_y_true_2, Non0_y_pred_2, sample_weight=masks, average='weighted')

        mae = np.mean(np.absolute(y_pred - y_true)).astype(np.float64)  # Average L1 distance between y_pred and y_true
        corr = np.corrcoef(y_pred, y_true)[0][1]  # Correlation Coefficient: get correlation matrix[0][1]

        eval_results = {
            "Has0_acc_2": round(Has0_acc_2, 4),
            "Has0_F1_score": round(Has0_F1_score, 4),
            "Non0_acc_2": round(Non0_acc_2, 4),
            "Non0_F1_score": round(Non0_F1_score, 4),
            "Mult_acc_3": round(Mult_acc_3, 4),
            "Mult_acc_5": round(Mult_acc_5, 4),
            "Mult_acc_7": round(Mult_acc_7, 4),
            "MAE": round(mae, 4),
            "Corr": round(corr, 4)
        }
        return eval_results

    def __eval_mosei_regression(self, y_pred, y_true, masks=None):
        return self.__eval_mosi_regression(y_pred, y_true, masks)

    def __multiclass_acc(self, y_pred, y_true):
        """
        Compute the multiclass accuracy with respect to groundtruth
        y_pred: Float array representing the predictions, dimension (N,)
        y_true: Float/int array representing the groundtruth classes, dimension (N,)
        return: Classification accuracy
        """
        return np.sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true))

    def negative_weighted_acc(self, preds, truths):
        preds = preds.view(-1)
        truths = truths.view(-1)

        total = len(preds)
        tp = 0
        tn = 0
        p = 0
        n = 0
        for i in range(total):
            if truths[i] == 0:
                n += 1
                if preds[i] == 0:
                    tn += 1
            elif truths[i] == 1:
                p += 1
                if preds[i] == 1:
                    tp += 1

        w_acc = (tp * n / p + tn) / (2 * n)

        # if verbose:
        #     fp = n - tn
        #     fn = p - tp
        #     recall = tp / (tp + fn + 1e-8)
        #     precision = tp / (tp + fp + 1e-8)
        #     f1 = 2 * recall * precision / (recall + precision + 1e-8)
        #     print('TP=', tp, 'TN=', tn, 'FP=', fp, 'FN=', fn, 'P=', p, 'N', n, 'Recall', recall, "f1", f1)

        return w_acc

    def __eval_single_recognition(self, y_pred, y_true, masks, emo_class, eval_mode='conversation'):
        """
        Emotion Recognition is a single-label classification task.
        y_pred: (batch_size, num_emotions)
        y_true: (batch_size, num_emotions)
        """
        y_pred = y_pred.cpu().detach()
        y_true = y_true.cpu().detach()
        emo_num = len(emo_class)
        if eval_mode == 'utterance': # https://github.com/wenliangdai/Modality-Transferable-MER
            y_pred = torch.sigmoid(y_pred)
            # aucs = roc_auc_score(y_true, y_pred, labels=range(len(emo_class)), average=None).tolist()
            # aucs.append(np.average(aucs))
            # zsl: 0.5 0.35 0.3
            th = [0.5] * emo_num
            for i in range(len(th)):
                pred = y_pred[:, i]
                pred[pred > th[i]] = 1.
                pred[pred <= th[i]] = 0.
                y_pred[:, i] = pred
                true = y_true[:, i]
                true[true > th[i]] = 1.
                true[true <= th[i]] = 0.
                y_true[:, i] = true

            eval_results = {}
            accs = []
            f1s = []
            for i in range(emo_num):
                pred_i = y_pred[:, i]
                truth_i = y_true[:, i]
                # acc = self.negative_weighted_acc(pred_i, truth_i, verbose=True)
                acc = accuracy_score(truth_i, pred_i)
                f1 = f1_score(truth_i, pred_i, average='binary')
                # f1 = f1_score(truth_i, pred_i, average='weighted')
                accs.append(acc)
                f1s.append(f1)

                eval_results[emo_class[i]] = round(acc, 4)  #f1
            eval_results['Accuracy'] = round(sum(accs)/len(accs), 4)
            eval_results['BinaryF1'] = round(sum(f1s)/len(f1s), 4)
        elif eval_mode == 'conversation': # https://github.com/leson502/CORECT_EMNLP2023
            y_pred = torch.argmax(y_pred, dim=-1)
            y_true = torch.argmax(y_true, dim=-1)

            # acc = accuracy_score(y_true, y_pred, sample_weight=masks) # GCNet
            # f1 = f1_score(y_true, y_pred, sample_weight=masks, average='weighted')

            report_results = classification_report(
                y_true, y_pred, labels=range(emo_num), sample_weight=masks, target_names=emo_class, output_dict=True
            ) # GCNet
            eval_results = {}
            for emo in emo_class:
                eval_results[emo] = round(report_results[emo]['f1-score'], 4)
            eval_results['Accuracy'] = round(report_results['accuracy'], 4)
            eval_results['WeightedF1'] = round(report_results['weighted avg']['f1-score'], 4)
        else:
            assert 0, f'Error eval_mode: {eval_mode}'

        return eval_results

    def __eval_iemocap6_recognition(self, y_pred, y_true, masks=None):
        """
        IEMOCAP6 Emotion Recognition is a multi-label classification task.
        Including ['happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated']
        """
        emo_class = ['Happy', 'Sad', 'Neutral', 'Angry', 'Excited', 'Frustrated']
        return self.__eval_single_recognition(y_pred, y_true, masks, emo_class)

    def __eval_meld_recognition(self, y_pred, y_true, masks=None):
        """
        MELD Emotion Recognition is a single-label multi-classification task.
        Including ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']
        """
        emo_class = ['Neutral', 'Surprise', 'Fear', 'Sadness', 'Joy', 'Disgust', 'Anger']
        return self.__eval_single_recognition(y_pred, y_true, masks, emo_class)
    
    def getMetrics(self, datasetName):
        return self.metrics_dict[datasetName.upper()] # upper() converts lowercase letters to uppercase letters
