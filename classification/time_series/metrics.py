from sklearn.metrics import\
        accuracy_score,\
        classification_report,\
        cohen_kappa_score,\
        f1_score,\
        recall_score


def compute_performance_metrics(y_true, y_pred, verbose=True, classnames=None):
    performance_metrics = dict()

    # OA
    OA = round(accuracy_score(y_true, y_pred), 3)
    performance_metrics['OA'] = OA

    # AA
    AA = round(recall_score(y_true, y_pred, average='macro'), 3)
    performance_metrics['AA'] = AA

    # weighted f-score
    f1_weighted = round(f1_score(y_true, y_pred, average='weighted'), 3)
    performance_metrics['f1_weighted'] = f1_weighted

    # kappa-metric
    kappa = round(cohen_kappa_score(y_true, y_pred), 3)
    performance_metrics['kappa'] = kappa

    # print classification report
    if verbose:
        print(classification_report(y_true, y_pred,
                                    target_names=classnames))

    return performance_metrics
