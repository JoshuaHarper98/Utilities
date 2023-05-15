from sklearn.metrics import roc_auc_score

def minimise_auroc(y_true, y_pred):
    metric_name = "minimise_auroc"
    value = 1 - roc_auc_score(y_true, y_pred)
    is_higher_better = False
    return metric_name, value, is_higher_better
