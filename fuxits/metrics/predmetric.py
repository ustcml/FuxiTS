import torchmetrics.functional as M
import torch.nn.functional as F
import torch
import fuxits.losses.losses as losses

def masked_mean_absolute_error(pred, target, mask_val=float('nan')):
    mask = losses.ismask(target, mask_val).float()
    error = torch.abs(pred - target)
    error = error * mask / mask.mean()
    torch.nan_to_num_(error)
    return torch.mean(error)

def masked_mean_absolute_percentage_error(pred, target, mask_val=float('nan')):
    mask = losses.ismask(target, mask_val).float()
    error = torch.abs(pred - target) / torch.clamp(torch.abs(target), min=1e-10)
    error = error * mask / mask.mean()
    torch.nan_to_num_(error)
    return torch.mean(error)
    


def masked_mean_squared_error(pred, target, mask_val=float('nan'), squared=True):
    mask =  losses.ismask(target, mask_val).float()
    error = (pred - target) ** 2
    error = error * mask / mask.mean()
    torch.nan_to_num_(error)
    if squared:
        return torch.mean(error)
    else:
        return torch.sqrt(torch.mean(error))

def r2_score(pred, target):
    numerator = torch.sum((pred - target) ** 2, dim=0)
    denominator = torch.sum((target - torch.mean(target, dim=0)) ** 2, dim=0)
    output_scores = 1 - numerator / denominator
    return torch.nan_to_num(output_scores).mean()

def explained_variance_score(pred, target):
    diff = target - pred
    numerator = torch.mean((diff - torch.mean(diff, dim=0)) ** 2, dim=0)
    denominator = torch.mean((target - torch.mean(target, dim=0)) ** 2, dim=0)
    outpu_scors = 1 - numerator / denominator
    return torch.nan_to_num(outpu_scors).mean()

metric_dict = {
    'mae': M.mean_absolute_error,
    'mape': M.mean_absolute_percentage_error,
    'mse': M.mean_squared_error,
    'rmse': lambda p, t: M.mean_squared_error(p, t, False),
    'masked_mae': masked_mean_absolute_error,
    'masked_mse': masked_mean_squared_error,
    'masked_mape': masked_mean_absolute_percentage_error,
    'masked_rmse': lambda p, t: masked_mean_squared_error(p, t, False),
    'r2': r2_score,
    'evar': explained_variance_score
}

def get_pred_metrics(metric):
    if not isinstance(metric, list):
        metric = [metric]
    pred_m = [(m, metric_dict[m]) for m in metric if m in metric_dict]
    return pred_m


def test():
    import sklearn.metrics as skm
    y_true = torch.tensor([3, -0.5, 2, 7])
    y_pred = torch.tensor([2.5, 0.0, 2, 8])
    print(explained_variance_score(y_pred, y_true))
    print(skm.explained_variance_score(y_true, y_pred))
    print(r2_score(y_pred, y_true))
    print(skm.r2_score(y_true, y_pred))
    y_true = torch.tensor([[0.5, 1], [-1, 1], [7, -6]])
    y_pred = torch.tensor([[0.0, 2], [-1, 2], [8, -5]])
    print(explained_variance_score(y_pred, y_true))
    print(skm.explained_variance_score(y_true, y_pred))
    print(r2_score(y_pred, y_true))
    print(skm.r2_score(y_true, y_pred))
