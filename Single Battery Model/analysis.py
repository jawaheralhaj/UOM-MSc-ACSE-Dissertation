import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
import torch


def cap_distance(outputs, targets):
    total_distance = 0
    k = 0
    for output, target in zip(outputs, targets):
        distance = abs(target - output)
        total_distance += distance
        tol = distance / abs(target)
        if tol.item() <= 0.025:
            k += 1
    return total_distance / len(outputs), k

def error_calc(outputs, targets):
    d, _ = cap_distance(outputs, targets)
    target_std = torch.std(targets)
    if target_std <= 0:
        print("error in calculating target_std")
        return float('inf')
    error = 100.0 * (d / target_std)
    return error.item()

def mean_absolute_error(output, target):
    return torch.sum(torch.abs(output - target)), output.numel()

def mean_squared_error(output, target):
    return torch.sum((output - target) ** 2), output.numel()

def root_mean_squared_error(output, target):
    mse_sum, n = mean_squared_error(output, target)
    return torch.sqrt(mse_sum / n), n

# def r_squared(output, target):
#     ss_res = torch.sum((target - output) ** 2)
#     ss_tot = torch.sum((target - torch.mean(target, dim=0)) ** 2)
#     return 1 - ss_res / ss_tot, output.numel()

def r_squared(output, target):
    ss_res = torch.sum((target - output) ** 2)
    target_mean = torch.mean(target, dim=0)
    ss_tot = torch.sum((target - target_mean) ** 2)
    
    # Handle the case where ss_tot is zero
    if ss_tot == 0:
        return float('-inf'), output.numel()
    
    return 1 - ss_res / ss_tot, output.numel()


def mean_squared_logarithmic_error(output, target):
    return torch.sum((torch.log1p(output) - torch.log1p(target)) ** 2), output.numel()

# def calculate_snr(signal, noise):
#     signal_power = np.mean(signal ** 2)
#     noise_power = np.mean(noise ** 2)
#     return 10 * np.log10(signal_power / noise_power)

def calculate_snr(signal, noise):
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    
    # Handle the case where noise_power is zero to avoid division by zero
    if noise_power == 0:
        return float('inf')
    
    # Handle the case where signal_power is zero to avoid log of zero
    if signal_power == 0:
        return float('-inf')
    
    return 10 * np.log10(signal_power / noise_power)


def Metrics(metrics_accum, output, target):
    mae_sum, mae_count = mean_absolute_error(output, target)
    mse_sum, mse_count = mean_squared_error(output, target)
    rmse_sum, rmse_count = root_mean_squared_error(output, target)
    r2_sum, r2_count = r_squared(output, target)
    
    metrics_accum['MAE'] += mae_sum
    metrics_accum['MSE'] += mse_sum
    metrics_accum['RMSE'] += rmse_sum * rmse_count  # RMSE should be calculated at the end
    metrics_accum['R-squared'] += r2_sum * r2_count
    metrics_accum['count'] += mae_count

def epoch_metrics(metrics_accum):
    count = metrics_accum['count']
    mae = metrics_accum['MAE'] / count
    mse = metrics_accum['MSE'] / count
    rmse = torch.sqrt(metrics_accum['RMSE'] / count)
    r2 = metrics_accum['R-squared'] / count

    print(f"##Metrics##:\n"
          f"MAE : {mae}\n"
          f"MSE : {mse}\n"
          f"RMSE : {rmse}\n"
          f"R-squared : {r2}\n"
          f"Count: {count}")

    return {
        'MAE': mae.item(),
        'MSE': mse.item(),
        'RMSE': rmse.item(),
        'R-squared': r2.item(),
    }
