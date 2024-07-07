from typing import Callable


adjusment_func = Callable[[float, float, float, int, float], float]


def adaptive_lr(current_lr: float, current_error: float, previous_error: float, epoch: int, min_lr: float = 0.00001) -> float:
    if epoch == 0 or previous_error == 0:
        return current_lr
    
    error_ratio = current_error / previous_error
    
    if error_ratio > 1:
        increase_percentage = (error_ratio - 1) * 100
        decay_rate = min(0.5, increase_percentage / 100)
        new_lr = max(current_lr * (1 - decay_rate), min_lr)
    elif error_ratio < 0.99:
        decrease_percentage = (1 - error_ratio) * 100
        increase_rate = min(0.1, decrease_percentage / 200)
        new_lr = min(current_lr * (1 + increase_rate), current_lr * 1.5)
    else:
        new_lr = current_lr
    
    return max(new_lr, min_lr)
    

def no_lr_adjustment(current_lr: float, current_error: float, 
                        previous_error: float, epoch: int, 
                        min_lr: float = 1e-6) -> float:
    return current_lr 


def adaptive_percentage(current_lr: float, current_error: float, previous_error: float, epoch: int, min_lr: float = 0.00001) -> float:
    if epoch == 0 or previous_error == 0:
        return current_lr
    error_ratio = current_error / previous_error
    if error_ratio > 1:
        increase_percentage = (error_ratio - 1) * 100
        decay_rate = min(0.5, increase_percentage / 100)
        new_lr = max(current_lr * (1 - decay_rate), min_lr)
    elif error_ratio < 0.99:
        decrease_percentage = (1 - error_ratio) * 100
        increase_rate = min(0.1, decrease_percentage / 200)
        new_lr = min(current_lr * (1 + increase_rate), current_lr * 1.5)
    else:
        new_lr = current_lr
    
    return max(new_lr, min_lr)