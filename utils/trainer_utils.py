

def update_learning_rate(optimizer, decay_rate=0.999, lowest=1e-5):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        lr = max(lr * decay_rate, lowest)
        param_group['lr'] = lr