def dice_coefficient(y_true, y_pred, smooth=1e-6):
    intersection = (y_true * y_pred).sum()
    return (2. * intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth)

