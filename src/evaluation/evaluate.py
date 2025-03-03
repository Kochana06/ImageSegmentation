import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from evaluation.dice import dice_coefficient
from evaluation.IoU import iou_score

# Function for Model Evaluation
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    dice_scores = []
    iou_scores = []
    accuracies = []
    with torch.no_grad():
        for images, masks, labels in test_loader:
            images, masks, labels = images.to(device), masks.to(device), labels.to(device)
            pred_masks, pred_labels = model(images)
            preds = torch.argmax(pred_labels, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            dice_scores.append(dice_coefficient(masks.cpu(), pred_masks.cpu()))
            iou_scores.append(iou_score(masks.cpu(), pred_masks.cpu()))
            accuracies.append(accuracy_score(labels.cpu().numpy(), preds))
    
    accuracy = np.mean(accuracies)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)
    print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
    print(f"Dice Score: {avg_dice:.4f}, IoU Score: {avg_iou:.4f}")
    print("Confusion Matrix:\n", cm)
    return accuracy, f1, avg_dice, avg_iou, accuracies