import torch
import torch.nn as nn
import pdb


class CenterNetLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        """
        CenterNet loss module combining focal loss and offset loss.
        Args:
            alpha: Weight for heatmap loss.
            beta: Weight for offset loss.
        """
        super(CenterNetLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def focal_loss(self, pred, gt, alpha=2, beta=4):
        """
        Focal loss for heatmap regression.
        Args:
            pred: Predicted heatmap (B, C, H, W).
            gt: Ground truth heatmap (B, C, H, W).
            alpha: Exponent for (1 - pred).
            beta: Exponent for (1 - gt).
        Returns:
            Loss value.
        """
        pos_mask = gt == 1
        neg_mask = gt < 1

        # print('inside of focal loss')
        # pdb.set_trace()

        pos_loss = -(1 - pred[pos_mask])**alpha * torch.log(pred[pos_mask] + 1e-10)
        neg_loss = -(1 - gt[neg_mask])**beta * (pred[neg_mask]**alpha) * torch.log(1 - pred[neg_mask] + 1e-10)

        return (pos_loss.sum() + neg_loss.sum()) / pos_mask.sum()

    def offset_loss(self, pred, gt, mask):
        """
        Offset regression loss (L1 loss).
        Args:
            pred: Predicted offsets (B, 2, H, W).
            gt: Ground truth offsets (B, 2, H, W).
            mask: Binary mask for locations with keypoints (B, 1, H, W).
        Returns:
            Loss value.
        """
        # pdb.set_trace()

        # Compute the L1 loss masked by valid keypoints
        loss = torch.abs(pred * mask - gt * mask).sum() / mask.sum()
        return loss
            
    def forward(self, hm_pred, offset_pred, heatmap_gt, offset_gt, mask):
        # Debugging:
        # print(f'max hm_pred: {hm_pred.max()}, min hm_pred: {hm_pred.min()}')
        # print(f'max offset_gt: {offset_gt.max()}, min offset_gt: {offset_gt.min()}')
        # print(f'mask stats: max {mask.max()}, min {mask.min()}')

        # Compute heatmap loss using focal loss
        heatmap_loss = self.focal_loss(hm_pred, heatmap_gt)
        # print(f'hml: {heatmap_loss}')
        # Compute offset loss using the mask
        offset_loss_val = self.offset_loss(offset_pred, offset_gt, mask)
        # print(f'ol: {offset_loss_val}')
        # Weighted total loss
        total_loss = self.alpha * heatmap_loss + self.beta * offset_loss_val

        return total_loss


if __name__ == "__main__":
    # Dummy input data
    batch_size = 2
    num_classes = 1
    grid_size = (36, 40)
    grid_H, grid_W = grid_size

    heatmap_pred = torch.rand(batch_size, num_classes, grid_H, grid_W, requires_grad=True)
    offsets_pred = torch.rand(batch_size, 2, grid_H, grid_W, requires_grad=True)

    # Keypoints for each batch
    keypoints = [
        [(5.3, 10.7), (15.4, 8.9)],  # Keypoints for batch 1
        [(7.8, 14.6)]                # Keypoints for batch 2
    ]

    loss_fn = CenterNetLoss(alpha=1.0, beta=1.0, input_size=(1080, 1920), grid_size=(grid_H, grid_W))
    total_loss = loss_fn(heatmap_pred, offsets_pred, keypoints)

    print("Total Loss:", total_loss.item())