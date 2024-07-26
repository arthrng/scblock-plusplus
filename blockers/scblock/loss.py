import torch
from torch import nn

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss as described in https://arxiv.org/pdf/2004.11362.pdf.
    """
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07, is_auxiliary=False, flood_type=None):
        """
        Initialize the SupConLoss.

        Args:
            temperature (float): Temperature scaling parameter.
            contrast_mode (str): Mode for contrast, either 'one' or 'all'.
            base_temperature (float): Base temperature for scaling.
            is_auxiliary (bool): Flag for auxiliary mode.
            flood_type (str): Type of flooding to apply (e.g., 'ada', 'iflood', 'regular').
        """
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.is_auxiliary = is_auxiliary
        self.flood_type = flood_type

    def forward(self, features, labels=None, mask=None, flood_levels=None):
        """
        Compute the Supervised Contrastive Loss.

        Args:
            features (torch.Tensor): Hidden vectors of shape [bsz, n_views, ...].
            labels (torch.Tensor): Ground truth labels of shape [bsz].
            mask (torch.Tensor): Contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j has the same class as sample i.
            flood_levels (torch.Tensor): Flooding levels for loss adjustment.

        Returns:
            torch.Tensor: Computed loss.
        """
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...], at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Number of labels does not match number of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # Compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        
        # Mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        mask = mask * logits_mask

        # Compute log probabilities
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Compute mean of log-likelihood over positive samples
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # Compute the loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        if self.is_auxiliary:
            return loss.view(anchor_count, batch_size)[0]
        else:
            if self.flood_type == 'ada' or self.flood_type == 'iflood':
                return (abs(loss - flood_levels) + flood_levels).mean()
            elif self.flood_type == 'regular':
                loss_corrected = (loss.view(anchor_count, batch_size).mean() - flood_levels).abs() + flood_levels
                return loss_corrected
            else:
                return loss.view(anchor_count, batch_size).mean()
