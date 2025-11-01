import torch
import torch.nn as nn
import torch.nn.functional as F

def contrastive_loss(z_img, z_txt, temperature: float = 0.07):
    """
    Symmetric InfoNCE-style contrastive loss for alignment.
    z_img, z_txt: [B, D] normalized embeddings.
    """
    z_img = F.normalize(z_img, dim=-1)
    z_txt = F.normalize(z_txt, dim=-1)
    logits = z_img @ z_txt.t() / temperature
    labels = torch.arange(z_img.size(0), device=z_img.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_i2t + loss_t2i)

def cross_entropy_caption_loss(logits, targets, ignore_index=-100):
    """
    Standard cross-entropy caption loss over token logits.
    logits: [B, T, V], targets: [B, T]
    """
    return F.cross_entropy(logits.transpose(1, 2), targets, ignore_index=ignore_index)

def vqa_accuracy(preds, labels):
    """
    Simple exact-match accuracy for VQA-style single-label answers.
    preds: [B, C] (logits), labels: [B] (int)
    """
    with torch.no_grad():
        acc = (preds.argmax(dim=-1) == labels).float().mean()
    return acc.item()
