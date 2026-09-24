"""Numerical-stability helpers for audited AWQ calibration."""

from __future__ import annotations


def float32_mse_sum(torch_module, reference, candidate):
    """Compute a reduction sum after casting both operands to FP32."""

    return torch_module.nn.functional.mse_loss(
        reference.float(),
        candidate.to(reference.device).float(),
        reduction="sum",
    )


def install_awq_float32_loss_patch(awq_modifier_class) -> None:
    """Install a process-local repair for FP16 overflow in AWQ loss sums."""

    from llmcompressor.modifiers.transform.awq import base

    torch = base.torch

    @torch.no_grad()
    def _compute_loss(self, fp16_outputs, int_w_outputs) -> float:
        session = base.active_session()
        loss_masks = session.state.loss_masks if session.state else None

        device = fp16_outputs[0].device
        loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        num_elements = torch.tensor(0, device=device, dtype=torch.long)

        for batch_idx, (fp16_batch, int_w_batch) in enumerate(
            zip(fp16_outputs, int_w_outputs)
        ):
            loss_mask = loss_masks[batch_idx] if loss_masks else None
            if loss_mask is not None:
                token_mask = loss_mask.to(fp16_batch.device) == 1
                fp16_batch = fp16_batch[token_mask]
                int_w_batch = int_w_batch.to(fp16_batch.device)[token_mask]
            loss += float32_mse_sum(torch, fp16_batch, int_w_batch)
            num_elements += fp16_batch.numel()

        if base.is_distributed():
            loss, num_elements = base._allreduce_data_sum(
                [loss, num_elements]
            )
        return (loss / num_elements).item()

    awq_modifier_class._compute_loss = _compute_loss
