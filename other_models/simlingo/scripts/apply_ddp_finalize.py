"""Finalize DDP support in the NVIDIA trainer + Modal entrypoint.

Run once from the repo root:

    python simlingo/scripts/apply_ddp_finalize.py

It applies the remaining 5 changes needed to make 4×B200 DDP training correct:

  1. Gate wandb.log / print / save_checkpoint to rank 0 (`self.is_main`).
  2. Use self._raw_model.state_dict() / load_state_dict() to avoid the
     `module.` prefix that DDP would otherwise inject into saved LoRA weights.
  3. Call train_sampler.set_epoch(epoch) at the start of each epoch so the
     shuffle order rotates per epoch.
  4. all_reduce val metrics across ranks so rank 0 logs the global mean
     (and best-checkpoint selection uses the true validation loss).
  5. Switch the Modal multi-GPU entrypoint from H100:2 to B200:4 (and bump
     CPU/memory accordingly).

The script is idempotent: re-running it on an already-patched file is a no-op.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINER = REPO_ROOT / "simlingo" / "scripts" / "nvidia_trainer.py"
MODAL = REPO_ROOT / "simlingo" / "modal_training.py"


def patch(text: str, old: str, new: str, label: str) -> str:
    """Replace `old` with `new`, or leave text unchanged if already patched."""
    if new in text:
        print(f"  [skip] {label}: already patched")
        return text
    if old not in text:
        raise SystemExit(
            f"  [FAIL] {label}: could not locate anchor text. Manual edit needed."
        )
    print(f"  [ok]   {label}")
    return text.replace(old, new, 1)


# ---------------------------------------------------------------------------
# nvidia_trainer.py
# ---------------------------------------------------------------------------
print(f"Patching {TRAINER}...")
src = TRAINER.read_text()

# (A) validate(): gate the wandb panel log to rank 0
src = patch(
    src,
    old=(
        "        # Log viz to W&B\n"
        "        if viz_panel is not None:\n"
        "            try:\n"
        "                import wandb\n"
        "                from scripts.viz_training import panel_to_wandb_image\n"
        "                wb_img = panel_to_wandb_image(viz_panel, caption=viz_caption)\n"
        "                if wb_img is not None:\n"
        "                    wandb.log(\n"
        '                        {"val/prediction_panel": wb_img},\n'
        "                        step=self.global_step,\n"
        "                    )\n"
        "            except Exception as e:\n"
        '                print(f"[viz] Failed to log panel to wandb: {e}")\n'
    ),
    new=(
        "        # all_reduce val metrics across ranks so every rank sees the global mean\n"
        "        if getattr(self, 'is_distributed', False):\n"
        "            import torch.distributed as dist\n"
        "            t = torch.tensor([avg_total, avg_wp, avg_route], device=self.device)\n"
        "            dist.all_reduce(t, op=dist.ReduceOp.SUM)\n"
        "            t /= self.world_size\n"
        "            avg_total, avg_wp, avg_route = t.tolist()\n"
        "\n"
        "        # Log viz to W&B (rank 0 only)\n"
        "        if viz_panel is not None and getattr(self, 'is_main', True):\n"
        "            try:\n"
        "                import wandb\n"
        "                from scripts.viz_training import panel_to_wandb_image\n"
        "                wb_img = panel_to_wandb_image(viz_panel, caption=viz_caption)\n"
        "                if wb_img is not None:\n"
        "                    wandb.log(\n"
        '                        {"val/prediction_panel": wb_img},\n'
        "                        step=self.global_step,\n"
        "                    )\n"
        "            except Exception as e:\n"
        '                print(f"[viz] Failed to log panel to wandb: {e}")\n'
    ),
    label="validate(): all_reduce + rank-0 panel log",
)

# (B) Epoch loop: set_epoch on the DistributedSampler so shuffle order rotates
src = patch(
    src,
    old=(
        "        for epoch in range(epochs):\n"
        "            self.current_epoch = epoch\n"
        "            self.model.train()\n"
        "\n"
        "            epoch_losses = []\n"
        '            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{epochs}")\n'
    ),
    new=(
        "        for epoch in range(epochs):\n"
        "            self.current_epoch = epoch\n"
        "            self.model.train()\n"
        "\n"
        "            # Rotate DistributedSampler shuffle per epoch\n"
        "            if getattr(self, 'is_distributed', False):\n"
        "                from torch.utils.data.distributed import DistributedSampler\n"
        "                if isinstance(self.train_loader.sampler, DistributedSampler):\n"
        "                    self.train_loader.sampler.set_epoch(epoch)\n"
        "\n"
        "            epoch_losses = []\n"
        "            pbar = tqdm(\n"
        "                self.train_loader,\n"
        '                desc=f"Epoch {epoch + 1}/{epochs}",\n'
        "                disable=not getattr(self, 'is_main', True),\n"
        "            )\n"
    ),
    label="epoch loop: set_epoch + rank-0 tqdm",
)

# (C) Step-level wandb.log -> rank 0 only
src = patch(
    src,
    old=(
        "                    # Logging\n"
        "                    if self.global_step % logging_steps == 0:\n"
        "                        wandb.log({\n"
        '                            "train/loss": accum_losses["total_loss"],\n'
    ),
    new=(
        "                    # Logging (rank 0 only)\n"
        "                    if self.global_step % logging_steps == 0 and getattr(self, 'is_main', True):\n"
        "                        wandb.log({\n"
        '                            "train/loss": accum_losses["total_loss"],\n'
    ),
    label="step-level train wandb.log: rank-0 gate",
)

# (D) Step-level eval + best-checkpoint: rank 0 only for log/save, all ranks run validate()
src = patch(
    src,
    old=(
        "                    # Evaluation\n"
        "                    if self.global_step % eval_steps == 0:\n"
        "                        val_metrics = self.validate()\n"
        "                        wandb.log({\n"
        '                            "val/loss": val_metrics["val_loss"],\n'
        '                            "val/wp_loss": val_metrics.get("val_wp_loss", 0.0),\n'
        '                            "val/route_loss": val_metrics.get("val_route_loss", 0.0),\n'
        "                        }, step=self.global_step)\n"
        "\n"
        "                        # Save best model\n"
        '                        if val_metrics["val_loss"] < self.best_val_loss:\n'
        '                            self.best_val_loss = val_metrics["val_loss"]\n'
        '                            self.save_checkpoint("best")\n'
        "\n"
        "                    # Save checkpoint\n"
        "                    if self.global_step % save_steps == 0:\n"
        '                        self.save_checkpoint(f"step_{self.global_step}")\n'
    ),
    new=(
        "                    # Evaluation (all ranks participate; only rank 0 logs/saves)\n"
        "                    if self.global_step % eval_steps == 0:\n"
        "                        val_metrics = self.validate()\n"
        "                        if getattr(self, 'is_main', True):\n"
        "                            wandb.log({\n"
        '                                "val/loss": val_metrics["val_loss"],\n'
        '                                "val/wp_loss": val_metrics.get("val_wp_loss", 0.0),\n'
        '                                "val/route_loss": val_metrics.get("val_route_loss", 0.0),\n'
        "                            }, step=self.global_step)\n"
        "\n"
        '                            if val_metrics["val_loss"] < self.best_val_loss:\n'
        '                                self.best_val_loss = val_metrics["val_loss"]\n'
        '                                self.save_checkpoint("best")\n'
        "\n"
        "                    # Save checkpoint (rank 0 only)\n"
        "                    if self.global_step % save_steps == 0 and getattr(self, 'is_main', True):\n"
        '                        self.save_checkpoint(f"step_{self.global_step}")\n'
    ),
    label="step-level eval + save: rank-0 gating",
)

# (E) End-of-epoch logging, save, and prints: rank 0 only
src = patch(
    src,
    old=(
        "            # End of epoch\n"
        "            avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else 0.0\n"
        '            print(f"Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")\n'
        '            all_metrics.append({"epoch": epoch + 1, "loss": avg_epoch_loss})\n'
        "\n"
        "            # End-of-epoch validation + viz (guarantees at least one val/viz per epoch)\n"
        '            print(f"Running end-of-epoch validation...")\n'
        "            val_metrics = self.validate()\n"
        "            wandb.log({\n"
        '                "val/loss": val_metrics["val_loss"],\n'
        '                "val/wp_loss": val_metrics.get("val_wp_loss", 0.0),\n'
        '                "val/route_loss": val_metrics.get("val_route_loss", 0.0),\n'
        '                "epoch": epoch + 1,\n'
        "            }, step=self.global_step)\n"
        '            print(f"Epoch {epoch + 1} val loss: {val_metrics[\'val_loss\']:.4f}")\n'
        "\n"
        "            # Save best checkpoint\n"
        '            if val_metrics["val_loss"] < self.best_val_loss:\n'
        '                self.best_val_loss = val_metrics["val_loss"]\n'
        '                self.save_checkpoint("best")\n'
        "\n"
        "            # Always save end-of-epoch checkpoint\n"
        '            self.save_checkpoint(f"epoch_{epoch + 1}")\n'
    ),
    new=(
        "            # End of epoch\n"
        "            avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else 0.0\n"
        "            if getattr(self, 'is_main', True):\n"
        '                print(f"Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")\n'
        '            all_metrics.append({"epoch": epoch + 1, "loss": avg_epoch_loss})\n'
        "\n"
        "            # End-of-epoch validation + viz (all ranks participate)\n"
        "            if getattr(self, 'is_main', True):\n"
        '                print(f"Running end-of-epoch validation...")\n'
        "            val_metrics = self.validate()\n"
        "            if getattr(self, 'is_main', True):\n"
        "                wandb.log({\n"
        '                    "val/loss": val_metrics["val_loss"],\n'
        '                    "val/wp_loss": val_metrics.get("val_wp_loss", 0.0),\n'
        '                    "val/route_loss": val_metrics.get("val_route_loss", 0.0),\n'
        '                    "epoch": epoch + 1,\n'
        "                }, step=self.global_step)\n"
        '                print(f"Epoch {epoch + 1} val loss: {val_metrics[\'val_loss\']:.4f}")\n'
        "\n"
        "                # Save best checkpoint\n"
        '                if val_metrics["val_loss"] < self.best_val_loss:\n'
        '                    self.best_val_loss = val_metrics["val_loss"]\n'
        '                    self.save_checkpoint("best")\n'
        "\n"
        "                # Always save end-of-epoch checkpoint\n"
        '                self.save_checkpoint(f"epoch_{epoch + 1}")\n'
    ),
    label="end-of-epoch logging + save: rank-0 gating",
)

# (F) save_checkpoint: pull state_dict from _raw_model to drop DDP `module.` prefix
src = patch(
    src,
    old=(
        "        # Save LoRA weights separately\n"
        "        lora_state = {}\n"
        "        for key, value in self.model.state_dict().items():\n"
        '            if "lora_A_" in key or "lora_B_" in key:\n'
        "                lora_state[key] = value\n"
    ),
    new=(
        "        # Save LoRA weights separately (use _raw_model to avoid DDP `module.` prefix)\n"
        "        _src_model = getattr(self, '_raw_model', self.model)\n"
        "        lora_state = {}\n"
        "        for key, value in _src_model.state_dict().items():\n"
        '            if "lora_A_" in key or "lora_B_" in key:\n'
        "                lora_state[key] = value\n"
    ),
    label="save_checkpoint: LoRA via _raw_model",
)

src = patch(
    src,
    old=(
        "        # Save full model state if needed\n"
        "        torch.save(self.model.state_dict(), ckpt_path)\n"
    ),
    new=(
        "        # Save full model state if needed (use _raw_model to avoid DDP `module.` prefix)\n"
        "        torch.save(getattr(self, '_raw_model', self.model).state_dict(), ckpt_path)\n"
    ),
    label="save_checkpoint: full state via _raw_model",
)

# (G) load_checkpoint: load into _raw_model
src = patch(
    src,
    old=(
        "        # Check if it's a LoRA-only checkpoint\n"
        '        if path.name.startswith("lora_") and path.suffix == ".pt":\n'
        "            lora_state = torch.load(path, map_location=self.device)\n"
        "            # Load LoRA weights\n"
        "            current_state = self.model.state_dict()\n"
        "            for key, value in lora_state.items():\n"
        "                if key in current_state:\n"
        "                    current_state[key] = value\n"
        "            self.model.load_state_dict(current_state)\n"
        '        elif path.suffix == ".pt":\n'
        "            state_dict = torch.load(path, map_location=self.device)\n"
        "            self.model.load_state_dict(state_dict, strict=False)\n"
    ),
    new=(
        "        # Check if it's a LoRA-only checkpoint\n"
        "        _dst_model = getattr(self, '_raw_model', self.model)\n"
        '        if path.name.startswith("lora_") and path.suffix == ".pt":\n'
        "            lora_state = torch.load(path, map_location=self.device)\n"
        "            current_state = _dst_model.state_dict()\n"
        "            for key, value in lora_state.items():\n"
        "                if key in current_state:\n"
        "                    current_state[key] = value\n"
        "            _dst_model.load_state_dict(current_state)\n"
        '        elif path.suffix == ".pt":\n'
        "            state_dict = torch.load(path, map_location=self.device)\n"
        "            _dst_model.load_state_dict(state_dict, strict=False)\n"
    ),
    label="load_checkpoint: load into _raw_model",
)

# (H) push_to_hub: pull state_dict from _raw_model
src = patch(
    src,
    old=(
        "        lora_state = {}\n"
        "        for key, value in self.model.state_dict().items():\n"
        '            if "lora_A_" in key or "lora_B_" in key:\n'
        "                lora_state[key] = value\n"
        "        torch.save(lora_state, lora_path)\n"
    ),
    new=(
        "        _src_model = getattr(self, '_raw_model', self.model)\n"
        "        lora_state = {}\n"
        "        for key, value in _src_model.state_dict().items():\n"
        '            if "lora_A_" in key or "lora_B_" in key:\n'
        "                lora_state[key] = value\n"
        "        torch.save(lora_state, lora_path)\n"
    ),
    label="push_to_hub: LoRA via _raw_model",
)

TRAINER.write_text(src)
print(f"  wrote {TRAINER}\n")


# ---------------------------------------------------------------------------
# modal_training.py
# ---------------------------------------------------------------------------
print(f"Patching {MODAL}...")
mod = MODAL.read_text()

mod = patch(
    mod,
    old=(
        '    gpu="H100:2",  # Multi-GPU for faster training\n'
        "    timeout=60 * 60 * 24,  # 24 hours max (Modal limit)\n"
        "    cpu=16,\n"
        "    memory=128 * 1024,\n"
    ),
    new=(
        '    gpu="B200:4",  # 4x B200 for fastest training\n'
        "    timeout=60 * 60 * 24,  # 24 hours max (Modal limit)\n"
        "    cpu=32,\n"
        "    memory=256 * 1024,\n"
    ),
    label="modal train_multigpu: H100:2 -> B200:4 + scaled cpu/mem",
)

mod = patch(
    mod,
    old=(
        '        "torchrun",\n'
        '        "--nproc_per_node=2",\n'
        '        "--master_port=29500",\n'
    ),
    new=(
        '        "torchrun",\n'
        '        "--nproc_per_node=4",\n'
        '        "--master_port=29500",\n'
    ),
    label="modal torchrun: --nproc_per_node 2 -> 4",
)

mod = patch(
    mod,
    old='    Same as `train` but uses 2x H100 GPUs for faster training on larger datasets.\n',
    new='    Same as `train` but uses 4x B200 GPUs for fastest training on larger datasets.\n',
    label="modal docstring",
)

MODAL.write_text(mod)
print(f"  wrote {MODAL}\n")

print("All DDP finalization patches applied.")
print("Next steps:")
print("  1. Review the diff: git diff simlingo/scripts/nvidia_trainer.py simlingo/modal_training.py")
print("  2. Launch training: modal run --detach simlingo/modal_training.py::train_multigpu")
