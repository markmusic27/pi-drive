"""Add opt-in vectorized train_step to NVIDIASimLingoTrainer.

Run from repo root:
    python simlingo/scripts/apply_vectorization.py

Adds:
  1. A new method `_build_batched_language_label(prompt_lls)` that stacks
     a list of per-sample LanguageLabel objects into one batched LanguageLabel
     with right-padding (pad_id=0 by default) on phrase_ids/phrase_valid/
     phrase_mask + concatenated placeholder_values list.
  2. A new method `_build_driving_input_batched(...)` that builds one DrivingInput
     with [B, ...] tensors for camera_images, intrinsics, extrinsics, speed,
     target_point + the batched LanguageLabel.
  3. A new method `train_step_batched(batch, accumulation_step, grad_accum)`
     that does ONE forward+backward over the whole batch (no per-sample loop).
  4. A dispatch in train() that picks `train_step_batched` when the config
     flag `training.vectorized: true` is set; otherwise falls back to the
     existing `train_step` (default behavior preserved).
  5. Forces `max_num=1` in `_process_single_image` ONLY when vectorized=true,
     so all images yield P=1 and can be safely stacked.

Default behavior unchanged: existing configs that don't set
`training.vectorized` continue to use the per-sample loop.

To enable, set in your yaml:
    training:
      vectorized: true

A separate smoke config `config/nvidia_smoke_vec.yaml` is also written to
let you A/B test the batched path against the unvectorized smoke.

Idempotent — safe to re-run.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINER = REPO_ROOT / "simlingo" / "scripts" / "nvidia_trainer.py"
SMOKE_VEC_YAML = REPO_ROOT / "simlingo" / "config" / "nvidia_smoke_vec.yaml"


def patch(text: str, old: str, new: str, label: str) -> str:
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
# 1. _process_single_image: honor `vectorized` flag to force max_num=1
# ---------------------------------------------------------------------------
print(f"Patching {TRAINER}...")
src = TRAINER.read_text()

src = patch(
    src,
    old=(
        "        # Apply InternVL2 dynamic preprocessing\n"
        "        pil_img = Image.fromarray(rgb)\n"
        "        patches = self.dynamic_preprocess(\n"
        "            pil_img,\n"
        "            image_size=448,\n"
        "            use_thumbnail=self.use_global_img,\n"
        "            max_num=2,\n"
        "        )\n"
    ),
    new=(
        "        # Apply InternVL2 dynamic preprocessing.\n"
        "        # When vectorized training is enabled, force max_num=1 so all images\n"
        "        # yield P=1 patches and can be safely stacked across the batch.\n"
        "        _vec_on = self.config.get('training', {}).get('vectorized', False)\n"
        "        pil_img = Image.fromarray(rgb)\n"
        "        patches = self.dynamic_preprocess(\n"
        "            pil_img,\n"
        "            image_size=448,\n"
        "            use_thumbnail=self.use_global_img,\n"
        "            max_num=1 if _vec_on else 2,\n"
        "        )\n"
    ),
    label="_process_single_image: force max_num=1 when vectorized",
)


# ---------------------------------------------------------------------------
# 2. Insert new methods after _build_driving_input_single
# ---------------------------------------------------------------------------
new_methods = '''
    def _build_batched_language_label(self, prompt_lls: list):
        """Stack a list of per-sample LanguageLabel objects into one batched
        LanguageLabel with right-padding on phrase_ids/phrase_valid/phrase_mask.

        Each input LanguageLabel has phrase_ids shape [1, tokens_i] (varying tokens_i).
        Output has shape [B, max_tokens], padded with pad_id=0 on the right.
        placeholder_values lists are concatenated (one dict per sample).
        """
        import torch.nn.functional as F

        max_len = max(ll.phrase_ids.shape[1] for ll in prompt_lls)
        pad_id = getattr(self.tokenizer, "pad_token_id", None) or 0

        phrase_ids_list = []
        phrase_valid_list = []
        phrase_mask_list = []
        loss_masking_list = []
        placeholder_values_combined = []
        language_strings = []
        any_loss_masking = False

        for ll in prompt_lls:
            cur_len = ll.phrase_ids.shape[1]
            pad_len = max_len - cur_len

            if pad_len > 0:
                pids = F.pad(ll.phrase_ids, (0, pad_len), value=pad_id)
                pvalid = F.pad(ll.phrase_valid.to(torch.bool), (0, pad_len), value=False)
                pmask = F.pad(ll.phrase_mask.to(torch.bool), (0, pad_len), value=False)
            else:
                pids = ll.phrase_ids
                pvalid = ll.phrase_valid.to(torch.bool)
                pmask = ll.phrase_mask.to(torch.bool)

            phrase_ids_list.append(pids)
            phrase_valid_list.append(pvalid)
            phrase_mask_list.append(pmask)

            if ll.loss_masking is not None:
                any_loss_masking = True
                lm = ll.loss_masking
                if pad_len > 0:
                    lm = F.pad(lm.to(torch.bool), (0, pad_len), value=False)
                loss_masking_list.append(lm)
            else:
                loss_masking_list.append(None)

            # placeholder_values is a list of dicts (one per sample). Each
            # per-sample LL was built with a single-element list, so we
            # extend rather than append.
            placeholder_values_combined.extend(ll.placeholder_values)
            language_strings.append(ll.language_string)

        phrase_ids = torch.cat(phrase_ids_list, dim=0)
        phrase_valid = torch.cat(phrase_valid_list, dim=0)
        phrase_mask = torch.cat(phrase_mask_list, dim=0)

        if any_loss_masking:
            # Replace any None entries with all-False masks of max_len
            for i, lm in enumerate(loss_masking_list):
                if lm is None:
                    loss_masking_list[i] = torch.zeros(
                        (1, max_len), dtype=torch.bool, device=phrase_ids.device
                    )
            loss_masking = torch.cat(loss_masking_list, dim=0)
        else:
            loss_masking = None

        return self.LanguageLabel(
            phrase_ids=phrase_ids,
            phrase_valid=phrase_valid,
            phrase_mask=phrase_mask,
            placeholder_values=placeholder_values_combined,
            language_string=language_strings,
            loss_masking=loss_masking,
        )

    def _build_driving_input_batched(
        self,
        pixel_values_batch: torch.Tensor,  # [B, 1, P, 3, 448, 448]
        speeds: torch.Tensor,              # [B]
        target_points_batch: torch.Tensor, # [B, 2, 2]
        prompt_ll_batched,                  # batched LanguageLabel
        prompt_inf_ll_batched,              # batched LanguageLabel
        HW: tuple[int, int],
    ):
        """Build a single DrivingInput with [B, ...] tensors."""
        B = pixel_values_batch.shape[0]
        H, W = HW
        K = self.get_camera_intrinsics(W, H, 110).to(self.device).float()        # [3, 3]
        E = self.get_camera_extrinsics().to(self.device).float()                  # [4, 4]
        intrinsics = K.unsqueeze(0).expand(B, -1, -1).contiguous()                # [B, 3, 3]
        extrinsics = E.unsqueeze(0).expand(B, -1, -1).contiguous()                # [B, 4, 4]

        vehicle_speed = speeds.to(self.device).float().reshape(B, 1)              # [B, 1]
        # Take the first target point per sample (matches single-sample build)
        target_point = target_points_batch[:, 0, :].to(self.device).float()       # [B, 2]

        return self.DrivingInput(
            camera_images=pixel_values_batch.to(self.device).bfloat16(),
            image_sizes=None,
            camera_intrinsics=intrinsics,
            camera_extrinsics=extrinsics,
            vehicle_speed=vehicle_speed,
            target_point=target_point,
            prompt=prompt_ll_batched,
            prompt_inference=prompt_inf_ll_batched,
        )

    def train_step_batched(self, batch: dict, accumulation_step: int = 0, grad_accum: int = 1) -> dict:
        """Fully batched training step: one forward + one backward over the whole batch.

        Replaces the per-sample loop in train_step. Use only when
        `training.vectorized: true` is set in the config AND the dataset is
        configured to produce a stable image patch count (max_num=1 path).
        """
        self.model.train()
        B = batch["image"].shape[0]
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not available - cannot build language labels")

        # ----- Preprocess all images -----
        pixel_values_list = []
        HW_ref = None
        for i in range(B):
            pv, hw = self._process_single_image(batch["image"][i])
            if HW_ref is None:
                HW_ref = hw
            pixel_values_list.append(pv)  # each: [1, 1, P, 3, 448, 448]

        # All should have same P (vectorized forces max_num=1 -> P=1)
        try:
            pixel_values_batch = torch.cat(pixel_values_list, dim=0)  # [B, 1, P, 3, 448, 448]
        except RuntimeError as e:
            raise RuntimeError(
                "Failed to stack pixel_values across batch. Ensure training.vectorized is "
                "set so max_num=1 is forced. Underlying error: " + str(e)
            )

        num_patches = pixel_values_batch.shape[2]
        num_image_tokens = num_patches * 256

        # ----- Build per-sample LanguageLabels -----
        meta_actions = [
            waypoints_to_meta_action(
                batch["waypoints"][i].cpu().numpy(),
                current_speed_mps=float(batch["speed_mps"][i].item()),
            )
            for i in range(B)
        ]
        prompt_lls, prompt_inf_lls = self._build_batch_language_labels(
            batch["speed_mps"],
            batch["target_points"],
            num_image_tokens,
            meta_actions=meta_actions,
        )

        # ----- Stack LanguageLabels into batched LanguageLabel -----
        prompt_ll_batched = self._build_batched_language_label(prompt_lls)
        prompt_inf_ll_batched = self._build_batched_language_label(prompt_inf_lls)

        # ----- Build batched DrivingInput -----
        driving_input = self._build_driving_input_batched(
            pixel_values_batch,
            batch["speed_mps"],
            batch["target_points"],
            prompt_ll_batched,
            prompt_inf_ll_batched,
            HW_ref,
        )

        # ----- One forward pass for the whole batch -----
        speed_wps, route_wps, language = self.model(driving_input)

        if self.use_mdn and self.mdn_head is not None:
            captured = self._last_features
            if captured is None:
                raise RuntimeError("MDN mode enabled but no features captured by hook")
            if captured.dim() == 3:
                n_drive = min(self.config["model"].get("num_waypoints", 11), captured.shape[1])
                feats = captured[:, -n_drive:, :].mean(dim=1)
            else:
                feats = captured
            mdn_output = self.mdn_head(feats.float())
            pred_wps = mdn_output  # dict with pi, mu, sigma
            self._last_features = None
        else:
            pred_wps = speed_wps

        if pred_wps is None:
            raise RuntimeError("Model returned None for waypoints in batched train_step")

        gt_wps = batch["waypoints"].to(self.device)  # [B, 11, 2]
        gt_route = batch["route"].to(self.device)    # [B, 20, 2]

        # ----- Shape alignment -----
        if isinstance(pred_wps, dict):
            gt_wps_aligned = gt_wps
        else:
            pred_wps = pred_wps.float()
            if pred_wps.shape[1] != gt_wps.shape[1]:
                min_len = min(pred_wps.shape[1], gt_wps.shape[1])
                pred_wps = pred_wps[:, :min_len, :]
                gt_wps_aligned = gt_wps[:, :min_len, :]
            else:
                gt_wps_aligned = gt_wps

        # ----- Single loss + backward for the whole batch -----
        loss, loss_dict = self.compute_loss(
            pred_wps,
            route_wps.float() if route_wps is not None else None,
            gt_wps_aligned,
            gt_route,
        )

        # compute_loss already does mean reduction across batch, so we only
        # divide by grad_accum (not also by batch_size).
        loss_scaled = loss / grad_accum
        loss_scaled.backward()

        return {
            "total_loss": loss_dict["total_loss"],
            "wp_loss": loss_dict["wp_loss"],
            "route_loss": loss_dict["route_loss"],
        }
'''

# Insert the new methods right after _build_driving_input_single's return
src = patch(
    src,
    old=(
        "            prompt=prompt_ll,\n"
        "            prompt_inference=prompt_inf_ll,\n"
        "        )\n"
        "\n"
        "    def compute_loss(\n"
    ),
    new=(
        "            prompt=prompt_ll,\n"
        "            prompt_inference=prompt_inf_ll,\n"
        "        )\n"
        + new_methods
        + "\n    def compute_loss(\n"
    ),
    label="insert batched methods after _build_driving_input_single",
)


# ---------------------------------------------------------------------------
# 3. Dispatch in train() loop: pick batched vs per-sample step function
# ---------------------------------------------------------------------------
src = patch(
    src,
    old=(
        "                # Training step (accumulates gradients)\n"
        "                loss_dict = self.train_step(batch, step % grad_accum, grad_accum)\n"
    ),
    new=(
        "                # Training step (accumulates gradients)\n"
        "                # Dispatch to batched implementation when opted in.\n"
        "                _step_fn = self.train_step_batched if self.config.get('training', {}).get('vectorized', False) else self.train_step\n"
        "                loss_dict = _step_fn(batch, step % grad_accum, grad_accum)\n"
    ),
    label="train(): dispatch vectorized vs per-sample step",
)

TRAINER.write_text(src)
print(f"  wrote {TRAINER}\n")


# ---------------------------------------------------------------------------
# 4. Write a vectorized smoke yaml (clone of nvidia_smoke.yaml + flag)
# ---------------------------------------------------------------------------
print(f"Writing {SMOKE_VEC_YAML}...")
SMOKE_VEC_YAML.write_text(
    (REPO_ROOT / "simlingo" / "config" / "nvidia_smoke.yaml").read_text()
    .replace(
        "# Smoke test uses the tiny extract (50 clips) at /nvidia_data/extracted.",
        "# VECTORIZED smoke: same as nvidia_smoke.yaml but with training.vectorized: true.",
    )
    .replace(
        "  precision: bf16\n  gradient_checkpointing: false\n  max_grad_norm: 1.0\n",
        (
            "  precision: bf16\n"
            "  gradient_checkpointing: false\n"
            "  max_grad_norm: 1.0\n"
            "\n"
            "  # Tier A: vectorize the per-sample forward loop.\n"
            "  # Forces max_num=1 in dynamic_preprocess so all images yield P=1.\n"
            "  vectorized: true\n"
        ),
    )
    .replace(
        '  project: "simlingo-nvidia-smoke"',
        '  project: "simlingo-nvidia-smoke-vec"',
    )
)
print(f"  wrote {SMOKE_VEC_YAML}\n")


print("Vectorization scaffolding applied.")
print("To smoke-test the batched path on H100:4:")
print("  cd simlingo && modal run --detach modal_training.py::train_multigpu \\")
print("    --config-path /app/config/nvidia_smoke_vec.yaml \\")
print("    --wandb-project simlingo-nvidia-smoke-vec")
print()
print("Compare loss curve to the unvectorized smoke run. If they match,")
print("flip `training.vectorized: true` in nvidia_finetune.yaml for prod.")
