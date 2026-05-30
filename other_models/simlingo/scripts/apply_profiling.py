"""Idempotent applier: instrument train_step_batched + train_step with per-section
timing so we can diagnose why vectorized is slower than per-sample.

Strategy
--------
* Insert a small `_prof_mark(name, state)` helper at module scope (after imports).
* In `train_step_batched`, mark these section boundaries:
    - image_prep        (per-sample CPU image preprocessing + cat)
    - meta_actions      (rule-based meta-action string generation)
    - lang_label_build  (per-sample tokenizer + chat template loop)
    - lang_label_pad    (batched LanguageLabel padding + concat)
    - driving_input     (build batched DrivingInput)
    - forward           (model.forward call + MDN handling)
    - loss_compute      (compute_loss)
    - backward          (loss_scaled.backward())
* In `train_step` (unvec), mark cumulative totals across the per-sample loop:
    - image_prep_total, lang_build_total, forward_total, loss_total, backward_total
* On rank-0, every 5 global_steps, print and wandb.log the timings as
  `prof_vec/*_ms` or `prof_unvec/*_ms`.

This applier is idempotent: it inserts a `# PROF_APPLIED` sentinel and refuses
to re-patch if already present.

Usage:
    python scripts/apply_profiling.py             # apply
    python scripts/apply_profiling.py --revert    # remove instrumentation
"""

from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

TRAINER = Path(__file__).parent / "nvidia_trainer.py"
SENTINEL = "# PROF_APPLIED_v1"

# -----------------------------------------------------------------------------
# Patches: each is (anchor, replacement). All must match exactly once.
# -----------------------------------------------------------------------------

HELPER_ANCHOR = "from PIL import Image\n"
HELPER_INSERT = """from PIL import Image

""" + SENTINEL + """
def _prof_mark(name, state):
    \"\"\"Profile helper: cuda-sync + record elapsed since last mark.\"\"\"
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    now = time.perf_counter()
    state[name] = now - state["_last"]
    state["_last"] = now


def _prof_log(path_prefix, step, is_main, prof):
    \"\"\"Print + wandb.log profile dict if rank-0. No-op otherwise.\"\"\"
    if not is_main:
        return
    if step % 5 != 0:
        return
    prof = {k: v for k, v in prof.items() if k != "_last"}
    msg = " ".join(f"{k}={v*1000:.0f}ms" for k, v in prof.items())
    print(f"[PROF {path_prefix} step={step}] {msg}", flush=True)
    try:
        import wandb as _wandb
        _wandb.log({f"prof_{path_prefix}/{k}_ms": v * 1000 for k, v in prof.items()},
                   step=step)
    except Exception:
        pass

"""

# train_step_batched: insert _prof init, then marks after each section, then log

# Section 1: init prof at start of method body (after tokenizer check)
VEC_INIT_ANCHOR = '''        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not available - cannot build language labels")

        # ----- Preprocess all images -----
'''
VEC_INIT_REPLACE = '''        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not available - cannot build language labels")

        _prof = {"_last": time.perf_counter()}
        # ----- Preprocess all images -----
'''

# Section 2: mark image_prep done (right before num_patches line)
VEC_AFTER_IMG_ANCHOR = '''        num_patches = pixel_values_batch.shape[2]
        num_image_tokens = num_patches * 256
'''
VEC_AFTER_IMG_REPLACE = '''        _prof_mark("image_prep", _prof)
        num_patches = pixel_values_batch.shape[2]
        num_image_tokens = num_patches * 256
'''

# Section 3: after meta_actions list comp
VEC_AFTER_META_ANCHOR = '''        meta_actions = [
            waypoints_to_meta_action(
                batch["waypoints"][i].cpu().numpy(),
                current_speed_mps=float(batch["speed_mps"][i].item()),
            )
            for i in range(B)
        ]
        prompt_lls, prompt_inf_lls = self._build_batch_language_labels(
'''
VEC_AFTER_META_REPLACE = '''        meta_actions = [
            waypoints_to_meta_action(
                batch["waypoints"][i].cpu().numpy(),
                current_speed_mps=float(batch["speed_mps"][i].item()),
            )
            for i in range(B)
        ]
        _prof_mark("meta_actions", _prof)
        prompt_lls, prompt_inf_lls = self._build_batch_language_labels(
'''

# Section 4: after _build_batch_language_labels returns (before pad section comment)
VEC_AFTER_LANGBUILD_ANCHOR = '''            meta_actions=meta_actions,
        )

        # ----- Stack LanguageLabels into batched LanguageLabel -----
'''
VEC_AFTER_LANGBUILD_REPLACE = '''            meta_actions=meta_actions,
        )
        _prof_mark("lang_label_build", _prof)

        # ----- Stack LanguageLabels into batched LanguageLabel -----
'''

# Section 5: after batched language label pad (before driving input build comment)
VEC_AFTER_LANGPAD_ANCHOR = '''        prompt_inf_ll_batched = self._build_batched_language_label(prompt_inf_lls)

        # ----- Build batched DrivingInput -----
'''
VEC_AFTER_LANGPAD_REPLACE = '''        prompt_inf_ll_batched = self._build_batched_language_label(prompt_inf_lls)
        _prof_mark("lang_label_pad", _prof)

        # ----- Build batched DrivingInput -----
'''

# Section 6: after driving_input build (before forward section comment)
VEC_AFTER_DI_ANCHOR = '''            HW_ref,
        )

        # ----- One forward pass for the whole batch -----
        speed_wps, route_wps, language = self.model(driving_input)
'''
VEC_AFTER_DI_REPLACE = '''            HW_ref,
        )
        _prof_mark("driving_input", _prof)

        # ----- One forward pass for the whole batch -----
        speed_wps, route_wps, language = self.model(driving_input)
        _prof_mark("forward", _prof)
'''

# Section 7: after compute_loss (before loss_scaled / backward)
VEC_AFTER_LOSS_ANCHOR = '''        loss, loss_dict = self.compute_loss(
            pred_wps,
            route_wps.float() if route_wps is not None else None,
            gt_wps_aligned,
            gt_route,
        )

        # compute_loss already does mean reduction across batch, so we only
        # divide by grad_accum (not also by batch_size).
        loss_scaled = loss / grad_accum
        loss_scaled.backward()

'''
VEC_AFTER_LOSS_REPLACE = '''        loss, loss_dict = self.compute_loss(
            pred_wps,
            route_wps.float() if route_wps is not None else None,
            gt_wps_aligned,
            gt_route,
        )
        _prof_mark("loss_compute", _prof)

        # compute_loss already does mean reduction across batch, so we only
        # divide by grad_accum (not also by batch_size).
        loss_scaled = loss / grad_accum
        loss_scaled.backward()
        _prof_mark("backward", _prof)
        _prof_log("vec",
                  getattr(self, "global_step", 0),
                  getattr(self, "is_main", True),
                  _prof)

'''

# -----------------------------------------------------------------------------
# train_step (unvec): track cumulative per-section totals
# -----------------------------------------------------------------------------

UNVEC_INIT_ANCHOR = '''        # Process each sample in the batch
        # Note: SimLingo's DrivingInput is designed for batch_size=1 internally,
        # so we process samples one at a time and accumulate gradients
        total_wp_loss = 0.0
        total_route_loss = 0.0
        valid_samples = 0

        for i in range(batch_size):
'''
UNVEC_INIT_REPLACE = '''        # Process each sample in the batch
        # Note: SimLingo's DrivingInput is designed for batch_size=1 internally,
        # so we process samples one at a time and accumulate gradients
        total_wp_loss = 0.0
        total_route_loss = 0.0
        valid_samples = 0

        _prof_totals = {"image_prep": 0.0, "lang_build": 0.0, "forward": 0.0,
                        "loss_compute": 0.0, "backward": 0.0}
        for i in range(batch_size):
            _prof_iter = {"_last": time.perf_counter()}
'''

# After _process_single_image
UNVEC_AFTER_IMG_ANCHOR = '''                # Process image for model
                pixel_values, HW = self._process_single_image(image)

                if self.global_step == 0 and i == 0:
                    print(f"  Processed image shape: {pixel_values.shape}, HW: {HW}")
'''
UNVEC_AFTER_IMG_REPLACE = '''                # Process image for model
                pixel_values, HW = self._process_single_image(image)
                _prof_mark("image_prep", _prof_iter)

                if self.global_step == 0 and i == 0:
                    print(f"  Processed image shape: {pixel_values.shape}, HW: {HW}")
'''

# After _build_batch_language_labels + _build_driving_input_single
UNVEC_AFTER_LANG_ANCHOR = '''                # Build DrivingInput
                driving_input = self._build_driving_input_single(
                    pixel_values, speed, target_points, prompt_lls[0], prompt_inf_lls[0], HW
                )

                # --- Forward pass --------------------------------------------------
'''
UNVEC_AFTER_LANG_REPLACE = '''                # Build DrivingInput
                driving_input = self._build_driving_input_single(
                    pixel_values, speed, target_points, prompt_lls[0], prompt_inf_lls[0], HW
                )
                _prof_mark("lang_build", _prof_iter)

                # --- Forward pass --------------------------------------------------
'''

# After forward (after MDN/non-MDN selection, before shape alignment block)
UNVEC_AFTER_FWD_ANCHOR = '''                # --- Skip if no prediction -----------------------------------------
                if pred_wps is None:
                    print(f"Warning: Sample {i} - Model returned None for waypoints")
                    continue
'''
UNVEC_AFTER_FWD_REPLACE = '''                # --- Skip if no prediction -----------------------------------------
                if pred_wps is None:
                    print(f"Warning: Sample {i} - Model returned None for waypoints")
                    continue
                _prof_mark("forward", _prof_iter)
'''

# After backward
UNVEC_AFTER_BWD_ANCHOR = '''                # Backward pass for this sample (accumulate gradients)
                # Scale by both batch_size and grad_accum for proper averaging
                loss_scaled = loss / (batch_size * grad_accum)
                loss_scaled.backward()

                total_wp_loss += loss_dict["wp_loss"]
                total_route_loss += loss_dict["route_loss"]
                valid_samples += 1
'''
UNVEC_AFTER_BWD_REPLACE = '''                # Backward pass for this sample (accumulate gradients)
                # Scale by both batch_size and grad_accum for proper averaging
                _prof_mark("loss_compute", _prof_iter)
                loss_scaled = loss / (batch_size * grad_accum)
                loss_scaled.backward()
                _prof_mark("backward", _prof_iter)

                for _k in ("image_prep", "lang_build", "forward", "loss_compute", "backward"):
                    _prof_totals[_k] += _prof_iter.get(_k, 0.0)

                total_wp_loss += loss_dict["wp_loss"]
                total_route_loss += loss_dict["route_loss"]
                valid_samples += 1
'''

# After the for-loop ends (before "If no valid samples")
UNVEC_AFTER_LOOP_ANCHOR = '''        # If no valid samples, return zero loss
        if valid_samples == 0:
'''
UNVEC_AFTER_LOOP_REPLACE = '''        _prof_totals["_last"] = 0.0  # satisfy _prof_log contract
        _prof_log("unvec",
                  getattr(self, "global_step", 0),
                  getattr(self, "is_main", True),
                  _prof_totals)

        # If no valid samples, return zero loss
        if valid_samples == 0:
'''

PATCHES = [
    ("helper",               HELPER_ANCHOR,            HELPER_INSERT),
    ("vec_init",             VEC_INIT_ANCHOR,          VEC_INIT_REPLACE),
    ("vec_after_img",        VEC_AFTER_IMG_ANCHOR,     VEC_AFTER_IMG_REPLACE),
    ("vec_after_meta",       VEC_AFTER_META_ANCHOR,    VEC_AFTER_META_REPLACE),
    ("vec_after_langbuild",  VEC_AFTER_LANGBUILD_ANCHOR, VEC_AFTER_LANGBUILD_REPLACE),
    ("vec_after_langpad",    VEC_AFTER_LANGPAD_ANCHOR, VEC_AFTER_LANGPAD_REPLACE),
    ("vec_after_di",         VEC_AFTER_DI_ANCHOR,      VEC_AFTER_DI_REPLACE),
    ("vec_after_loss",       VEC_AFTER_LOSS_ANCHOR,    VEC_AFTER_LOSS_REPLACE),
    ("unvec_init",           UNVEC_INIT_ANCHOR,        UNVEC_INIT_REPLACE),
    ("unvec_after_img",      UNVEC_AFTER_IMG_ANCHOR,   UNVEC_AFTER_IMG_REPLACE),
    ("unvec_after_lang",     UNVEC_AFTER_LANG_ANCHOR,  UNVEC_AFTER_LANG_REPLACE),
    ("unvec_after_fwd",      UNVEC_AFTER_FWD_ANCHOR,   UNVEC_AFTER_FWD_REPLACE),
    ("unvec_after_bwd",      UNVEC_AFTER_BWD_ANCHOR,   UNVEC_AFTER_BWD_REPLACE),
    ("unvec_after_loop",     UNVEC_AFTER_LOOP_ANCHOR,  UNVEC_AFTER_LOOP_REPLACE),
]


def apply(content: str) -> str:
    if SENTINEL in content:
        print(f"Sentinel '{SENTINEL}' already present. No-op.", file=sys.stderr)
        return content

    new = content
    for name, anchor, replacement in PATCHES:
        count = new.count(anchor)
        if count != 1:
            raise RuntimeError(
                f"Anchor '{name}' matched {count} times (expected 1). "
                f"Refusing to apply ambiguous patch."
            )
        new = new.replace(anchor, replacement, 1)
        print(f"  applied: {name}", file=sys.stderr)
    return new


def revert(content: str) -> str:
    if SENTINEL not in content:
        print(f"Sentinel '{SENTINEL}' not present. No-op.", file=sys.stderr)
        return content

    new = content
    for name, anchor, replacement in reversed(PATCHES):
        count = new.count(replacement)
        if count != 1:
            raise RuntimeError(
                f"Reverse anchor '{name}' matched {count} times. "
                f"Cannot cleanly revert."
            )
        new = new.replace(replacement, anchor, 1)
        print(f"  reverted: {name}", file=sys.stderr)
    return new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revert", action="store_true")
    args = parser.parse_args()

    content = TRAINER.read_text()
    if args.revert:
        new = revert(content)
    else:
        new = apply(content)
    TRAINER.write_text(new)
    print(f"OK: wrote {TRAINER}", file=sys.stderr)


if __name__ == "__main__":
    main()
