"""Visualize extracted driving data on Modal and download results.

Usage:
    modal run pi05/modal_visualize.py::visualize --scale tiny
    # Then download: modal volume get pi05-cache /cache/viz/ ./pi05/viz_output/
"""

from __future__ import annotations

import modal

APP_NAME = "pi05-visualize"
CACHE_DIR = "/cache"

viz_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "matplotlib",
        "pandas",
        "pyarrow",
        "Pillow",
        "numpy",
    )
)

cache_volume = modal.Volume.from_name("pi05-cache", create_if_missing=True)
VOLUMES = {CACHE_DIR: cache_volume}

app = modal.App(APP_NAME)


@app.function(
    image=viz_image,
    volumes=VOLUMES,
    timeout=60 * 10,
    memory=8 * 1024,
)
def visualize(scale: str = "tiny"):
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from PIL import Image

    output_dir = f"{CACHE_DIR}/extracted/{scale}"
    viz_dir = f"{CACHE_DIR}/viz"
    os.makedirs(viz_dir, exist_ok=True)

    samples_path = f"{output_dir}/samples.parquet"
    if not os.path.exists(samples_path):
        raise FileNotFoundError(f"No extracted data at {samples_path}")

    df = pd.read_parquet(samples_path)
    print(f"Loaded {len(df)} samples from {samples_path}")

    # Debug action structure from parquet
    sample_action = df["actions"].iloc[0]
    print(f"Action type: {type(sample_action)}, len: {len(sample_action)}")
    if hasattr(sample_action, '__len__') and len(sample_action) > 0:
        print(f"  First element type: {type(sample_action[0])}, value: {sample_action[0]}")

    # Parse nested lists from parquet
    actions_list = []
    for a in df["actions"].values:
        arr = np.array([np.array(step, dtype=np.float32) for step in a])
        actions_list.append(arr)
    actions_all = np.stack(actions_list)  # (N, 64, 2)
    print(f"Actions shape: {actions_all.shape}")

    # --- 1. Navigation prompt distribution ---
    task_counts = df["nav_prompt"].value_counts()
    print("\nNavigation prompt distribution:")
    for label, count in task_counts.items():
        print(f"  {label}: {count} ({100*count/len(df):.1f}%)")

    has_traj = "nav_prompt_traj" in df.columns

    if has_traj:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        traj_counts = df["nav_prompt_traj"].value_counts()

        labels_v = task_counts.index.tolist()
        colors = plt.cm.Set2(np.linspace(0, 1, max(len(labels_v), len(traj_counts))))

        bars = axes[0].barh(labels_v, task_counts.values, color=colors[:len(labels_v)])
        axes[0].set_xlabel("Sample count")
        axes[0].set_title("VLM Navigation Labels (Gemini Flash)")
        for bar, count in zip(bars, task_counts.values):
            axes[0].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                         str(count), va="center", fontsize=9)

        labels_t = traj_counts.index.tolist()
        bars = axes[1].barh(labels_t, traj_counts.values, color=colors[:len(labels_t)])
        axes[1].set_xlabel("Sample count")
        axes[1].set_title("Trajectory-Based Labels (baseline)")
        for bar, count in zip(bars, traj_counts.values):
            axes[1].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                         str(count), va="center", fontsize=9)
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        labels_v = task_counts.index.tolist()
        colors = plt.cm.Set2(np.linspace(0, 1, len(labels_v)))
        bars = ax.barh(labels_v, task_counts.values, color=colors)
        ax.set_xlabel("Sample count")
        ax.set_title("Navigation Prompt Distribution")
        for bar, count in zip(bars, task_counts.values):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    str(count), va="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(f"{viz_dir}/nav_distribution.png", dpi=150)
    print("Saved nav_distribution.png")
    plt.close()

    # --- 2. Action distributions (all 64 timesteps) ---
    accels = actions_all[:, :, 0].flatten()
    curvs = actions_all[:, :, 1].flatten()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(accels, bins=100, color="steelblue", edgecolor="none", alpha=0.8)
    axes[0].set_xlabel("Acceleration (m/s²)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Acceleration (all timesteps)\n"
                      f"μ={accels.mean():.4f}, σ={accels.std():.4f}, "
                      f"range=[{accels.min():.3f}, {accels.max():.3f}]")
    axes[0].axvline(0, color="red", linestyle="--", alpha=0.5)

    axes[1].hist(curvs, bins=100, color="coral", edgecolor="none", alpha=0.8)
    axes[1].set_xlabel("Curvature (1/m)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Curvature (all timesteps)\n"
                      f"μ={curvs.mean():.4f}, σ={curvs.std():.4f}, "
                      f"range=[{curvs.min():.5f}, {curvs.max():.5f}]")
    axes[1].axvline(0, color="red", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(f"{viz_dir}/action_distributions.png", dpi=150)
    print("Saved action_distributions.png")
    plt.close()

    # --- 3. State distributions ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    speeds = df["speed"].values
    axes[0].hist(speeds, bins=60, color="seagreen", edgecolor="none", alpha=0.8)
    axes[0].set_xlabel("Speed (m/s)")
    axes[0].set_ylabel("Count")
    mph = speeds * 2.237
    axes[0].set_title(f"Speed\nμ={speeds.mean():.2f} m/s ({mph.mean():.0f} mph), "
                      f"range=[{speeds.min():.1f}, {speeds.max():.1f}]")

    hr = df["heading_rate"].values
    axes[1].hist(hr, bins=60, color="orchid", edgecolor="none", alpha=0.8)
    axes[1].set_xlabel("Heading Rate (rad/s)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Heading Rate\nμ={hr.mean():.4f}, σ={hr.std():.4f}, "
                      f"range=[{hr.min():.3f}, {hr.max():.3f}]")
    axes[1].axvline(0, color="red", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(f"{viz_dir}/state_distributions.png", dpi=150)
    print("Saved state_distributions.png")
    plt.close()

    # --- 4. Sample image grid with labels ---
    nav_cats = task_counts.index.tolist()
    fig, axes = plt.subplots(len(nav_cats), 3, figsize=(15, 4 * len(nav_cats)))
    if len(nav_cats) == 1:
        axes = [axes]

    for row_i, cat in enumerate(nav_cats):
        cat_df = df[df["nav_prompt"] == cat]
        samples = cat_df.head(3)
        for col_i, (_, sample) in enumerate(samples.iterrows()):
            img_path = f"{output_dir}/{sample['image_path']}"
            ax = axes[row_i][col_i] if len(nav_cats) > 1 else axes[0][col_i]
            if os.path.exists(img_path):
                img = Image.open(img_path)
                ax.imshow(img)
            ax.set_title(
                f'"{cat}"\n'
                f'spd={sample["speed"]:.1f} m/s  '
                f'lat={sample.get("lateral_disp", 0):.1f}m  '
                f'hdg={sample.get("heading_change_deg", 0):.0f}°',
                fontsize=8
            )
            ax.axis("off")
        for col_i in range(len(samples), 3):
            ax = axes[row_i][col_i] if len(nav_cats) > 1 else axes[0][col_i]
            ax.axis("off")

    plt.suptitle("Sample Images by VLM Navigation Label", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(f"{viz_dir}/sample_grid.png", dpi=150, bbox_inches="tight")
    print("Saved sample_grid.png")
    plt.close()

    # --- 5. Action trajectories by nav category ---
    target_cats = [c for c in ["continue straight", "turn left", "turn right"] if c in nav_cats]
    if len(target_cats) > 0:
        fig, axes = plt.subplots(2, len(target_cats), figsize=(5 * len(target_cats), 8))
        if len(target_cats) == 1:
            axes = [[axes[0]], [axes[1]]]

        for col, cat in enumerate(target_cats):
            cat_samples = df[df["nav_prompt"] == cat].head(5)
            t = np.arange(64) * 0.1
            for _, sample in cat_samples.iterrows():
                acts = np.array([np.array(s, dtype=np.float32) for s in sample["actions"]])
                axes[0][col].plot(t, acts[:, 0], alpha=0.6, linewidth=1)
                axes[1][col].plot(t, acts[:, 1], alpha=0.6, linewidth=1)

            axes[0][col].set_ylabel("Acceleration (m/s²)")
            axes[0][col].set_title(f'"{cat}" ({len(cat_samples)} samples)')
            axes[0][col].axhline(0, color="gray", linestyle="--", alpha=0.3)
            axes[1][col].set_ylabel("Curvature (1/m)")
            axes[1][col].set_xlabel("Time (s)")
            axes[1][col].axhline(0, color="gray", linestyle="--", alpha=0.3)

        plt.suptitle("Action Trajectories by Navigation Category (5 samples each)", fontsize=13)
        plt.tight_layout()
        fig.savefig(f"{viz_dir}/action_trajectories.png", dpi=150)
        print("Saved action_trajectories.png")
        plt.close()

    # --- 6. Action box plots by nav category ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Mean acceleration per sample, grouped by nav category
    mean_accels_by_cat = []
    mean_curvs_by_cat = []
    cat_labels = []
    for cat in nav_cats:
        cat_idx = df["nav_prompt"] == cat
        cat_actions = actions_all[cat_idx]
        mean_accels_by_cat.append(cat_actions[:, :, 0].mean(axis=1))
        mean_curvs_by_cat.append(cat_actions[:, :, 1].mean(axis=1))
        cat_labels.append(cat)

    bp1 = axes[0].boxplot(mean_accels_by_cat, labels=cat_labels, vert=True, patch_artist=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(cat_labels)))
    for patch, color in zip(bp1["boxes"], colors):
        patch.set_facecolor(color)
    axes[0].set_ylabel("Mean Acceleration (m/s²)")
    axes[0].set_title("Acceleration by Nav Category")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].axhline(0, color="gray", linestyle="--", alpha=0.3)

    bp2 = axes[1].boxplot(mean_curvs_by_cat, labels=cat_labels, vert=True, patch_artist=True)
    for patch, color in zip(bp2["boxes"], colors):
        patch.set_facecolor(color)
    axes[1].set_ylabel("Mean Curvature (1/m)")
    axes[1].set_title("Curvature by Nav Category")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].axhline(0, color="gray", linestyle="--", alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"{viz_dir}/action_by_nav.png", dpi=150)
    print("Saved action_by_nav.png")
    plt.close()

    # --- 7. VLM vs trajectory label confusion (if available) ---
    if has_traj:
        vlm_labels = df["nav_prompt"].values
        traj_labels = df["nav_prompt_traj"].values
        all_labels = sorted(set(list(vlm_labels) + list(traj_labels)))

        confusion = np.zeros((len(all_labels), len(all_labels)), dtype=int)
        for v, t in zip(vlm_labels, traj_labels):
            vi = all_labels.index(v)
            ti = all_labels.index(t)
            confusion[vi, ti] += 1

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(confusion, cmap="Blues")
        ax.set_xticks(range(len(all_labels)))
        ax.set_yticks(range(len(all_labels)))
        ax.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(all_labels, fontsize=9)
        ax.set_xlabel("Trajectory-based label")
        ax.set_ylabel("VLM label (Gemini Flash)")
        ax.set_title("VLM vs Trajectory Label Agreement")

        for i in range(len(all_labels)):
            for j in range(len(all_labels)):
                if confusion[i, j] > 0:
                    ax.text(j, i, str(confusion[i, j]), ha="center", va="center",
                            color="white" if confusion[i, j] > confusion.max() * 0.5 else "black",
                            fontsize=10)

        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        fig.savefig(f"{viz_dir}/vlm_vs_traj_confusion.png", dpi=150)
        print("Saved vlm_vs_traj_confusion.png")
        plt.close()

        agreement = (vlm_labels == traj_labels).mean()
        print(f"\nVLM vs trajectory agreement: {agreement:.1%}")

    # --- Summary ---
    print(f"\n{'='*50}")
    print(f"DATASET SUMMARY ({scale} scale)")
    print(f"{'='*50}")
    print(f"Total samples: {len(df)}")
    print(f"Train: {(df['split'] == 'train').sum()}, Eval: {(df['split'] == 'eval').sum()}")
    print(f"Action shape: (64, 2) per sample = {len(df) * 64} total action frames")
    print(f"\nAction stats (across all 64 timesteps):")
    print(f"  Acceleration:  μ={accels.mean():.4f}, σ={accels.std():.4f}, [{accels.min():.3f}, {accels.max():.3f}]")
    print(f"  Curvature:     μ={curvs.mean():.5f}, σ={curvs.std():.5f}, [{curvs.min():.5f}, {curvs.max():.5f}]")
    print(f"\nState stats:")
    print(f"  Speed:         μ={speeds.mean():.2f} ({mph.mean():.0f} mph), [{speeds.min():.1f}, {speeds.max():.1f}] m/s")
    print(f"  Heading rate:  μ={hr.mean():.4f}, [{hr.min():.3f}, {hr.max():.3f}] rad/s")

    cache_volume.commit()
    print(f"\nAll plots saved to {viz_dir}/")
    return viz_dir
