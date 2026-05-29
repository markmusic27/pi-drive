"""Visualize the pi05-driving-bc LeRobot v2 dataset from HuggingFace."""

import os
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
from huggingface_hub import hf_hub_download

REPO = "markmusic/pi05-driving-bc"
OUTPUT_DIR = "/Users/mmusic/Developer/Projects/cart/pi-drive/pi05/viz_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Downloading dataset files...")
data_path = hf_hub_download(REPO, "data/chunk-000/file-000.parquet", repo_type="dataset")
tasks_path = hf_hub_download(REPO, "meta/tasks.parquet", repo_type="dataset")
episodes_path = hf_hub_download(REPO, "meta/episodes/chunk-000/file-000.parquet", repo_type="dataset")
stats_path = hf_hub_download(REPO, "meta/stats.json", repo_type="dataset")

df = pd.read_parquet(data_path)
tasks_df = pd.read_parquet(tasks_path)
episodes_df = pd.read_parquet(episodes_path)
with open(stats_path) as f:
    stats = json.load(f)

print(f"Loaded {len(df)} frames, {len(episodes_df)} episodes, {len(tasks_df)} tasks")
print(f"\nTasks: {tasks_df.to_dict()}")

# Map task_index to task text
task_map = dict(zip(tasks_df["task_index"], tasks_df["task"]))
df["task"] = df["task_index"].map(task_map)

# --- 1. Navigation prompt distribution ---
task_counts = df["task"].value_counts()
print("\nNavigation prompt distribution (frames):")
for label, count in task_counts.items():
    print(f"  {label}: {count} ({100*count/len(df):.1f}%)")

# Episode-level distribution
ep_tasks = df.groupby("episode_index")["task"].first().value_counts()
print("\nNavigation prompt distribution (episodes):")
for label, count in ep_tasks.items():
    print(f"  {label}: {count} ({100*count/len(episodes_df):.1f}%)")

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

labels_f = task_counts.index.tolist()
counts_f = task_counts.values
colors = plt.cm.Set2(np.linspace(0, 1, len(labels_f)))
bars = axes[0].barh(labels_f, counts_f, color=colors)
axes[0].set_xlabel("Frame count")
axes[0].set_title("Nav Prompt Distribution (Frames)")
for bar, count in zip(bars, counts_f):
    axes[0].text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                 str(count), va="center", fontsize=9)

labels_e = ep_tasks.index.tolist()
counts_e = ep_tasks.values
bars = axes[1].barh(labels_e, counts_e, color=colors[:len(labels_e)])
axes[1].set_xlabel("Episode count")
axes[1].set_title("Nav Prompt Distribution (Episodes)")
for bar, count in zip(bars, counts_e):
    axes[1].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                 str(count), va="center", fontsize=9)

plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/nav_distribution.png", dpi=150)
print("Saved nav_distribution.png")
plt.close()

# --- 2. Action distributions ---
actions = np.stack(df["action"].values)
accels = actions[:, 0]
curvs = actions[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(accels, bins=100, color="steelblue", edgecolor="none", alpha=0.8)
axes[0].set_xlabel("Acceleration (m/s²)")
axes[0].set_ylabel("Count")
axes[0].set_title(f"Acceleration\nμ={accels.mean():.4f}, σ={accels.std():.4f}\nrange=[{accels.min():.3f}, {accels.max():.3f}]")
axes[0].axvline(0, color="red", linestyle="--", alpha=0.5)

axes[1].hist(curvs, bins=100, color="coral", edgecolor="none", alpha=0.8)
axes[1].set_xlabel("Curvature (1/m)")
axes[1].set_ylabel("Count")
axes[1].set_title(f"Curvature\nμ={curvs.mean():.4f}, σ={curvs.std():.4f}\nrange=[{curvs.min():.4f}, {curvs.max():.4f}]")
axes[1].axvline(0, color="red", linestyle="--", alpha=0.5)

plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/action_distributions.png", dpi=150)
print("Saved action_distributions.png")
plt.close()

# --- 3. State distributions ---
states = np.stack(df["observation.state"].values)
speeds = states[:, 0]
heading_rates = states[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(speeds, bins=80, color="seagreen", edgecolor="none", alpha=0.8)
axes[0].set_xlabel("Speed (m/s)")
axes[0].set_ylabel("Count")
axes[0].set_title(f"Speed\nμ={speeds.mean():.2f} m/s, σ={speeds.std():.2f}\nrange=[{speeds.min():.2f}, {speeds.max():.2f}]")

axes[1].hist(heading_rates, bins=80, color="orchid", edgecolor="none", alpha=0.8)
axes[1].set_xlabel("Heading Rate (rad/s)")
axes[1].set_ylabel("Count")
axes[1].set_title(f"Heading Rate\nμ={heading_rates.mean():.4f}, σ={heading_rates.std():.4f}\nrange=[{heading_rates.min():.3f}, {heading_rates.max():.3f}]")
axes[1].axvline(0, color="red", linestyle="--", alpha=0.5)

plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/state_distributions.png", dpi=150)
print("Saved state_distributions.png")
plt.close()

# --- 4. Action trajectories by nav category ---
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
target_tasks = ["continue straight", "turn left", "turn right"]

for col, task in enumerate(target_tasks):
    task_eps = df[df["task"] == task]["episode_index"].unique()
    if len(task_eps) == 0:
        continue
    ep = task_eps[0]
    ep_df = df[df["episode_index"] == ep].sort_values("frame_index")
    ep_actions = np.stack(ep_df["action"].values)
    t = np.arange(len(ep_actions)) * 0.1

    axes[0][col].plot(t, ep_actions[:, 0], color="steelblue", linewidth=1.5)
    axes[0][col].set_ylabel("Acceleration (m/s²)")
    axes[0][col].set_title(f'"{task}" (ep {ep}, {len(ep_actions)} frames)')
    axes[0][col].axhline(0, color="gray", linestyle="--", alpha=0.3)
    axes[0][col].set_ylim(accels.min() * 1.1, accels.max() * 1.1)

    axes[1][col].plot(t, ep_actions[:, 1], color="coral", linewidth=1.5)
    axes[1][col].set_ylabel("Curvature (1/m)")
    axes[1][col].set_xlabel("Time (s)")
    axes[1][col].axhline(0, color="gray", linestyle="--", alpha=0.3)
    axes[1][col].set_ylim(curvs.min() * 1.1, curvs.max() * 1.1)

plt.suptitle("Action Trajectories by Navigation Category", fontsize=14)
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/action_trajectories.png", dpi=150)
print("Saved action_trajectories.png")
plt.close()

# --- 5. Actions conditioned on nav category (box plots) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

tasks_list = df["task"].unique()
accel_by_task = [actions[df["task"].values == t, 0] for t in tasks_list]
curv_by_task = [actions[df["task"].values == t, 1] for t in tasks_list]

bp1 = axes[0].boxplot(accel_by_task, labels=tasks_list, vert=True, patch_artist=True)
for patch, color in zip(bp1["boxes"], plt.cm.Set2(np.linspace(0, 1, len(tasks_list)))):
    patch.set_facecolor(color)
axes[0].set_ylabel("Acceleration (m/s²)")
axes[0].set_title("Acceleration by Nav Category")
axes[0].tick_params(axis="x", rotation=30)

bp2 = axes[1].boxplot(curv_by_task, labels=tasks_list, vert=True, patch_artist=True)
for patch, color in zip(bp2["boxes"], plt.cm.Set2(np.linspace(0, 1, len(tasks_list)))):
    patch.set_facecolor(color)
axes[1].set_ylabel("Curvature (1/m)")
axes[1].set_title("Curvature by Nav Category")
axes[1].tick_params(axis="x", rotation=30)

plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/action_by_nav.png", dpi=150)
print("Saved action_by_nav.png")
plt.close()

# --- 6. Episode length distribution ---
ep_lengths = df.groupby("episode_index").size()
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(ep_lengths, bins=30, color="teal", edgecolor="none", alpha=0.8)
ax.set_xlabel("Frames per episode")
ax.set_ylabel("Count")
ax.set_title(f"Episode Length Distribution\nμ={ep_lengths.mean():.1f}, range=[{ep_lengths.min()}, {ep_lengths.max()}]")
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/episode_lengths.png", dpi=150)
print("Saved episode_lengths.png")
plt.close()

# --- 7. Try to extract sample frames from video ---
try:
    video_path = hf_hub_download(
        REPO, "videos/observation.images.front/chunk-000/file-000.mp4",
        repo_type="dataset"
    )
    import cv2
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\nVideo: {total_frames} frames")

    # Sample frames from different episodes/nav categories
    sample_info = []
    for task in ["continue straight", "turn left", "turn right", "change lanes left",
                 "change lanes right", "stop", "u-turn"]:
        task_rows = df[df["task"] == task]
        if len(task_rows) == 0:
            continue
        # Get first frame of first episode with this task
        first_ep = task_rows["episode_index"].iloc[0]
        first_frame = task_rows[task_rows["episode_index"] == first_ep].iloc[0]
        sample_info.append({
            "task": task,
            "index": int(first_frame["index"]),
            "speed": float(first_frame["observation.state"][0]),
            "accel": float(first_frame["action"][0]),
            "curv": float(first_frame["action"][1]),
        })

    n = len(sample_info)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for i, info in enumerate(sample_info):
        r, c = divmod(i, cols)
        cap.set(cv2.CAP_PROP_POS_FRAMES, info["index"])
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            axes[r][c].imshow(frame_rgb)
        axes[r][c].set_title(
            f'"{info["task"]}"\nspd={info["speed"]:.1f} m/s  a={info["accel"]:.2f}  κ={info["curv"]:.4f}',
            fontsize=9
        )
        axes[r][c].axis("off")

    for i in range(len(sample_info), rows * cols):
        r, c = divmod(i, cols)
        axes[r][c].axis("off")

    plt.suptitle("Sample Frames by Navigation Category", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/sample_frames.png", dpi=150, bbox_inches="tight")
    print("Saved sample_frames.png")
    plt.close()
    cap.release()

except Exception as e:
    print(f"\nCouldn't extract video frames: {e}")
    print("(Install opencv-python: pip install opencv-python)")

# --- Summary ---
print(f"\n{'='*50}")
print(f"DATASET SUMMARY")
print(f"{'='*50}")
print(f"Total frames:    {len(df)}")
print(f"Total episodes:  {len(episodes_df)}")
print(f"Total tasks:     {len(tasks_df)}")
print(f"FPS:             10 Hz")
print(f"Action dim:      2 (acceleration, curvature)")
print(f"State dim:       2 (speed, heading_rate)")
print(f"Image:           480×640×3")
print(f"Frames/episode:  {ep_lengths.mean():.0f} avg ({ep_lengths.min()}-{ep_lengths.max()})")
print(f"\nAction stats:")
print(f"  Acceleration:  μ={accels.mean():.4f}, σ={accels.std():.4f}, [{accels.min():.3f}, {accels.max():.3f}]")
print(f"  Curvature:     μ={curvs.mean():.5f}, σ={curvs.std():.5f}, [{curvs.min():.5f}, {curvs.max():.5f}]")
print(f"\nState stats:")
print(f"  Speed:         μ={speeds.mean():.2f}, σ={speeds.std():.2f}, [{speeds.min():.2f}, {speeds.max():.2f}] m/s")
print(f"  Heading rate:  μ={heading_rates.mean():.4f}, σ={heading_rates.std():.4f}, [{heading_rates.min():.4f}, {heading_rates.max():.4f}] rad/s")
print(f"\nAll plots saved to {OUTPUT_DIR}/")
