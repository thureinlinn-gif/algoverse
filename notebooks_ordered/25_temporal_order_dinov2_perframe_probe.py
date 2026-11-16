"""
Order-aware DINOv2 baseline: does DINOv2's per-frame features encode clip
direction, given a pooling that can actually see order? (V-JEPA2's raw-
embedding probe -- script 14 -- gets ~70% at this same task; mean-pooled
DINOv2 cannot even be asked the question, since the mean discards time.)

Per mentor feedback (2026-09-08): DINOv2 encodes each frame independently, so
a reversed clip's per-frame stack is EXACTLY the forward stack flipped along
the time axis -- torch.flip(x, [0]). No reversed-video re-extraction needed;
script 22's forward-only per-frame stacks are sufficient. This script is CPU
only, no GPU, and runs in well under a minute once script 22 has written the
validation segments' (T, 2048) stacks.

Two pools are compared, both fed the same forward/reversed pair per segment:

  * MEAN pool (the control). mean(F) == mean(flip(F)) EXACTLY -- averaging is
    invariant to the order of its terms, full stop. So this is not something
    that CAN be found to work; the linear probe fed this pool must land at
    EXACTLY 50.00% test accuracy, because every segment contributes one
    forward example and one reversed example with the identical feature
    vector, so whatever decision boundary the probe learns necessarily
    assigns them the same label. If this prints anything else, something
    upstream is broken (e.g. forward and reversed were not actually paired
    correctly) -- treat that as a critical bug, not a finding.

  * HALVES pool (the cheapest order-aware pooling, per REPRODUCE.md Phase 6):
    concat(mean(first half of T), mean(second half of T)), a 4096-dim vector.
    Reversing the full sequence swaps which original half lands in which
    position (mean(first half of flip(F)) == mean(second half of F), and
    vice versa) without changing either half's own mean -- so forward and
    reversed produce [A,B] vs [B,A]. A linear probe can trivially separate
    those whenever A != B on average across the corpus, so this pool CAN
    show above-chance accuracy if DINOv2's per-frame features carry any
    coarse forward/backward asymmetry -- exactly the question Phase 6 asks.

Reads : embeddings/<...>/dinov2_perframe/<segment_id>.pt (script 22's output,
        validation segments only)
        annotations/segments_verbs_timed_clean.csv
Writes: processed/nb8_temporal_order_dinov2_perframe_probe.csv
"""
import os, json, random
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F

with open('/home/ubuntu/youcook2-videos/paths.json') as f:
    PATHS = json.load(f)
OUT_DIR = PATHS['processed']
PERFRAME_DIR = os.path.join(PATHS['embeddings'], 'dinov2_perframe')
DEVICE = torch.device('cpu')   # deliberately CPU -- no forward passes, just pooling + a linear probe
SEED = 0

df = pd.read_csv(os.path.join(PATHS['annotations'], 'segments_verbs_timed_clean.csv'))
df_val = df[df['subset'] == 'validation'].reset_index(drop=True)

available = {f[:-3] for f in os.listdir(PERFRAME_DIR) if f.endswith('.pt')}
df_val = df_val[df_val['segment_id'].isin(available)].reset_index(drop=True)
print(f"Validation segments with per-frame DINOv2 features on disk: "
      f"{len(df_val):,} / {len(df[df['subset'] == 'validation']):,}")
if len(df_val) < len(df[df['subset'] == 'validation']):
    print("(script 22 may still be running -- this run covers whatever is on disk now)")

# ── Load per-frame stacks, build forward/reversed pairs for both pools ──────
mean_fwd_list, mean_rev_list = [], []
halves_fwd_list, halves_rev_list = [], []
seg_ids, video_ids = [], []
max_mean_gap = 0.0

for _, row in df_val.iterrows():
    stack = torch.load(os.path.join(PERFRAME_DIR, f"{row['segment_id']}.pt"), map_location='cpu')
    T = stack.shape[0]
    rev = torch.flip(stack, [0])

    mean_fwd = stack.mean(dim=0)
    mean_rev = rev.mean(dim=0)
    gap = (mean_fwd - mean_rev).abs().max().item()
    max_mean_gap = max(max_mean_gap, gap)

    half = T // 2
    if half == 0:  # degenerate single-frame segment -- both halves collapse to the same frame
        half = 1 if T > 1 else T
    halves_fwd = torch.cat([stack[:half].mean(dim=0), stack[half:].mean(dim=0)]) \
        if T > half else torch.cat([stack.mean(dim=0), stack.mean(dim=0)])
    halves_rev = torch.cat([rev[:half].mean(dim=0), rev[half:].mean(dim=0)]) \
        if T > half else torch.cat([rev.mean(dim=0), rev.mean(dim=0)])

    mean_fwd_list.append(mean_fwd.numpy()); mean_rev_list.append(mean_rev.numpy())
    halves_fwd_list.append(halves_fwd.numpy()); halves_rev_list.append(halves_rev.numpy())
    seg_ids.append(row['segment_id']); video_ids.append(row['video_id'])

print(f"\nControl check: max |mean(forward) - mean(reversed)| across "
      f"{len(seg_ids)} segments = {max_mean_gap:.2e}")
assert max_mean_gap < 1e-4, (
    "mean(forward) and mean(reversed) are NOT identical -- forward/reversed pairing is "
    "broken upstream (e.g. torch.flip applied to the wrong axis, or mismatched segments). "
    "This is a critical bug, not a finding. Stopping before the probe would report a "
    "meaningless number.")
print("Confirmed: mean-pool destroys order exactly as expected, forward/reversed pairing "
      "is correct. The MEAN-pool probe below is therefore guaranteed to land at exactly "
      "50.00% -- that is the control passing, not a result to interpret.")


def video_split(video_ids, seed):
    random.seed(seed)
    uniq = sorted(set(video_ids))
    random.shuffle(uniq)
    vsplit = int(0.8 * len(uniq))
    train_v, test_v = set(uniq[:vsplit]), set(uniq[vsplit:])
    train_idx = [i for i, v in enumerate(video_ids) if v in train_v]
    test_idx = [i for i, v in enumerate(video_ids) if v in test_v]
    return train_idx, test_idx, len(train_v), len(test_v)


train_idx, test_idx, n_train_v, n_test_v = video_split(video_ids, SEED)
print(f"\nVideo-level split: {n_train_v} train videos / {n_test_v} test videos "
      f"({len(train_idx)} / {len(test_idx)} segments)")


def build_xy(fwd_list, rev_list, idx):
    X, y = [], []
    for i in idx:
        X.append(fwd_list[i]); y.append(0)
        X.append(rev_list[i]); y.append(1)
    return np.stack(X).astype(np.float32), np.array(y, dtype=np.float32)


def train_probe(fwd_list, rev_list, label):
    X_train, y_train = build_xy(fwd_list, rev_list, train_idx)
    X_test, y_test = build_xy(fwd_list, rev_list, test_idx)

    torch.manual_seed(SEED)
    X_train_t = torch.tensor(X_train); y_train_t = torch.tensor(y_train)
    X_test_t = torch.tensor(X_test); y_test_t = torch.tensor(y_test)

    probe = nn.Linear(X_train.shape[1], 1)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-2)
    for epoch in range(200):
        probe.train()
        opt.zero_grad()
        logits = probe(X_train_t).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y_train_t)
        loss.backward(); opt.step()

    probe.eval()
    with torch.no_grad():
        train_acc = ((torch.sigmoid(probe(X_train_t).squeeze(-1)) > 0.5).float() == y_train_t).float().mean().item()
        test_acc = ((torch.sigmoid(probe(X_test_t).squeeze(-1)) > 0.5).float() == y_test_t).float().mean().item()

    print(f"\n=== {label} pool: linear probe, forward vs. reversed ===")
    print(f"Train accuracy: {train_acc*100:.2f}%")
    print(f"Test accuracy:  {test_acc*100:.2f}%  (chance = 50.00%)")
    return train_acc, test_acc


mean_train_acc, mean_test_acc = train_probe(mean_fwd_list, mean_rev_list, 'MEAN (control)')
halves_train_acc, halves_test_acc = train_probe(halves_fwd_list, halves_rev_list, 'HALVES (order-aware)')

print("\n" + "=" * 78)
print("SUMMARY")
print("=" * 78)
print(f"MEAN pool   (control)     : test acc = {mean_test_acc*100:.2f}%  (must be ~50.00%)")
print(f"HALVES pool (order-aware) : test acc = {halves_test_acc*100:.2f}%  (chance = 50.00%)")
print(f"V-JEPA2 raw-embedding probe (script 14, pilot n=600): 69.17% -- for comparison")
print(f"V-JEPA2 probe, full scale (script 07b): 70.19% -- for comparison")
print()
if abs(mean_test_acc - 0.5) > 0.03:
    print("!! MEAN-pool control did NOT land near 50% -- investigate before trusting the "
          "HALVES-pool number above.")
elif halves_test_acc > 0.60:
    print("HALVES pool clears chance by a wide margin -- DINOv2's per-frame features DO carry "
          "forward/backward asymmetry once the pooling can see it. This narrows the claim to "
          "what REPRODUCE.md already concedes: the published mean-pool DINOv2 baseline was "
          "built to lose, not that DINOv2's features structurally cannot encode order.")
else:
    print("HALVES pool stays near chance -- the order-sensitivity gap looks like a genuine "
          "property of what the two encoders represent, not an artifact of pooling choice. "
          "This STRENGTHENS the V-JEPA2 claim.")

pd.DataFrame([
    {'pool': 'mean_control', 'n_segments': len(seg_ids), 'train_acc': mean_train_acc, 'test_acc': mean_test_acc},
    {'pool': 'halves_order_aware', 'n_segments': len(seg_ids), 'train_acc': halves_train_acc, 'test_acc': halves_test_acc},
]).to_csv(os.path.join(OUT_DIR, 'nb8_temporal_order_dinov2_perframe_probe.csv'), index=False)
print("\nSaved -> nb8_temporal_order_dinov2_perframe_probe.csv")
print("DONE")
