"""
Phase 7, shared step: cache each DinoTXT text tower's features for the 282 bare
verbs *before* `text_model.head.linear_projection`, so every trained head in
Phase 7 becomes one matmul with no text-tower forward.

Why this is exact, not an approximation: in `TextHead.forward` the projection is
the last op (`return self.linear_projection(text_tokens)`), and `TextTower.forward`
then takes the first token. Selecting a token commutes with a linear map, so
    encode_text(tokens) == text_model(tokens with projection=Identity) @ W.T
for any W -- the published W, or any head trained by scripts 07/11/13/27, which
only ever replace that one Linear. `logit_scale` is not in the DINOv2 state dicts
(it stays at its init, 1/0.07) and is frozen in every training script, so it is
cached here too.

Verified on every run (aborts otherwise): the cached features reproduce
encode_text(normalize=True) to 1e-5 through the original projection AND through
two trained head checkpoints (DINOv2 out_dim=2048, V-JEPA2 out_dim=1024). TF32 is
deliberately left off so the check compares fp32 against fp32.

DINOv3 is cached only if its two weight files are present under
/home/ubuntu/weights (same paths as scripts 07/21); otherwise it is skipped, and
rerunning this script after restoring them adds it without touching DINOv2's file.

Reads : repos/dinov2 (+ repos/dinov3 if weights present), checkpoints/nb8_dinov2_seed0_bestepoch9.pt,
        checkpoints/nb8_vjepa2_pretrainedpool_seed1_bestepoch9.pt
Writes: processed/nb8_text_features_dinov2_dinotxt.pt
        processed/nb8_text_features_dinov3_dinotxt.pt   (only if DINOv3 weights present)
        Each is a dict: verbs (list[282]), features (282,1280) fp32, orig_projection (W),
        logit_scale (float), max_abs_err (float).
Seconds on GPU (plus a one-time ~1.5 GB DinoTXT download on a fresh box).
"""
import os, sys, json
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F

with open('/home/ubuntu/youcook2-videos/paths.json') as f:
    PATHS = json.load(f)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cuda.matmul.allow_tf32 = False   # fp32 vs fp32 for the 1e-5 check
torch.backends.cudnn.allow_tf32 = False

OUT_DIR, CKPT_DIR = PATHS['processed'], PATHS['checkpoints']
TOL = 1e-5

DINOV3_REPO = os.path.join(PATHS['repos'], 'dinov3')
DINOV3_BACKBONE_WEIGHTS = '/home/ubuntu/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
DINOV3_DINOTXT_WEIGHTS = '/home/ubuntu/weights/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth'

sys.path.insert(0, os.path.join(PATHS['repos'], 'dinov2'))
from dinov2.hub.dinotxt import dinov2_vitl14_reg4_dinotxt_tet1280d20h24l, get_tokenizer

df = pd.read_csv(os.path.join(PATHS['annotations'], 'segments_verbs_timed_clean.csv'))
verb_vocab = sorted(df['extracted_verb'].unique().tolist())
assert len(verb_vocab) == 282, f"Expected 282 verbs, got {len(verb_vocab)}"


@torch.no_grad()
def pre_projection_features(model, tokens):
    orig = model.text_model.head.linear_projection
    model.text_model.head.linear_projection = nn.Identity()
    feats = model.text_model(tokens)
    model.text_model.head.linear_projection = orig
    return feats


@torch.no_grad()
def check(model, tokens, feats, proj, label):
    """encode_text through `proj` vs. normalize(feats @ W.T); returns max abs err."""
    orig = model.text_model.head.linear_projection
    model.text_model.head.linear_projection = proj
    ref = model.encode_text(tokens, normalize=True)
    model.text_model.head.linear_projection = orig
    recon = F.normalize(feats @ proj.weight.T, dim=-1)
    err = (recon - ref).abs().max().item()
    status = 'OK' if err < TOL else 'FAIL'
    print(f"  [{status}] {label:<44} max |diff| = {err:.2e}")
    assert err < TOL, f"{label}: cached features do not reproduce encode_text ({err:.2e} >= {TOL})"
    return err


def trained_head(ckpt_name, out_dim):
    proj = nn.Linear(1280, out_dim, bias=False).to(DEVICE)
    state = torch.load(os.path.join(CKPT_DIR, ckpt_name), map_location=DEVICE)
    proj.load_state_dict(state)
    return proj


def cache(model, tok, name, trained_checks):
    print(f"\n=== {name} ===")
    model = model.to(DEVICE).eval()
    tokens = tok.tokenize(verb_vocab).to(DEVICE)
    feats = pre_projection_features(model, tokens)
    print(f"  features: {tuple(feats.shape)}  logit_scale.exp() = {model.logit_scale.exp().item():.4f}")
    orig = model.text_model.head.linear_projection
    errs = [check(model, tokens, feats, orig, 'original projection')]
    for ckpt_name, out_dim in trained_checks:
        errs.append(check(model, tokens, feats, trained_head(ckpt_name, out_dim), ckpt_name))
    out = {'verbs': verb_vocab, 'features': feats.cpu(), 'orig_projection': orig.weight.detach().cpu(),
           'logit_scale': model.logit_scale.exp().item(), 'max_abs_err': max(errs)}
    path = os.path.join(OUT_DIR, f'nb8_text_features_{name}.pt')
    torch.save(out, path)
    print(f"  Saved -> {path}")


cache(dinov2_vitl14_reg4_dinotxt_tet1280d20h24l(), get_tokenizer(), 'dinov2_dinotxt',
      trained_checks=[('nb8_dinov2_seed0_bestepoch9.pt', 2048),
                      ('nb8_vjepa2_pretrainedpool_seed1_bestepoch9.pt', 1024)])

if os.path.exists(DINOV3_BACKBONE_WEIGHTS) and os.path.exists(DINOV3_DINOTXT_WEIGHTS):
    m3, tok3 = torch.hub.load(DINOV3_REPO, 'dinov3_vitl16_dinotxt_tet1280d20h24l', source='local',
                              weights=DINOV3_DINOTXT_WEIGHTS, backbone_weights=DINOV3_BACKBONE_WEIGHTS)
    cache(m3, tok3, 'dinov3_dinotxt', trained_checks=[('nb8_dinov3_seed0_bestepoch9.pt', 2048)])
else:
    print("\nDINOv3 SKIPPED -- weights not under /home/ubuntu/weights. Restore them and rerun "
          "this script to add nb8_text_features_dinov3_dinotxt.pt.")

print("DONE")
