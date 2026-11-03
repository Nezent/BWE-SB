"""Load a task config and turn it into the experiment dict the trainer consumes."""
import json
from pathlib import Path


def load(path, tier="short", overrides=None):
    cfg = json.loads(Path(path).read_text())
    if tier not in cfg["tiers"]:
        raise KeyError(f"tier {tier!r} not in {sorted(cfg['tiers'])}")
    cfg["tier"] = tier
    cfg["active"] = dict(cfg["tiers"][tier])
    for k, v in (overrides or {}).items():
        _set(cfg, k, v)
    return cfg


def _set(cfg, dotted, value):
    node = cfg
    keys = dotted.split(".")
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = _coerce(value)


def _coerce(v):
    if isinstance(v, str):
        low = v.lower()
        if low in ("true", "false"):
            return low == "true"
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            pass
        if v.startswith("["):
            return json.loads(v)
    return v


def model_hparams(cfg):
    t, s, o = cfg["task"], cfg["stft"], cfg["optim"]
    h = dict(batch_size=cfg["active"].get("batch_size", o["batch_size"]), segment_size=o["segment_size"],
             learning_rate=o["learning_rate"], adam_b1=o["adam_b1"], adam_b2=o["adam_b2"], lr_decay=o["lr_decay"],
             seed=o["seed"], num_workers=o["num_workers"], n_fft=s["n_fft"], hop_size=s["hop_size"], win_size=s["win_size"],
             hr_sampling_rate=t["hr_sampling_rate"], lr_sampling_rate=t["lr_sampling_rate"], le_window_sizes=[128, 256, 512])
    h.update(cfg["model"])
    h["gen_cond_dim"] = cfg["degradation"].get("cond_dim", 0)
    return h


def build_exp(cfg, name, out_dir, wav_dir, split_dir, nisqa_dir=None):
    a, l, b, d = cfg["active"], cfg["loss"], cfg["bridge"], cfg["degradation"]
    split_dir = Path(split_dir)
    return dict(name=name, generator="cnx", critics=list(cfg["critics"]), hpc_weight=l["hpc"], cfg=model_hparams(cfg),
                epochs=a["epochs"], seed=cfg["optim"]["seed"], max_steps=a["max_steps"] or None, val_every=a["val_every"],
                val_files=a["val_files"], log_every=25, train_bank=d["train_bank"], eval_banks=list(d["eval_banks"]),
                desc_weight=l["desc"], eval_max_recipes=a["eval_max_recipes"], sb_steps=b["sb_steps"], sb_sigma=b["sb_sigma"],
                sb_t_max=b["t_max"], sb_x0_prior=b["x0_prior"], backbone="selssm", precond=b["precond"],
                bpd_weight=l["bpd"], sc_weight=l["sc"], be_weight=l["be"], lm_weight=l["lm"], amp_loss_hf_only=l["amp_loss_hf_only"],
                adv_anchor_weight=l["adv_anchor"], adv_aux_weight=l["adv_aux"], patience=a["patience"],
                nisqa_dir=str(nisqa_dir) if nisqa_dir else None, ckpt_every=a["ckpt_every"], resume=True,
                stats_files=a["stats_files"], minutes_budget=0, out_dir=str(out_dir), wavs_dir=str(wav_dir),
                train_list=str(split_dir / "training.txt"), test_list=str(split_dir / "test.txt"),
                val_list=str(split_dir / "validation.txt"))


def run_tag(cfg, n_steps):
    return f"{cfg['tier']}-{cfg['task']['name']}-{n_steps}steps"
