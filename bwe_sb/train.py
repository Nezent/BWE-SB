from __future__ import annotations
"""Training loop: degradation-aware bridge training, validation on held-out training speakers, best-checkpoint selection, NISQA scoring."""

import argparse, json, random, re, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

def main(exp=None):
    if exp is None:
        ap = argparse.ArgumentParser(); ap.add_argument("--exp", required=True)
        exp = json.loads(Path(ap.parse_args().exp).read_text())


    torch.manual_seed(exp["seed"]); np.random.seed(exp["seed"]); random.seed(exp["seed"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(exp["out_dir"]); out.mkdir(parents=True, exist_ok=True)

    from bwe_sb.data.dataset import Dataset, amp_pha_stft, amp_pha_istft
    from bwe_sb.models.critics.apbwe import phase_losses, feature_loss, discriminator_loss, generator_loss
    import bwe_sb.models.critics.registry as REG
    import bwe_sb.data.degrade as DG
    import bwe_sb.metrics.metrics as MT
    import bwe_sb.models.bridge as SB
    from bwe_sb.models.generator import baseband_phase_diff
    from bwe_sb.losses.phase import antiwrap, frame_f0, hpc_loss
    import bwe_sb.losses.amplitude as LA

    class H(dict):
        __getattr__ = dict.get
    h = H(exp["cfg"])
    DG.BASE = int(h.lr_sampling_rate)
    sr, f_cut = h.hr_sampling_rate, h.lr_sampling_rate / 2
    n_bins = h.n_fft // 2 + 1

    from bwe_sb.models.generator import DualStreamBWE
    G = DualStreamBWE(h).to(dev)
    conditioned = bool(h.gen_cond_dim)
    n_g = sum(p.numel() for p in G.parameters())

    enc = DG.DegradationEncoder(n_bins, emb=int(h.gen_cond_dim or 64)).to(dev) \
        if conditioned else None
    n_e = sum(p.numel() for p in enc.parameters()) if enc else 0
    w_desc = float(exp.get("desc_weight", 10.0)) if enc else 0.0

    sb_steps = int(exp.get("sb_steps", 0) or 0)
    sched = SB.SBSchedule(steps=max(1, sb_steps),
                          sigma=float(exp.get("sb_sigma", 0.05) or 0.0),
                          t_max=float(exp.get("sb_t_max", 1.0) or 1.0))
    use_sb = bool(getattr(h, "gen_sb", False)) and sb_steps > 0
    sb_x0_prior = bool(exp.get("sb_x0_prior", True))
    use_precond = bool(exp.get("precond", False)) and use_sb
    precond_sd = float(exp.get("precond_sigma_d", 1.0) or 1.0)
    lam_t = None
    w_bpd = float(exp.get("bpd_weight", 0.0) or 0.0)
    n_bins_g = h.n_fft // 2 + 1

    def cut_bin_from_desc(d):
        if d is None:
            return None
        hz = (d[:, 0].clamp(0, 1) * (sr / 2.0)) / 2.0
        return (hz / (sr / 2) * (n_bins_g - 1)).round().long().clamp(4, n_bins_g - 1)

    Ds = {n: REG.build(n, h).to(dev) for n in exp["critics"]}
    n_d = {n: sum(p.numel() for p in d.parameters()) for n, d in Ds.items()}
    d_params = [p for d in Ds.values() for p in d.parameters()]

    g_params = list(G.parameters()) + (list(enc.parameters()) if enc else [])
    optim_g = torch.optim.AdamW(g_params, h.learning_rate,
                                betas=[h.adam_b1, h.adam_b2], weight_decay=0.01)
    optim_d = (torch.optim.AdamW(d_params, h.learning_rate,
                                 betas=[h.adam_b1, h.adam_b2], weight_decay=0.01)
               if d_params else None)
    sch_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay)
    sch_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay) if optim_d else None

    tr_ids = Path(exp["train_list"]).read_text().split()
    te_ids = Path(exp["test_list"]).read_text().split()
    if exp.get("val_list") and Path(exp["val_list"]).exists():
        va_ids = Path(exp["val_list"]).read_text().split()
    else:
        _n = max(20, min(200, len(tr_ids) // 20))
        va_ids = tr_ids[::max(1, len(tr_ids) // _n)][:_n]
    _va = set(va_ids)
    tr_ids = [x for x in tr_ids if x not in _va]
    assert not (set(va_ids) & set(te_ids)), "validation list overlaps the official test list"
    print(f"  split: train {len(tr_ids)} | validation {len(va_ids)} (training speakers, "
          f"used for checkpoint selection) | test {len(te_ids)} (final evaluation only)",
          flush=True)
    mk = lambda ids, split: Dataset(ids, exp["wavs_dir"], h.segment_size, sr,
                                    h.lr_sampling_rate, split=split, shuffle=split,
                                    n_cache_reuse=0)
    dl = DataLoader(mk(tr_ids, True), batch_size=h.batch_size, shuffle=True,
                    num_workers=h.num_workers, pin_memory=True, drop_last=True)
    vl_val = DataLoader(mk(va_ids, False), batch_size=1, shuffle=False, num_workers=0)
    vl_test = DataLoader(mk(te_ids, False), batch_size=1, shuffle=False, num_workers=0)

    train_recipes = DG.bank(exp.get("train_bank", "sinc"))
    eval_banks = exp.get("eval_banks", ["sinc"])
    w_hpc = float(exp.get("hpc_weight", 0.0))
    w_sc = float(exp.get("sc_weight", 10.0))
    w_be = float(exp.get("be_weight", 1.0))
    w_lm = float(exp.get("lm_weight", 5.0))
    amp_hf_only = bool(exp.get("amp_loss_hf_only", True))
    _cb = int(round(f_cut / (sr / 2) * (n_bins - 1)))
    _fw = torch.ones(n_bins, device=dev)
    _fw[_cb:] = torch.linspace(1.0, 3.0, n_bins - _cb, device=dev)
    _fw = (_fw / _fw.mean()).view(1, -1, 1)
    w_anchor = float(exp.get("adv_anchor_weight", 1.0))
    w_aux = float(exp.get("adv_aux_weight", 0.1))
    steps_per_epoch = max(1, len(tr_ids) // h.batch_size)
    stop_at = exp.get("max_steps") or exp["epochs"] * steps_per_epoch

    curves = out / "curves.csv"
    if not curves.exists():
        curves.write_text("step,loss_g,loss_d,mag,pha,com,stft,fm,adv,hpc,desc,bpd,sc,be,lm\n")

    CKPT = out / "ckpt.pt"
    start_step = 0

    def save_ckpt(step):
        tmp = out / "ckpt.pt.tmp"
        torch.save(dict(G=G.state_dict(), enc=enc.state_dict() if enc else None,
                        Ds={n: d.state_dict() for n, d in Ds.items()},
                        optim_g=optim_g.state_dict(),
                        optim_d=optim_d.state_dict() if optim_d else None,
                        sch_g=sch_g.state_dict(),
                        sch_d=sch_d.state_dict() if sch_d else None,
                        step=step), tmp)
        tmp.replace(CKPT)

    BEST = out / "best.pt"
    best_state = {"metric": float("inf"), "step": -1}
    SEL_METRIC = str(exp.get("select_metric", "lsd_hf"))
    SEL_BANK = str(exp.get("select_bank", "") or (exp.get("eval_banks") or ["sinc"])[0])

    def save_best(step, summary):
        """Keep the weights with the lowest validation LSD-HF, not the last ones."""
        v = (summary.get(SEL_BANK) or {}).get(SEL_METRIC)
        if v is None or not np.isfinite(v):
            return
        if v < best_state["metric"]:
            best_state.update(metric=float(v), step=int(step))
            tmp = out / "best.pt.tmp"
            torch.save(dict(G=G.state_dict(), enc=enc.state_dict() if enc else None,
                            step=int(step), metric=float(v),
                            select=f"{SEL_METRIC}@{SEL_BANK}"), tmp)
            tmp.replace(BEST)
            print(f"  best so far: {SEL_METRIC}@{SEL_BANK} {v:.4f} at step {step}",
                  flush=True)

    if BEST.exists() and exp.get("resume", True):
        try:
            _pb = torch.load(BEST, map_location="cpu", weights_only=False)
            best_state.update(metric=float(_pb["metric"]), step=int(_pb["step"]))
            print(f"carried over best {_pb['select']} {_pb['metric']:.4f} "
                  f"from step {_pb['step']}", flush=True)
        except Exception as e:
            print(f"  !! could not read best.pt ({type(e).__name__}); starting fresh")

    if CKPT.exists() and exp.get("resume", True):
        try:
            ck = torch.load(CKPT, map_location=dev, weights_only=False)
            G.load_state_dict(ck["G"])
            if enc and ck.get("enc"):
                enc.load_state_dict(ck["enc"])
            for n, d in Ds.items():
                if n in ck.get("Ds", {}):
                    d.load_state_dict(ck["Ds"][n])
            if ck.get("optim_g"):
                optim_g.load_state_dict(ck["optim_g"])
            if optim_d and ck.get("optim_d"):
                optim_d.load_state_dict(ck["optim_d"])
            if ck.get("sch_g"):
                sch_g.load_state_dict(ck["sch_g"])
            if sch_d and ck.get("sch_d"):
                sch_d.load_state_dict(ck["sch_d"])
            start_step = int(ck.get("step", 0))
            print(f"  resumed {exp['name']} from step {start_step}", flush=True)
        except Exception as e:
            print(f"  !! checkpoint unreadable ({e}); starting from scratch", flush=True)
            start_step = 0

    import inspect
    _gen_params = set(inspect.signature(G.forward).parameters)

    def _gkw(**kw):
        return {k: v for k, v in kw.items() if k in _gen_params and v is not None}

    if use_sb and "t" not in _gen_params:
        raise ValueError(
            f"sb_steps={sb_steps} but generator {type(G).__name__} takes no `t`: the "
            f"bridge timestep would be silently ignored and the run would be mislabelled")

    def forward(mag_i, pha_i, t=None, cut_from=None, x0=None):
        """One generator call."""
        if enc is None:
            return (G(mag_i, pha_i, **_gkw(cut_bin=cut_from, t=t,
                                           x0_mag=(x0[0] if x0 else None),
                                           x0_pha=(x0[1] if x0 else None))), None)
        emb, desc_hat = enc(mag_i)
        cb = cut_from if cut_from is not None else cut_bin_from_desc(desc_hat)
        return (G(mag_i, pha_i, **_gkw(cond=emb, cut_bin=cb, t=t,
                                       x0_mag=(x0[0] if x0 else None),
                                       x0_pha=(x0[1] if x0 else None))), desc_hat)

    def predict(mag_i, pha_i):
        """Inference."""
        if not use_sb:
            out, _ = forward(mag_i, pha_i)
            return out[0], out[1]
        m0 = (G.coarse_prior(mag_i) if sb_x0_prior and hasattr(G, "coarse_prior")
              else mag_i)
        def _step(m_, p_, tv):
            if use_precond:
                m_ = m_ * SB.precond(tv, sched.sigma, precond_sd)[0]
            return forward(m_, p_, t=tv, x0=(m0, pha_i))[0]
        res = SB.sb_infer(_step, m0, pha_i, sched)
        return res[0], res[1]

    @torch.no_grad()
    def run_val(step, final=False):
        """Per-degradation evaluation."""
        G.eval()
        if enc: enc.eval()
        want_p = final or bool(exp.get("perceptual_every_val"))
        per_recipe, summary, utt_rows = {}, {}, []
        nisqa_dir = exp.get("nisqa_dir") if final else None
        nisqa_root = out / "nisqa_wavs"
        nisqa_paths = {}
        if nisqa_dir:
            import shutil as _sh
            _sh.rmtree(nisqa_root, ignore_errors=True)
        cap = int(exp.get("eval_max_recipes", 8) or 0)
        n_default = int(exp.get("val_files", 20))
        n_stats = int(exp.get("stats_files", 0) or 0)
        for bname in eval_banks:
            n_files = n_stats if (final and n_stats and bname == eval_banks[0]) else n_default
            recipes = DG.bank(bname)
            if cap and len(recipes) > cap and bname != "aurora" and not re.fullmatch(r"R\d", bname):
                keep = [r for r in recipes if set(r) == {"cutoff"}]
                rest = [r for r in recipes if set(r) != {"cutoff"}]
                n_rest = max(0, cap - len(keep))
                if rest and n_rest:
                    idx = np.linspace(0, len(rest) - 1, n_rest).round().astype(int)
                    rest = [rest[i] for i in sorted(set(idx.tolist()))]
                else:
                    rest = []
                recipes = keep + rest
            for ri, recipe in enumerate(recipes):
                acc = {}
                for i, (hr, _) in enumerate(vl_test if final else vl_val):
                    if i >= n_files:
                        break
                    hr = hr.to(dev)
                    lr, _ = DG.degrade(hr, sr, recipe)
                    mi, pi, _ = amp_pha_stft(lr, h.n_fft, h.hop_size, h.win_size)
                    mg, pg = predict(mi, pi)
                    est = amp_pha_istft(mg, pg, h.n_fft, h.hop_size, h.win_size)
                    k = min(est.shape[-1], hr.shape[-1])
                    m = MT.evaluate(est[..., :k], hr[..., :k], sr,
                                    recipe.get("cutoff", h.lr_sampling_rate) / 2,
                                    dev=dev, want_perceptual=want_p)
                    for kk, vv in m.items():
                        if isinstance(vv, float):
                            acc.setdefault(kk, []).append(vv)
                    m_in = MT.evaluate(lr[..., :k], hr[..., :k], sr,
                                       recipe.get("cutoff", h.lr_sampling_rate) / 2,
                                       dev=dev, want_perceptual=False)
                    for kk, vv in m_in.items():
                        if isinstance(vv, float):
                            acc.setdefault(f"in_{kk}", []).append(vv)
                    if nisqa_dir:
                        import soundfile as _sf
                        wdir = nisqa_root / bname; wdir.mkdir(parents=True, exist_ok=True)
                        wpath = wdir / f"r{ri}_u{i}.wav"
                        _sf.write(str(wpath), est[0, :k].detach().cpu().numpy(), sr)
                        nisqa_paths.setdefault(bname, []).append((wpath.name, ri, i))
                    if final:
                        row = dict(bank=bname, recipe_idx=ri, utt=i,
                                   rec_tier=str(recipe.get("tier", "")),
                                   rec_cutoff=recipe.get("cutoff", ""),
                                   rec_snr=recipe.get("snr", ""),
                                   rec_noise=recipe.get("noise", ""))
                        row.update({kk: vv for kk, vv in m.items()
                                    if isinstance(vv, float)})
                        row.update({f"in_{kk}": vv for kk, vv in m_in.items()
                                    if isinstance(vv, float)})
                        utt_rows.append(row)
                key = f"{bname}|" + "_".join(f"{a}={b}" for a, b in sorted(recipe.items()))
                per_recipe[key] = {kk: float(np.nanmean(vv)) for kk, vv in acc.items()}
            rows = [v for kk, v in per_recipe.items() if kk.startswith(bname + "|")]
            summary[bname] = {kk: float(np.nanmean([r[kk] for r in rows if kk in r]))
                              for kk in rows[0]} if rows else {}
        if nisqa_dir and nisqa_paths:
            import subprocess as _sp, csv as _csv
            for bname, items in nisqa_paths.items():
                wdir = nisqa_root / bname
                cmd = [sys.executable, "run_predict.py", "--mode", "predict_dir",
                       "--pretrained_model", str(Path(nisqa_dir) / "weights" / "nisqa.tar"),
                       "--data_dir", str(wdir), "--num_workers", "0", "--bs", "10",
                       "--output_dir", str(wdir)]
                r = _sp.run(cmd, cwd=nisqa_dir, capture_output=True, text=True)
                csvp = wdir / "NISQA_results.csv"
                if r.returncode != 0 or not csvp.exists():
                    raise RuntimeError(f"NISQA failed on bank {bname}: "
                                       f"{(r.stderr or r.stdout)[-600:]}")
                with csvp.open() as f:
                    scores = {Path(row["deg"]).name: float(row["mos_pred"])
                              for row in _csv.DictReader(f)}
                by_recipe = {}
                for name, ri, i in items:
                    v = scores.get(name)
                    if v is None:
                        continue
                    by_recipe.setdefault(ri, []).append(v)
                    for row in utt_rows:
                        if row["bank"] == bname and row["recipe_idx"] == ri and row["utt"] == i:
                            row["nisqa_mos"] = v
                keys = [kk for kk in per_recipe if kk.startswith(bname + "|")]
                for ri, vals in by_recipe.items():
                    per_recipe[keys[ri]]["nisqa_mos"] = float(np.mean(vals))
                allv = [v for vals in by_recipe.values() for v in vals]
                if allv:
                    summary[bname]["nisqa_mos"] = float(np.mean(allv))
            print(f"  NISQA-MOS: " + " | ".join(f"{b}={summary[b].get('nisqa_mos', float('nan')):.3f}"
                                               for b in summary), flush=True)
        G.train()
        if enc: enc.train()
        msg = " | ".join(f"{b}: lsd_hf={v.get('lsd_hf', float('nan')):.3f} "
                         f"si_sdr={v.get('si_sdr', float('nan')):.2f}"
                         for b, v in summary.items())
        print(f"  val@{step}  {msg}", flush=True)
        if final and utt_rows:
            try:
                import csv
                with (out / "per_utt.csv").open("w", newline="") as f:
                    cols = list(dict.fromkeys(k for r in utt_rows for k in r))
                    wtr = csv.DictWriter(f, fieldnames=cols, restval="")
                    wtr.writeheader()
                    wtr.writerows(utt_rows)
                print(f"  wrote {len(utt_rows)} per-utterance rows", flush=True)
            except Exception as e:
                print(f"  !! per_utt.csv failed ({type(e).__name__}: {e})", flush=True)
        return summary, per_recipe

    @torch.no_grad()
    def dump_samples(n=3, banks=None):
        """Save (narrowband input, prediction, reference) for a few held-out utterances so the notebook can draw AP-BW..."""
        G.eval()
        if enc: enc.eval()
        banks = banks or [b for b in ("sinc", "unseen", "R4") if b in eval_banks] \
            or eval_banks[:1]
        blob = {}
        for bn in banks:
            recipe = DG.bank(bn)[0]
            keep = []
            for i, (hr, _) in enumerate(vl_test):
                if i >= n:
                    break
                hr = hr.to(dev)
                lr, _ = DG.degrade(hr, sr, recipe)
                mi, pi, _ = amp_pha_stft(lr, h.n_fft, h.hop_size, h.win_size)
                mg, pg = predict(mi, pi)
                est = amp_pha_istft(mg, pg, h.n_fft, h.hop_size, h.win_size)
                kk = min(est.shape[-1], hr.shape[-1], lr.shape[-1])
                keep.append(dict(nb=lr[0, :kk].cpu(), est=est[0, :kk].cpu(),
                                 ref=hr[0, :kk].cpu()))
            blob[bn] = dict(recipe=recipe, items=keep)
        torch.save(dict(sr=sr, banks=blob, recipe=blob[banks[0]]["recipe"],
                        items=blob[banks[0]]["items"]), out / "samples.pt")
        G.train()
        if enc: enc.train()

    NONFINITE_MAX = int(exp.get("nonfinite_max", 25))
    nonfinite = 0
    PATIENCE = int(exp.get("patience", 4))
    stopped_early = False
    last_val_step = -1

    step, t0 = start_step, time.time()
    ckpt_every = int(exp.get("ckpt_every", 2000) or 0)
    budget_s = float(exp.get("minutes_budget", 0) or 0) * 60
    partial = False
    G.train()
    for epoch in range(start_step // steps_per_epoch, exp["epochs"]):
        for hr, _ in dl:
            if step >= stop_at:
                break
            hr = hr.to(dev, non_blocking=True)
            recipe = DG.sample_recipe(train_recipes)
            lr, desc = DG.degrade(hr, sr, recipe)
            mag_t, pha_t, com_t = amp_pha_stft(hr, h.n_fft, h.hop_size, h.win_size)
            mag_i, pha_i, _ = amp_pha_stft(lr, h.n_fft, h.hop_size, h.win_size)
            t_cur = None
            if use_sb:
                k_t = min(mag_i.shape[-1], mag_t.shape[-1])
                t_cur = sched.sample_t(mag_i.shape[0], dev)
                mag_x0 = (G.coarse_prior(mag_i) if sb_x0_prior and hasattr(G, "coarse_prior")
                          else mag_i)
                mag_in = SB.bridge_linear(mag_x0[..., :k_t], mag_t[..., :k_t], t_cur,
                                          sched.sigma)
                pha_in = SB.bridge_phase(pha_i[..., :k_t], pha_t[..., :k_t], t_cur,
                                         sched.sigma)
                if use_precond:
                    c_in, _, lam_t = SB.precond(t_cur, sched.sigma, precond_sd)
                    mag_in = mag_in * c_in
                x0_pair = (mag_x0[..., :k_t], pha_i[..., :k_t])
            else:
                mag_in, pha_in = mag_i, pha_i
                x0_pair = (mag_i, pha_i)
            (mag_g, pha_g, com_g), desc_hat = forward(mag_in, pha_in, t=t_cur, x0=x0_pair)
            _kf = min(mag_g.shape[-1], mag_t.shape[-1], com_t.shape[-2])
            mag_t, pha_t = mag_t[..., :_kf], pha_t[..., :_kf]
            com_t = com_t[..., :_kf, :]
            mag_g, pha_g = mag_g[..., :_kf], pha_g[..., :_kf]
            com_g = com_g[..., :_kf, :]
            audio_g = amp_pha_istft(mag_g, pha_g, h.n_fft, h.hop_size, h.win_size)
            k = min(audio_g.shape[-1], hr.shape[-1])
            yr, yf = hr[..., :k].unsqueeze(1), audio_g[..., :k].unsqueeze(1)
            nb = lr[..., :k].unsqueeze(1)

            def call(d, detach):
                f = yf.detach() if detach else yf
                if d.call_kind == "spec":
                    return d(pha_t, mag_t, pha_g.detach() if detach else pha_g,
                             mag_g.detach() if detach else mag_g, nb)
                return d(yr, f)

            loss_d = torch.zeros((), device=dev)
            if optim_d:
                optim_d.zero_grad(set_to_none=True)
                for n, d in Ds.items():
                    r, g, _, _ = call(d, True)
                    loss_d = loss_d + discriminator_loss(r, g)[0]
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(d_params, 10.0)
                optim_d.step()

            optim_g.zero_grad(set_to_none=True)
            _fwk = _fw[:, :mag_t.shape[1]]
            if use_precond and lam_t is not None:
                _w = lam_t / lam_t.mean().clamp(min=1e-6)
                l_mag = ((mag_t - mag_g) ** 2 * _w * _fwk).mean() * 45
            else:
                l_mag = ((mag_t - mag_g) ** 2 * _fwk).mean() * 45
            ip, gd, iaf = phase_losses(pha_t, pha_g)
            l_pha = (ip + gd + iaf) * 100
            l_com = F.mse_loss(com_t, com_g) * 90
            _, _, com_hat = amp_pha_stft(audio_g, h.n_fft, h.hop_size, h.win_size)
            t = min(com_hat.shape[-2], com_g.shape[-2])
            l_stft = F.mse_loss(com_g[..., :t, :], com_hat[..., :t, :]) * 90
            l_fm = torch.zeros((), device=dev); l_adv = torch.zeros((), device=dev)
            for n, d in Ds.items():
                r, g, fr, fg = call(d, False)
                sc = w_anchor if n in ("MPD", "MRAD", "MRPD") else w_aux
                l_fm = l_fm + feature_loss(fr, fg) * sc
                l_adv = l_adv + generator_loss(g)[0] * sc
            l_bpd = torch.zeros((), device=dev)
            if w_bpd:
                b_g = baseband_phase_diff(pha_g, h.hop_size, h.n_fft)
                b_t = baseband_phase_diff(pha_t, h.hop_size, h.n_fft)
                l_bpd = antiwrap(b_g - b_t).abs().mean() * w_bpd
            l_hpc = torch.zeros((), device=dev)
            if w_hpc:
                f0, vo = frame_f0(nb.squeeze(1), sr, h.hop_size, h.win_size)
                l_hpc = hpc_loss(pha_g, mag_g, f0, vo, sr, h.n_fft, h.hop_size, f_cut) * w_hpc
            l_desc = torch.zeros((), device=dev)
            if desc_hat is not None:
                l_desc = F.mse_loss(desc_hat, desc) * w_desc

            l_sc = torch.zeros((), device=dev)
            l_lm = torch.zeros((), device=dev)
            l_be = torch.zeros((), device=dev)
            if w_sc or w_lm:
                sc_, lm_ = LA.mrstft_loss(audio_g.squeeze(1), hr.squeeze(1))
                l_sc, l_lm = sc_ * w_sc, lm_ * w_lm
            if w_be:
                l_be = LA.band_energy_loss(audio_g.squeeze(1), hr.squeeze(1), sr,
                                           f_lo=(f_cut if amp_hf_only else None)) * w_be

            loss_g = (l_mag + l_pha + l_com + l_stft + l_fm + l_adv + l_hpc + l_desc
                      + l_bpd + l_sc + l_be + l_lm)
            if not torch.isfinite(loss_g):
                nonfinite += 1
                print(f"[{exp['name']}] step {step}: non-finite generator loss "
                      f"({nonfinite}/{NONFINITE_MAX}) - skipping the update. "
                      f"mag {l_mag.item():.3g} sc {l_sc.item():.3g} be {l_be.item():.3g} "
                      f"lm {l_lm.item():.3g} pha {l_pha.item():.3g}", flush=True)
                optim_g.zero_grad(set_to_none=True)
                if nonfinite >= NONFINITE_MAX:
                    raise RuntimeError(
                        f"{nonfinite} non-finite generator losses; aborting instead of "
                        f"training into NaN. Lower the amplitude-loss weights "
                        f"(sc={w_sc}, be={w_be}, lm={w_lm}) or the learning rate.")
                step += 1
                continue
            nonfinite = 0
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(g_params, 10.0)
            optim_g.step()
            if step % 200 == 0 and not all(torch.isfinite(p_).all()
                                           for p_ in G.parameters()):
                raise RuntimeError(f"generator weights became non-finite at step {step}")

            if step % exp.get("log_every", 50) == 0:
                with curves.open("a") as f:
                    f.write(f"{step},{loss_g.item():.4f},{loss_d.item():.4f},{l_mag.item():.4f},"
                            f"{l_pha.item():.4f},{l_com.item():.4f},{l_stft.item():.4f},"
                            f"{l_fm.item():.4f},{l_adv.item():.4f},{l_hpc.item():.4f},"
                            f"{l_desc.item():.4f},{l_bpd.item():.4f},"
                            f"{l_sc.item():.4f},{l_be.item():.4f},{l_lm.item():.4f}\n")
                print(f"[{exp['name']}] step {step}/{stop_at} G {loss_g.item():.2f} "
                      f"D {loss_d.item():.2f} sc {l_sc.item():.2f} "
                      f"be {l_be.item():.2f} lm {l_lm.item():.2f} hpc {l_hpc.item():.3f} "
                      f"desc {l_desc.item():.3f} bpd {l_bpd.item():.3f} "
                      f"({(time.time()-t0)/max(1, step-start_step+1):.2f}s/step)",
                      flush=True)
            if step and step % exp.get("val_every", 250) == 0:
                _summ, _ = run_val(step)
                save_best(step, _summ)
                last_val_step = step
                if best_state["step"] >= 0 and \
                        step - best_state["step"] >= PATIENCE * exp.get("val_every", 250):
                    stopped_early = True
                    print(f"[{exp['name']}] early stop at step {step}: no {SEL_METRIC} "
                          f"improvement since step {best_state['step']}", flush=True)
                    break
            if ckpt_every and step and step % ckpt_every == 0:
                save_ckpt(step)
            step += 1
            if budget_s and (time.time() - t0) > budget_s:
                partial = True
                break
        if partial or stopped_early:
            break
        sch_g.step()
        if sch_d: sch_d.step()
        if step >= stop_at:
            break

    if partial:
        save_ckpt(step)
        print(f"PARTIAL {exp['name']} stopped at step {step}/{stop_at} "
              f"(time budget); resume in a new session", flush=True)
        raise SystemExit(42)

    @torch.no_grad()
    def complexity(seconds=1.0, reps=20):
        """MACs, FLOPs, real-time factor and inference time: UDSS-BWE's Table 5."""
        G.eval()
        n = int(sr * seconds)
        x = torch.zeros(1, n, device=dev)
        mi, pi, _ = amp_pha_stft(x, h.n_fft, h.hop_size, h.win_size)
        macs = None
        try:
            from thop import profile
            macs, _ = profile(G, inputs=(mi, pi), verbose=False)
        except Exception:
            pass
        for _ in range(3):
            predict(mi, pi)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        for _ in range(reps):
            predict(mi, pi)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        per = (time.time() - t) / reps
        G.train()
        return dict(macs_g=(macs / 1e9 if macs else None),
                    flops_g=(2 * macs / 1e9 if macs else None),
                    infer_ms=per * 1e3, rtf=per / seconds, device=str(dev),
                    sb_steps=(sched.steps if use_sb else 0))

    save_ckpt(step)

    def safely(label, fn, default=None):
        try:
            return fn()
        except Exception as e:
            import traceback
            print(f"  !! {label} failed ({type(e).__name__}: {e}) - continuing", flush=True)
            traceback.print_exc()
            return default

    if not partial and last_val_step != step:
        _summ, _ = run_val(step)
        save_best(step, _summ)
    sel_note = "last"
    if BEST.exists() and exp.get("use_best", True):
        _b = torch.load(BEST, map_location=dev, weights_only=False)
        if _b.get("step", -1) != step:
            G.load_state_dict(_b["G"])
            if enc is not None and _b.get("enc"):
                enc.load_state_dict(_b["enc"])
            sel_note = f"best({_b['select']}={_b['metric']:.4f}@step{_b['step']})"
            print(f"final evaluation uses the BEST checkpoint: {sel_note} "
                  f"(last step was {step})", flush=True)
        else:
            sel_note = f"best==last@step{step}"

    cx = safely("complexity", complexity, {}) or {}
    if cx:
        print("COMPLEXITY " + json.dumps(cx), flush=True)
    summary, per_recipe = safely("final evaluation",
                                 lambda: run_val(step, final=True), ({}, {}))
    safely("sample dump", dump_samples)
    res = dict(name=exp["name"], generator=exp.get("generator", "cnx"),
               critics="+".join(exp["critics"]) or "none", hpc_weight=w_hpc,
               sc_weight=w_sc, be_weight=w_be, lm_weight=w_lm, bpd_weight=w_bpd,
               adv_anchor=w_anchor, adv_aux=w_aux, sb_t_max=sched.t_max,
               conditioned=conditioned, train_bank=exp.get("train_bank", "sinc"),
               backbone=exp.get("backbone", "?"), sb_x0=("prior" if sb_x0_prior else "raw"),
               weights=sel_note, best_step=best_state["step"], stopped_early=stopped_early,
               n_train=len(tr_ids), n_val=len(va_ids), n_test=len(te_ids),
               gen_params=n_g, enc_params=n_e, disc_params=sum(n_d.values()),
               disc_breakdown=n_d, steps=step, epochs=exp["epochs"], seed=exp["seed"],
               minutes=(time.time() - t0) / 60,
               by_bank=summary, by_recipe=per_recipe)
    res.update(cx)
    primary = eval_banks[0]
    for kk, vv in summary.get(primary, {}).items():
        res[kk] = vv
    (out / "result.json").write_text(json.dumps(res, indent=2))

    print("RESULT " + json.dumps({k: v for k, v in res.items()
                                  if k not in ("by_recipe", "by_bank")}), flush=True)

if __name__ == "__main__":
    main()
