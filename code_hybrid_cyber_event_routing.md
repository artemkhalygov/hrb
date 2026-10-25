# Код до статті «Hybrid Rule-Based and AI-Driven Architecture for Intelligent Cyber Event Routing and Filtering»

## Частина 1. Експерименти

### README

`hybrid-routing-exp/README.md`

```markdown
# Hybrid rule-based + ML (+ LLM) + HITL cyber event routing — experiment code

Reproduces Tables 3–5 of the article "Hybrid Rule-Based and AI-Driven Architecture for
Intelligent Cyber Event Routing and Filtering".

## Data (not redistributed here; download into `data/`)
| File | Source used in the experiments |
|---|---|
| `labelled_train.csv`, `labelled_validation.csv`, `labelled_test.csv` | BETH, processed 7-field version (fields: processId, threadId, parentProcessId, userId, mountNamespace, argsNum, returnValue, sus_label). Copy used: github.com/Bou0ajmi/Detecting-Cybersecurity-Threats-Using-Deep-Learning. Original: BETH dataset (Kaggle). |
| `HDFS_100k.log_structured.csv`, `anomaly_label.csv` | Loghub HDFS_v1 sample, from github.com/logpai/loglizer (`data/HDFS`). Original: Loghub (Zenodo). |

Check the licence/terms of use of both datasets before publishing the data together with the code.

## Run
```
pip install numpy pandas scipy scikit-learn        # all BSD-licensed
python src/experiment.py                           # ~10 min on 1 CPU core
```
Outputs in `results/`: `<DS>_runs.csv` (per seed), `<DS>_summary.csv` (mean/std),
`<DS>_params.csv` (selected thresholds/weights per seed), `<DS>_info.json`
(sizes, ML model selection, routing shares, latency, McNemar test, bootstrap CIs),
`<DS>_llm_queue.csv` (events routed to the LLM layer).

## LLM layer (reported run: claude-sonnet-4-6 via the Claude artifact runtime)

Reported run: `python src/prepare_llm_items.py` built the fixed event sets (`results/llm_items.json`, labels in `results/llm_labels.json`);
`src/llm_layer_eval.html` sent them to Claude (claude-sonnet-4-6, effort low, batches of 15/50) and stored the answers
(`results/llm_db/llm_results/*.json`); `python src/evaluate_llm.py` computes Table 6 (`results/llm_eval.json`).

Alternative with an API key:
```
pip install anthropic                              # MIT licence
export ANTHROPIC_API_KEY=...
python src/llm_layer.py queue HDFS                 # events routed to the LLM by the hybrid policy
python src/llm_layer.py sample HDFS 300            # LLM-only baseline on a stratified test sample
```
Model is set by `LLM_MODEL` (default in the script); temperature 0; JSON output.
After running, merge the C score into Eq. (3) and report the LLM rows in Table 4.

## Environment of the reported run
Python 3.12.3, numpy, pandas 3.0.2, scipy 1.17.1, scikit-learn 1.8.0; 1 vCPU Intel Xeon 2.8 GHz, 3 GB RAM.
Seeds: 11, 22, 33, 44, 55.

## Note on procedure
Before validation-based ML model selection was added to the script, test-set results of
alternative detectors were inspected once. The final choice is made on validation data only,
but for a strictly clean protocol the candidate list should be fixed in advance and the run repeated.
```

### Основний експеримент: правила, ML, гібрид, HITL, статистика, латентність

`hybrid-routing-exp/src/experiment.py`

```python
"""
Hybrid rule-based + ML (+ LLM) + HITL cyber event routing: reproducible experiment.

Runs the rule-based-only, ML-only and hybrid variants on BETH and HDFS,
simulates the human-in-the-loop (HITL) layer, and writes all tables to ../results.
The LLM layer is NOT executed here: events routed to it are exported to
results/<dataset>_llm_queue.csv and processed by llm_layer.py (requires an API key).
Until then, the hybrid decision for those events falls back to the preliminary risk R'.

Usage:  python src/experiment.py
"""
import json, time, itertools, os
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from scipy.stats import binomtest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, RES = os.path.join(ROOT, "data"), os.path.join(ROOT, "results")
SEEDS = [11, 22, 33, 44, 55]
LLM_BUDGET = 0.10          # max share of events forwarded to the LLM layer (B)
HITL_CAPACITY = 10 / 1000  # analyst capacity K: 10 escalations per 1,000 events
HITL_EPS = [0.05, 0.10]    # simulated analyst error rates
N_BOOT = 1000

# --------------------------------------------------------------------------- BETH
def load_beth():
    sp = {}
    for name, f in [("train", "labelled_train.csv"), ("val", "labelled_validation.csv"), ("test", "labelled_test.csv")]:
        d = pd.read_csv(os.path.join(DATA, f))
        sp[name] = d
    return sp

def beth_features(d):
    # standard BETH preprocessing (Highnam et al., 2021)
    return pd.DataFrame({
        "proc_is_system": d.processId.isin([0, 1, 2]).astype(int),
        "parent_is_system": d.parentProcessId.isin([0, 1, 2]).astype(int),
        "user_is_system": (d.userId < 1000).astype(int),
        "mnt_is_host": (d.mountNamespace == 4026531840).astype(int),
        "ret_negative": (d.returnValue < 0).astype(int),
        "ret_positive": (d.returnValue > 0).astype(int),
        "args_num": d.argsNum,
    })

BETH_RULES = [  # (name, weight, function on raw frame) -- defined a priori, not fitted to labels
    ("R1_non_system_user", 0.6, lambda d: d.userId >= 1000),
    ("R2_failed_operation", 0.4, lambda d: d.returnValue < 0),
]

# --------------------------------------------------------------------------- HDFS
EVENTS = [f"E{i}" for i in range(1, 30)]
def load_hdfs():
    d = pd.read_csv(os.path.join(DATA, "HDFS_100k.log_structured.csv"))
    lab = pd.read_csv(os.path.join(DATA, "anomaly_label.csv"))
    d["blk"] = d.Content.str.extract(r"(blk_-?\d+)")[0]
    d["ts"] = pd.to_datetime(d.Date.astype(str).str.zfill(6) + d.Time.astype(str).str.zfill(6), format="%y%m%d%H%M%S")
    g = d.groupby("blk")
    cnt = pd.crosstab(d.blk, d.EventId).reindex(columns=EVENTS, fill_value=0)
    blocks = pd.DataFrame({"first_ts": g.ts.min(), "n_lines": g.size(),
                           "text": g.Content.apply(lambda s: "\n".join(s))}).join(cnt)
    blocks["label"] = blocks.index.map(dict(zip(lab.BlockId, (lab.Label == "Anomaly").astype(int))))
    blocks = blocks.sort_values("first_ts")
    n = len(blocks); a, b = int(0.6 * n), int(0.8 * n)        # chronological 60/20/20
    return {"train": blocks.iloc[:a], "val": blocks.iloc[a:b], "test": blocks.iloc[b:]}

def hdfs_features(d):
    return np.log1p(d[EVENTS])

HDFS_RULES = [
    ("R1_exception_or_interrupt", 0.4, lambda d: (d[["E7", "E8", "E10", "E14"]].sum(axis=1) > 0)),
    ("R2_abnormal_transfer_event", 0.2, lambda d: (d[["E13", "E15", "E27"]].sum(axis=1) > 0)),
    ("R3_incomplete_write_pipeline", 0.4, lambda d: (d.E9 + d.E6 < d.E5) | (d.E26 < d.E9 + d.E6)),
]

# --------------------------------------------------------------------------- components
def rule_score(d, rules):
    return sum(w * f(d).astype(float).values for _, w, f in rules)

ML_CANDIDATES = ["isolation_forest", "one_class_svm"]
MAX_TRAIN = 20000  # normal training events sub-sampled for model fitting

class MLModule:
    def __init__(self, seed, kind="isolation_forest"):
        self.kind, self.seed = kind, seed
        if kind == "isolation_forest":
            self.m = IsolationForest(n_estimators=200, max_samples=256, contamination="auto", random_state=seed, n_jobs=1)
        else:
            self.m = OneClassSVM(kernel="rbf", nu=0.01, gamma="scale")
    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        if len(X) > MAX_TRAIN:
            X = X[rng.choice(len(X), MAX_TRAIN, replace=False)]
        self.m.fit(X)
        s = -self.m.score_samples(X); self.med, self.sd = np.median(s), max(np.std(s), 1e-6)
        return self
    def score(self, X):  # anomaly score A in [0, 1): 0 at the median normal score, -> 1 as deviation grows
        s = -self.m.score_samples(X)
        return 1 - np.exp(-np.maximum(0, s - self.med) / self.sd)

def metrics(y, yhat, score=None):
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    out = dict(precision=precision_score(y, yhat, zero_division=0), recall=recall_score(y, yhat, zero_division=0),
               f1=f1_score(y, yhat, zero_division=0), fpr=fp / max(fp + tn, 1))
    out["roc_auc"] = roc_auc_score(y, score) if score is not None and len(np.unique(y)) > 1 else np.nan
    return out

def fast_f1(y, yhat):
    tp = np.sum((y == 1) & (yhat == 1)); fp = np.sum((y == 0) & (yhat == 1)); fn = np.sum((y == 1) & (yhat == 0))
    return 2 * tp / max(2 * tp + fp + fn, 1)

def best_threshold(y, s):
    """Threshold on score s maximising F1 (exact, over all distinct score values)."""
    o = np.argsort(-s, kind="mergesort"); ss, yy = s[o], y[o]
    tp = np.cumsum(yy); fp = np.cumsum(1 - yy); P = yy.sum()
    last = np.r_[ss[1:] != ss[:-1], True]          # evaluate only at the end of tied groups
    f1 = np.where(last, 2 * tp / np.maximum(tp + fp + P, 1), -1)
    return float(ss[int(np.argmax(f1))])

# --------------------------------------------------------------------------- hybrid routing policy
def route(S, A_fn, p):
    """Return decision, preliminary risk R', layer label for every event (vectorised)."""
    n = len(S); layer = np.empty(n, dtype=object); R = S.copy().astype(float)
    high = S >= p["theta_H"]; low = (S < p["theta_L"]) & ~high
    mid = ~high & ~low
    layer[high] = "rule_high"; layer[low] = "monitor"
    A = np.full(n, np.nan)
    if mid.any():
        A[mid] = A_fn(mid)
        R[mid] = (p["w1"] * S[mid] + p["w2"] * A[mid]) / (p["w1"] + p["w2"])
        layer[mid] = "ml"
        band = mid & (np.abs(R - p["theta_R"]) < p["delta"])
        layer[band] = "llm_queue"
    dec = np.where(high, 1, np.where(low, 0, (R >= p["theta_R"]).astype(int)))
    R[high] = np.maximum(R[high], 1.0); R[low] = np.minimum(R[low], S[low])
    return dec, R, A, layer

def calibrate_delta(R, mid_mask, theta_R, budget):
    """Largest uncertainty band so that the LLM share does not exceed the budget B."""
    dist = np.abs(R[mid_mask] - theta_R)
    if len(dist) == 0: return 0.0
    k = int(budget * len(R))
    if k < 1: return 0.0
    if k >= len(dist): return float(dist.max()) + 1e-9
    return float(np.sort(dist)[k])   # strict band |R'-theta_R| < delta keeps the share <= B

def fit_hybrid(S_val, A_val, y_val):
    grid = []
    for th_H, th_L, w1 in itertools.product([0.6, 0.8, 1.0, 1.01], [0.0, 0.2, 0.4, 0.6], [0.2, 0.4, 0.5, 0.6, 0.8]):
        if th_L > th_H: continue
        p = dict(theta_H=th_H, theta_L=th_L, w1=w1, w2=1 - w1)
        high = S_val >= th_H; low = (S_val < th_L) & ~high; mid = ~high & ~low
        R = np.where(mid, (w1 * S_val + (1 - w1) * A_val), S_val)
        th_R = best_threshold(y_val[mid], R[mid]) if mid.sum() > 0 and len(np.unique(y_val[mid])) > 1 else 0.5
        dec = np.where(high, 1, np.where(low, 0, (R >= th_R).astype(int)))
        f1 = fast_f1(y_val, dec)
        ml_share = mid.mean()
        grid.append((f1, -ml_share, dict(p, theta_R=th_R)))
    grid.sort(key=lambda g: (round(g[0], 4), g[1]), reverse=True)   # max F1, then min ML load
    p = grid[0][2]
    mid = ~(S_val >= p["theta_H"]) & ~((S_val < p["theta_L"]))
    R = np.where(mid, p["w1"] * S_val + p["w2"] * A_val, S_val)
    p["delta"] = calibrate_delta(R, mid, p["theta_R"], LLM_BUDGET)
    return p

def hitl(dec, R, S, A, layer, y, p, eps, rng):
    """Simulated analyst: escalates disagreements and uncertain events, limited capacity, error rate eps."""
    rule_pos = S >= 0.5
    ml_pos = np.where(np.isnan(A), rule_pos, A >= p["theta_A"])
    cand = np.where(((rule_pos != ml_pos) & (layer != "monitor")) | (layer == "llm_queue"))[0]
    cap = int(HITL_CAPACITY * len(y))
    order = cand[np.argsort(np.abs(R[cand] - p["theta_R"]))][:cap]
    out = dec.copy()
    correct = rng.random(len(order)) >= eps
    out[order] = np.where(correct, y[order], 1 - y[order])
    return out, len(order)

def mcnemar(y, a, b):
    ca, cb = a == y, b == y
    n01, n10 = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 > 0 else 1.0
    return n01, n10, p

def boot_ci(y, yhat, rng):
    n = len(y); vals = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n); vals.append(fast_f1(y[i], yhat[i]))
    return np.percentile(vals, [2.5, 97.5])

# --------------------------------------------------------------------------- latency
def per_event_latency(fn, rows, n=2000):
    t = []
    for i in range(min(n, len(rows))):
        s = time.perf_counter(); fn(rows[i:i + 1]); t.append((time.perf_counter() - s) * 1000)
    return np.array(t)  # ms

# --------------------------------------------------------------------------- main
def run(name, sp, feat_fn, rules, text_col=None):
    y = {k: sp[k]["label"].values if "label" in sp[k] else sp[k]["sus_label"].values for k in sp}
    X = {k: np.asarray(feat_fn(sp[k]), dtype=float) for k in sp}
    S = {k: rule_score(sp[k], rules) for k in sp}
    train_norm = X["train"][y["train"] == 0]
    info = dict(dataset=name, sizes={k: int(len(y[k])) for k in sp}, positives={k: int(y[k].sum()) for k in sp},
                train_normal_used=int(len(train_norm)))
    # ML model selection on the validation subset (never on test)
    sel = {}
    for kind in ML_CANDIDATES:
        m = MLModule(SEEDS[0], kind).fit(train_norm); a = m.score(X["val"])
        sel[kind] = fast_f1(y["val"], (a >= best_threshold(y["val"], a)).astype(int))
    ml_kind = max(sel, key=sel.get); info["ml_selection_val_f1"] = sel; info["ml_model"] = ml_kind
    print(name, "ML selection on validation:", sel, "->", ml_kind, flush=True)
    # rule-only threshold
    th_RB = best_threshold(y["val"], S["val"])
    rows, routing, stats, params_all = [], [], {}, []
    preds = {}
    for seed in SEEDS:
        ml = MLModule(seed, ml_kind).fit(train_norm)
        A_val, A_test = ml.score(X["val"]), ml.score(X["test"])
        th_A = best_threshold(y["val"], A_val)
        p = fit_hybrid(S["val"], A_val, y["val"]); p["theta_A"] = th_A; p["theta_RB"] = th_RB
        params_all.append(p)
        yr = (S["test"] >= th_RB).astype(int); ym = (A_test >= th_A).astype(int)
        dec, R, A, layer = route(S["test"], lambda m: A_test[m], p)
        rows += [dict(seed=seed, variant="Rule-based only", **metrics(y["test"], yr, S["test"])),
                 dict(seed=seed, variant="ML only", **metrics(y["test"], ym, A_test)),
                 dict(seed=seed, variant="Hybrid (rules + ML, no HITL)", **metrics(y["test"], dec, R))]
        rng = np.random.default_rng(seed)
        for eps in HITL_EPS:
            dh, nesc = hitl(dec, R, S["test"], A, layer, y["test"], p, eps, rng)
            rows.append(dict(seed=seed, variant=f"Hybrid + HITL (eps={eps:.2f})", **metrics(y["test"], dh, R)))
        lay = pd.Series(layer).value_counts(normalize=True).to_dict()
        lay["hitl_escalated"] = nesc / len(y["test"])
        routing.append(lay)
        preds[seed] = dict(rule=yr, ml=ym, hyb=dec, hitl=dh)
        if seed == SEEDS[0]:
            q = sp["test"].iloc[np.where(layer == "llm_queue")[0]].copy()
            q["S_RB"], q["A"], q["R_prelim"] = S["test"][layer == "llm_queue"], A[layer == "llm_queue"], R[layer == "llm_queue"]
            q.to_csv(os.path.join(RES, f"{name}_llm_queue.csv"))
            # latency (ms per event, single-event processing)
            Xt = X["test"]; raw = sp["test"]
            t_rule = per_event_latency(lambda r: rule_score(r, rules), raw)
            t_ml = per_event_latency(lambda r: ml.score(r), Xt)
            stats["latency_ms"] = {"rule": [float(np.median(t_rule)), float(np.percentile(t_rule, 95)), float(t_rule.mean())],
                                   "ml": [float(np.median(t_ml)), float(np.percentile(t_ml, 95)), float(t_ml.mean())]}
            ml_share = float(np.isin(layer, ["ml", "llm_queue"]).mean())
            stats["latency_ms"]["hybrid_mean_without_llm"] = float(t_rule.mean() + ml_share * t_ml.mean())
    df = pd.DataFrame(rows)
    summ = df.groupby("variant", sort=False).agg(["mean", "std"]).drop(columns="seed")
    # significance + CI (seed 1)
    s0 = SEEDS[0]; rng = np.random.default_rng(0)
    base_name = "ml" if df[df.variant == "ML only"].f1.mean() >= df[df.variant == "Rule-based only"].f1.mean() else "rule"
    stats["mcnemar_hybrid_vs_best_baseline"] = dict(zip(["hybrid_correct_only", "baseline_correct_only", "p_value"],
                                                          mcnemar(y["test"], preds[s0]["hyb"], preds[s0][base_name])))
    stats["best_baseline"] = base_name
    stats["f1_ci95"] = {k: boot_ci(y["test"], preds[s0][k], rng).round(4).tolist() for k in ["rule", "ml", "hyb", "hitl"]}
    rout = pd.DataFrame(routing).fillna(0).mean().to_dict()
    pd.DataFrame(params_all).to_csv(os.path.join(RES, f"{name}_params.csv"), index=False)
    df.to_csv(os.path.join(RES, f"{name}_runs.csv"), index=False)
    summ.to_csv(os.path.join(RES, f"{name}_summary.csv"))
    json.dump(dict(info=info, stats=stats, routing=rout), open(os.path.join(RES, f"{name}_info.json"), "w"), indent=2, default=float)
    print(f"\n===== {name} =====\n", json.dumps(info), "\n", summ.round(4).to_string(), "\nrouting", rout, "\n", json.dumps(stats, default=float, indent=1))
    print(pd.DataFrame(params_all).round(4).to_string())

if __name__ == "__main__":
    os.makedirs(RES, exist_ok=True)
    run("BETH", load_beth(), beth_features, BETH_RULES)
    run("HDFS", load_hdfs(), hdfs_features, HDFS_RULES)
```

### Підготовка фіксованих наборів подій для LLM

`hybrid-routing-exp/src/prepare_llm_items.py`

```python
"""Builds the fixed event sets for the LLM evaluation (results/llm_items.json, labels in llm_labels.json)."""
import sys, os, json, pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); import experiment as ex
RES = ex.RES
def strat(df, ycol, n, seed=0):
    parts = [g.sample(max(1, int(round(n * len(g) / len(df)))), random_state=seed) for _, g in df.groupby(ycol)]
    return pd.concat(parts)
def lines(t, k=40):
    L = t.split('\n'); return '\n'.join(L[:k]) + (f'\n... [{len(L)-k} more lines]' if len(L) > k else '')
q = pd.read_csv(os.path.join(RES, 'HDFS_llm_queue.csv'), index_col=0)
test = ex.load_hdfs()['test']; bt = ex.load_beth()['test'].reset_index(drop=True)
hs = strat(test, 'label', 300); bs = strat(bt, 'sus_label', 300)
keys = ["processId", "threadId", "parentProcessId", "userId", "mountNamespace", "argsNum", "returnValue"]
items, labels = [], {}
for blk, r in q.iterrows():
    i = f"HQ_{blk}"; items.append(dict(id=i, task="hybrid_hdfs", text=lines(r.text), s_rule=round(float(r.S_RB), 3), s_ml=round(float(r.A), 3))); labels[i] = int(r.label)
for blk, r in hs.iterrows():
    i = f"HS_{blk}"; items.append(dict(id=i, task="llmonly_hdfs", text=lines(r.text))); labels[i] = int(r.label)
for idx, r in bs.iterrows():
    i = f"BS_{idx}"; items.append(dict(id=i, task="llmonly_beth", text=json.dumps({k: int(r[k]) for k in keys}))); labels[i] = int(r.sus_label)
json.dump(items, open(os.path.join(RES, 'llm_items.json'), 'w')); json.dump(labels, open(os.path.join(RES, 'llm_labels.json'), 'w'))
print(len(q), int(q.label.sum()), len(hs), int(hs.label.sum()), len(bs), int(bs.sus_label.sum()), len(items), max(len(x['text']) for x in items))
```

### LLM-шар через Anthropic API (альтернатива з API-ключем)

`hybrid-routing-exp/src/llm_layer.py`

```python
"""
LLM context layer (C score, explanation, recommended action).

Two modes:
  python src/llm_layer.py queue  HDFS   -> processes results/HDFS_llm_queue.csv (hybrid routing)
  python src/llm_layer.py sample HDFS N -> LLM-only baseline on a stratified sample of N test events

Requires: pip install anthropic ; environment variable ANTHROPIC_API_KEY.
Model and settings are fixed below and must be reported in the paper.
"""
import os, sys, json, time
import pandas as pd, numpy as np

MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-5-5")
TEMPERATURE = 0.0
MAX_TOKENS = 400

SYSTEM = (
    "You are a security analyst assistant in a SOC. You receive one security event "
    "(a kernel/process event or an HDFS block log session) together with scores from a "
    "rule-based gate (S_RB) and an anomaly detector (A), both in [0,1]. "
    "Assess whether the event is malicious/anomalous. Respond ONLY with a JSON object: "
    '{"label": "benign" | "malicious", "confidence": <float 0..1, probability that the event is malicious>, '
    '"explanation": "<max 2 sentences>", "recommended_action": "<one action>"}'
)

def describe(row, dataset):
    if dataset == "HDFS":
        body = row["text"][:6000]
        return f"HDFS block session log lines:\n{body}"
    keys = ["processId", "threadId", "parentProcessId", "userId", "mountNamespace", "argsNum", "returnValue"]
    return "Kernel event: " + json.dumps({k: int(row[k]) for k in keys if k in row})

def ask(client, row, dataset):
    extra = ""
    if "S_RB" in row and not pd.isna(row["S_RB"]):
        extra = f"\nS_RB={row['S_RB']:.3f}, A={row['A']:.3f}"
    t0 = time.perf_counter()
    msg = client.messages.create(model=MODEL, max_tokens=MAX_TOKENS, temperature=TEMPERATURE, system=SYSTEM,
                                 messages=[{"role": "user", "content": describe(row, dataset) + extra}])
    dt = time.perf_counter() - t0
    txt = "".join(b.text for b in msg.content if b.type == "text").strip().strip("`").removeprefix("json").strip()
    try:
        out = json.loads(txt)
    except json.JSONDecodeError:
        out = {"label": "malicious", "confidence": 0.5, "explanation": "PARSE_ERROR", "recommended_action": "manual review"}
    out.update(latency_s=dt, in_tokens=msg.usage.input_tokens, out_tokens=msg.usage.output_tokens)
    return out

def main():
    import anthropic
    mode, dataset = sys.argv[1], sys.argv[2]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    res = os.path.join(root, "results")
    if mode == "queue":
        df = pd.read_csv(os.path.join(res, f"{dataset}_llm_queue.csv"), index_col=0)
    else:
        sys.path.insert(0, os.path.join(root, "src"))
        import experiment as ex
        sp = ex.load_hdfs() if dataset == "HDFS" else ex.load_beth()
        test = sp["test"]; ycol = "label" if "label" in test else "sus_label"
        n = int(sys.argv[3]); rng = np.random.default_rng(0)
        df = test.groupby(ycol, group_keys=False).apply(
            lambda g: g.sample(max(1, int(round(n * len(g) / len(test)))), random_state=0))
    client = anthropic.Anthropic()
    rows = []
    for idx, row in df.iterrows():
        r = ask(client, row, dataset); r["index"] = idx; rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(res, f"{dataset}_llm_{mode}_out.csv"), index=False)
    print(out[["latency_s", "in_tokens", "out_tokens"]].describe())

if __name__ == "__main__":
    main()
```

### Оцінка LLM-шару

`hybrid-routing-exp/src/evaluate_llm.py`

```python
"""
Merges the LLM answers (exported from the evaluation page's database into results/llm_db/llm_results/*.json)
with the seed-11 pipeline and reports:
  * Hybrid + LLM on the full HDFS test subset (LLM label replaces R' for events in the uncertainty band);
  * LLM-only vs. rule/ML/hybrid on the same stratified samples (HDFS, BETH);
  * LLM latency per event (batch time / batch size).
Usage: python src/evaluate_llm.py
"""
import sys, os, json, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); import experiment as ex

def load_answers():
    rows, batches = [], []
    for f in glob.glob(os.path.join(ex.RES, "llm_db", "**", "*.json"), recursive=True):
        d = json.load(open(f)); d = d.get("data", d)
        if "answers" not in d: continue
        batches.append(dict(batch=d["batch"], task=d["task"], n=d["n"], seconds=d["seconds"], model=d.get("modelApplied") or d.get("tierApplied")))
        for a in d["answers"]:
            rows.append(dict(a, task=d["task"]))
    return pd.DataFrame(rows), pd.DataFrame(batches)

def pipeline(name):
    sp = ex.load_hdfs() if name == "HDFS" else ex.load_beth()
    feat, rules = (ex.hdfs_features, ex.HDFS_RULES) if name == "HDFS" else (ex.beth_features, ex.BETH_RULES)
    ycol = "label" if name == "HDFS" else "sus_label"
    y = {k: sp[k][ycol].values for k in sp}; X = {k: np.asarray(feat(sp[k]), float) for k in sp}
    S = {k: ex.rule_score(sp[k], rules) for k in sp}
    ml = ex.MLModule(ex.SEEDS[0], "one_class_svm").fit(X["train"][y["train"] == 0])
    Av, At = ml.score(X["val"]), ml.score(X["test"])
    p = ex.fit_hybrid(S["val"], Av, y["val"]); thA = ex.best_threshold(y["val"], Av); thRB = ex.best_threshold(y["val"], S["val"])
    dec, R, A, layer = ex.route(S["test"], lambda m: At[m], p)
    idx = sp["test"].index
    return pd.DataFrame({"y": y["test"], "rule": (S["test"] >= thRB).astype(int), "ml": (At >= thA).astype(int),
                         "hybrid": dec, "layer": layer, "R": R, "S": S["test"], "A": At}, index=idx), p

def main():
    ans, bt = load_answers()
    ans["pred"] = (ans.label.str.lower().str.startswith("mal")).astype(int)
    out = {"batches": len(bt), "models": bt.model.value_counts().to_dict(),
           "latency_s_per_event": {t: float((g.seconds / g.n).median()) for t, g in bt.groupby("task")},
           "latency_s_per_event_p95": {t: float(np.percentile(g.seconds / g.n, 95)) for t, g in bt.groupby("task")}}
    # HDFS hybrid + LLM
    h, p = pipeline("HDFS")
    q = ans[ans.task == "hybrid_hdfs"].assign(blk=lambda d: d.id.str[3:]).set_index("blk")
    h["hybrid_llm"] = h.hybrid
    hit = h.index.intersection(q.index); h.loc[hit, "hybrid_llm"] = q.loc[hit, "pred"].values
    score = h.R.copy(); score.loc[hit] = q.loc[hit, "confidence"].values
    out["HDFS_full_test"] = {v: ex.metrics(h.y.values, h[v].values) for v in ["rule", "ml", "hybrid", "hybrid_llm"]}
    out["HDFS_llm_queue"] = {"n": int(len(hit)), "positives": int(h.loc[hit, "y"].sum()),
                             "hybrid_without_llm_correct": int((h.loc[hit, "hybrid"] == h.loc[hit, "y"]).sum()),
                             "hybrid_with_llm_correct": int((h.loc[hit, "hybrid_llm"] == h.loc[hit, "y"]).sum())}
    out["HDFS_mcnemar_hybrid_llm_vs_hybrid"] = dict(zip(["llm_correct_only", "hybrid_correct_only", "p"], ex.mcnemar(h.y.values, h.hybrid_llm.values, h.hybrid.values)))
    # LLM-only on stratified samples
    for name, task, pref in [("HDFS", "llmonly_hdfs", "HS_"), ("BETH", "llmonly_beth", "BS_")]:
        s = ans[ans.task == task].copy(); s["key"] = s.id.str[len(pref):]
        base = h if name == "HDFS" else pipeline("BETH")[0].reset_index(drop=True)
        base.index = base.index.astype(str); s = s.set_index("key"); keys = base.index.intersection(s.index)
        b = base.loc[keys]; llm = s.loc[keys]
        out[f"{name}_sample"] = {"n": int(len(keys)), "positives": int(b.y.sum()),
                                 **{v: ex.metrics(b.y.values, b[v].values) for v in ["rule", "ml", "hybrid"]},
                                 "llm_only": ex.metrics(b.y.values, llm.pred.values, llm.confidence.fillna(0.5).values)}
    json.dump(out, open(os.path.join(ex.RES, "llm_eval.json"), "w"), indent=2, default=float)
    print(json.dumps(out, indent=1, default=float))

if __name__ == "__main__":
    main()
```