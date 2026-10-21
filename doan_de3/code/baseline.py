"""Baseline TF-IDF + Logistic Regression / Linear SVM cho 4 bai toan, co xu ly mat can bang lop.
Chay: python baseline.py --data data/clean.csv --out results_baseline
Bai toan: topic (primary_topic, multiclass) | multilabel (topics) | sentiment | priority
Moi to hop {lr, svm} x {none, class_weight, oversample}: chon C tren val Macro-F1, danh gia tren test va ood.
Dau ra: results_baseline.csv, confusion_<task>.png, per_class_<task>.csv (mo hinh tot nhat tren val moi bai toan).
"""
import argparse, itertools, os, time
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix, classification_report

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/clean.csv")
p.add_argument("--out", default="results_baseline")
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
rng = np.random.RandomState(args.seed)
df = pd.read_csv(args.data, encoding="utf-8-sig", keep_default_na=False)
S = {s: df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test", "ood")}
# ---------- dac trung: tu (1-2 gram) + ky tu (2-5 gram, chiu duoc cau thieu dau); chi fit tren train
wv = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(S["train"].text)
cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2).fit(S["train"].text)
feat = lambda t: hstack([wv.transform(t), cv.transform(t)]).tocsr()
X = {s: feat(d.text) for s, d in S.items()}
TASKS = {"topic": "primary_topic", "sentiment": "sentiment", "priority": "priority", "multilabel": "topics"}
split_t = lambda s: [t.strip() for t in s.split(";")]
def oversample(x, y):  # nhan ban ngau nhien mau lop thieu cho bang lop lon nhat (chi tren train)
    cnt = np.bincount(y); idx = np.concatenate([rng.choice(np.where(y == k)[0], cnt.max()) for k in range(len(cnt)) if cnt[k]])
    return x[idx], y[idx]
def macro(y, pred, multilabel):  # multilabel: chi tinh nhan co mau trong tap danh gia
    if not multilabel: return f1_score(y, pred, average="macro", zero_division=0)
    sup = y.sum(0) > 0; return f1_score(y[:, sup], pred[:, sup], average="macro", zero_division=0)
rows, best_models = [], {}
for task, col in TASKS.items():
    ml = task == "multilabel"
    if ml:
        mlb = MultiLabelBinarizer().fit([split_t(t) for t in df[col]]); classes = list(mlb.classes_)
        Y = {s: mlb.transform([split_t(t) for t in d[col]]) for s, d in S.items()}
    else:
        classes = sorted(df[col].unique()); Y = {s: np.array([classes.index(v) for v in d[col]]) for s, d in S.items()}
    best = None
    for model, balance in itertools.product(("lr", "svm"), ("none", "class_weight") + (() if ml else ("oversample",))):
        cw = "balanced" if balance == "class_weight" else None
        cand = None
        for C in (0.1, 1, 10, 100):  # chinh hyperparameter tren val
            base = LogisticRegression(C=C, class_weight=cw, max_iter=3000) if model == "lr" else LinearSVC(C=C, class_weight=cw, max_iter=5000)
            x, y = X["train"], Y["train"]
            if balance == "oversample": x, y = oversample(x, y)
            t0 = time.time(); m = (OneVsRestClassifier(base) if ml else base).fit(x, y); sec = time.time() - t0
            v = macro(Y["val"], m.predict(X["val"]), ml)
            if cand is None or v > cand[0]: cand = (v, m, C, sec)
        v, m, C, sec = cand
        lat = []  # do tre tung mau (TF-IDF + du doan)
        for t in S["test"].text[:200]:
            t0 = time.perf_counter(); m.predict(feat([t])); lat.append((time.perf_counter() - t0) * 1000)
        for s in ("val", "test", "ood"):
            pr = m.predict(X[s]); y = Y[s]
            r = dict(task=task, model=model, balance=balance, C=C, split=s, macro_f1=macro(y, pr, ml),
                     micro_f1=f1_score(y, pr, average="micro", zero_division=0), weighted_f1=f1_score(y, pr, average="weighted", zero_division=0),
                     latency_ms=np.mean(lat) if s == "test" else np.nan, train_sec=round(sec, 2))
            if task == "priority" and s != "val":  # precision/recall lop khan cap
                k = classes.index("cao"); P, R, _, _ = precision_recall_fscore_support(y == k, pr == k, average="binary", zero_division=0)
                r.update(urgent_precision=P, urgent_recall=R)
            if ml and s == "test":  # phan tich cau don nhan vs da nhan
                multi = S["test"].topics.str.contains(";").values
                f = lambda k: f1_score(y[k], pr[k], average="micro", zero_division=0) if k.any() else float("nan")  # tap test co the khong co mau da nhan
                r["note"] = "micro_f1 don nhan=%.3f (n=%d); da nhan=%.3f (n=%d)" % (f(~multi), (~multi).sum(), f(multi), multi.sum())
            rows.append(r)
        if best is None or v > best[0]: best = (v, m, model, balance)
    best_models[task] = (best[1], classes, f"{best[2]}+{best[3]}")
res = pd.DataFrame(rows); res.round(4).to_csv(f"{args.out}/results_baseline.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 220)
print("=== KET QUA TREN TEST ==="); print(res[res.split == "test"].drop(columns=["split", "train_sec"]).round(3).to_string(index=False))
print("\n=== KET QUA TREN OOD (14 mau, chi tham khao) ==="); print(res[res.split == "ood"][["task", "model", "balance", "macro_f1", "micro_f1"]].round(3).to_string(index=False))
# ---------- confusion matrix + per-class (mo hinh tot nhat tren val, tap test)
for task, col in TASKS.items():
    if task == "multilabel": continue
    m, classes, name = best_models[task]; y = np.array([classes.index(v) for v in S["test"][col]]); pr = m.predict(X["test"])
    labels = [i for i in range(len(classes)) if i in set(y) | set(pr)]; names = [classes[i] for i in labels]
    cm = confusion_matrix(y, pr, labels=labels)
    fig, ax = plt.subplots(figsize=(7.5, 6.5)); ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), names, rotation=45, ha="right", fontsize=8); ax.set_yticks(range(len(labels)), names, fontsize=8)
    for i, j in itertools.product(range(len(labels)), repeat=2):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế"); ax.set_title(f"Confusion matrix - {task} ({name})")
    plt.tight_layout(); plt.savefig(f"{args.out}/confusion_{task}.png", dpi=130); plt.close()
    pd.DataFrame(classification_report(y, pr, labels=labels, target_names=names, output_dict=True, zero_division=0)).T.round(3).to_csv(f"{args.out}/per_class_{task}.csv", encoding="utf-8-sig")
print(f"\nDa luu {args.out}/results_baseline.csv, confusion_*.png, per_class_*.csv")
