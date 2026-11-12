#!/usr/bin/env python
"""
ehg_analysis.py -- does uterine EHG predict preterm birth once leakage and
confounders are handled? (BME 304 project, Shriya Minocha & Saanvee Sunkara)

Follows the Methods section in order:
  1. load the WFDB datasets in this repo and parse their headers
  2. clean: log exclusions, assign pregnancy IDs, relabel preterm (< 37 wk)
  3. filter (4th-order zero-phase Butterworth), trim 180 s, extract
     RMS / median & peak frequency / sample entropy; check against the
     feature values shipped with TPEHG DB
  4. RQ1: LR, RF, SVM x {none, class weights, SMOTE, ADASYN}, repeated
     pregnancy-grouped CV, bootstrap CIs, permutation test, and a deliberately
     leaky run to show the Vandewiele et al. effect
  5. RQ2: GA only / clinical / GA+clinical / EHG / EHG+clinical+GA,
     split by early (< 26 wk) vs late recordings
  6. RQ3: specification curve over band x channel x feature set x period
  7. external validation on the Icelandic 16-electrode database
  8. write everything to outputs/ (tables, figures, results.txt)

Expected folders in the repo root (as downloaded from PhysioNet):
    term-preterm-ehg-database-1.0.1/term-preterm-ehg-database-1.0.1/   TPEHG DB
    term-preterm-ehg-dataset-with-tocogram-1.0.0/                       TPEHGT DS
    icelandic-ehg-dataset/                                              EHGDB (Iceland)
    icehg-ds-1.0.1/                                                     ICEHG DS
All three Ljubljana sets come from the same hospital pool and share record
numbers. Six ICEHG DS records are byte-identical copies of TPEHG DB records
(dropped as duplicates), and ICEHG's Pair_RecID field links early/late
recordings of one pregnancy, including some pairs that both sit inside
TPEHG DB. Those links are used to build pregnancy IDs so grouped CV never
splits a pregnancy.

Stages: load, features, verify, rq1, rq2, rq3, external, report.
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from joblib import Parallel, delayed
from scipy import signal as sps
from scipy import stats
from scipy.spatial import cKDTree

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------------------------------------------------------
# Configuration. The primary configuration is fixed up front so nothing gets
# tuned on the test results.
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIG = OUT / "figures"

DATASET_DIRS = {
    "tpehg": ROOT / "term-preterm-ehg-database-1.0.1" / "term-preterm-ehg-database-1.0.1" / "tpehgdb",
    "tpehgt": ROOT / "term-preterm-ehg-dataset-with-tocogram-1.0.0",
    "ice": ROOT / "icelandic-ehg-dataset",
}
_icehg = sorted(ROOT.glob("icehg-ds*")) + sorted(ROOT.glob("*induced*cesarean*"))  # optional dataset
if _icehg:
    DATASET_DIRS["icehg"] = _icehg[0]

LJUBLJANA = ("tpehg", "tpehgt", "icehg")  # these may be pooled (cohort B)
DATASET_LABEL = {"tpehg": "TPEHG DB", "tpehgt": "TPEHGT DS", "icehg": "ICEHG DS", "ice": "EHGDB (Iceland)"}

FS_TARGET = 20.0          # Hz, the Ljubljana sampling rate
TRIM_S = 180.0            # seconds dropped from each end after filtering
MAX_DURATION_S = 1800.0   # Icelandic records cut to 30 min
MIN_DURATION_S = 600.0    # need at least 10 min left after trimming
PRETERM_WEEKS = 37.0
EARLY_WEEKS = 26.0

BANDS = {"0.08-4": (0.08, 4.0), "0.3-3": (0.3, 3.0), "0.3-4": (0.3, 4.0)}
CHANNELS = ["S1", "S2", "S3"]
CHANNEL_OPTIONS = CHANNELS + ["all"]       # "all" = features of all three channels
CORE_FEATURES = ["rms", "f_median", "f_peak", "sampen"]
EXTENDED_FEATURES = CORE_FEATURES + [
    "skewness", "kurtosis", "spectral_entropy", "zero_cross_rate",
    "hjorth_mobility", "hjorth_complexity", "line_length", "lf_hf_ratio",
]
FEATURE_SETS = {"core": CORE_FEATURES, "extended": EXTENDED_FEATURES}
SAMPEN_M, SAMPEN_R = 3, 0.15               # as in Fele-Zorz et al. 2008
WELCH_NPERSEG = 1024                       # 51 s windows, 0.02 Hz bins

CLINICAL_FEATURES = [
    "age", "parity", "abortions", "weight_kg", "hypertension", "diabetes",
    "placenta_anterior", "bleeding_t1", "bleeding_t2", "funneling", "smoker",
]
GA_FEATURE = ["ga_recording_wk"]

MODELS = ["logreg", "rf", "svm"]
SAMPLERS = ["none", "class_weight", "smote", "adasyn"]
PERIODS = ["all", "early", "late"]

PRIMARY = dict(band="0.3-4", channel="S3", feature_set="core", period="all",
               model="logreg", sampler="smote")

SEED = 304
N_REPEATS = 10      # repeats of grouped 5-fold CV
N_SPLITS = 5
N_PERM = 1000       # label shuffles for the permutation test
N_BOOT = 2000       # bootstrap resamples for AUROC CIs
N_JOBS = -1

# Plot style: colour-blind-safe palette in a fixed order, light grid, dark text.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def log(msg: str) -> None:
    print(msg, flush=True)


# -----------------------------------------------------------------------------
# 1. Header parsing -> one metadata schema for every dataset
# -----------------------------------------------------------------------------
META_COLUMNS = [
    "record_id", "dataset", "pregnancy_id", "rec_type",
    "ga_delivery_wk", "ga_recording_wk",
    "age", "parity", "abortions", "weight_kg", "bmi",
    "hypertension", "diabetes", "placenta_anterior", "bleeding_t1", "bleeding_t2",
    "funneling", "smoker", "prev_caesarean", "delivery_mode", "pair_record_num", "n_samples", "fs",
]

_MISSING = {"none", "n/a", "na", "", "nan"}


def _num(v) -> float:
    """Float from a header value; None/N/A -> NaN."""
    if v is None or str(v).strip().lower() in _MISSING:
        return np.nan
    try:
        return float(str(v).strip())
    except ValueError:
        return np.nan


def _yes_no(v) -> float:
    s = str(v).strip().lower()
    if s in _MISSING:
        return np.nan
    if s in {"yes", "y", "true", "positive", "t"}:
        return 1.0
    if s in {"no", "n", "false", "negative", "f"}:
        return 0.0
    return np.nan


def _placenta_anterior(v) -> float:
    """TPEHG says front/end, Iceland says Anterior/Posterior/Fundus. Anterior -> 1."""
    s = str(v).strip().lower()
    if s in _MISSING:
        return np.nan
    if s.startswith("front") or s.startswith("anterior"):
        return 1.0
    if s.startswith("end") or s.startswith("posterior") or s.startswith("fundus") or s.startswith("lateral"):
        return 0.0
    return np.nan


def _weeks_days(v) -> float:
    """'40/4' -> 40.57 weeks."""
    s = str(v).strip()
    if s.lower() in _MISSING:
        return np.nan
    if "/" in s:
        w, d = s.split("/")[:2]
        return _num(w) + _num(d) / 7.0
    return _num(s)


def _comment_pairs(comments: list[str], sep: str | None) -> dict[str, str]:
    """Header comment lines -> {key: value}. Ljubljana uses 'Key value', Iceland 'Key:value'."""
    out = {}
    for line in comments:
        line = line.strip().lstrip("#").strip()
        if not line:
            continue
        if sep == ":":
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
        else:
            parts = line.split(None, 1)
            if len(parts) == 1:
                continue
            k, v = parts
        out[k.strip().lower()] = v.strip()
    return out


def parse_ljubljana_header(hea: Path, dataset: str) -> dict:
    """TPEHG DB, TPEHGT DS and ICEHG DS share the same comment fields.
    ICEHG's newborn weight is ignored on purpose (not known before birth)."""
    hdr = wfdb.rdheader(str(hea.with_suffix("")))
    kv = _comment_pairs(hdr.comments, sep=None)
    return {
        "record_id": hea.stem,
        "dataset": dataset,
        "pregnancy_id": None,                         # set in assign_pregnancy_ids
        "rec_type": kv.get("rectype", "Pregnancy"),  # TPEHGT: Preterm / Term / Non-pregnant
        "ga_delivery_wk": _num(kv.get("gestation")),
        "ga_recording_wk": _num(kv.get("rectime")),
        "age": _num(kv.get("age")),
        "parity": _num(kv.get("parity")),
        "abortions": _num(kv.get("abortions")),
        "weight_kg": _num(kv.get("weight")),
        "bmi": np.nan,
        "hypertension": _yes_no(kv.get("hypertension")),
        "diabetes": _yes_no(kv.get("diabetes")),
        "placenta_anterior": _placenta_anterior(kv.get("placental_position")),
        "bleeding_t1": _yes_no(kv.get("bleeding_first_trimester")),
        "bleeding_t2": _yes_no(kv.get("bleeding_second_trimester")),
        "funneling": _yes_no(kv.get("funneling")),
        "smoker": _yes_no(kv.get("smoker")),
        "prev_caesarean": np.nan,
        "delivery_mode": np.nan,
        "pair_record_num": _num(kv.get("pair_recid")),   # ICEHG: the other recording of this pregnancy
        "n_samples": hdr.sig_len,
        "fs": float(hdr.fs),
    }


def parse_icelandic_header(hea: Path) -> dict:
    """Icelandic headers vary slightly in key spelling, so match on lower-cased prefixes."""
    hdr = wfdb.rdheader(str(hea.with_suffix("")))
    kv = _comment_pairs(hdr.comments, sep=":")

    def get(prefix: str) -> str | None:
        for k, v in kv.items():
            if k.startswith(prefix):
                return v
        return None

    return {
        "record_id": hea.stem,
        "dataset": "ice",
        "pregnancy_id": get("id"),                     # participant ID, e.g. ice002
        "rec_type": (get("record type") or "").lower(),  # pregnancy / labour
        "ga_delivery_wk": _weeks_days(get("gestational age at delivery")),
        "ga_recording_wk": _weeks_days(get("gestational age at recording")),
        "age": _num(get("age")),
        "parity": _num(get("parity")),
        "abortions": np.nan,
        "weight_kg": np.nan,
        "bmi": _num(get("bmi at recording")),
        "hypertension": np.nan,
        "diabetes": np.nan,
        "placenta_anterior": _placenta_anterior(get("placental position")),
        "bleeding_t1": np.nan,
        "bleeding_t2": np.nan,
        "funneling": np.nan,
        "smoker": np.nan,
        "prev_caesarean": _yes_no(get("previous caesarean")),
        "delivery_mode": get("mode of delivery"),
        "pair_record_num": np.nan,
        "n_samples": hdr.sig_len,
        "fs": float(hdr.fs),
    }


_ICEHG_PATHS: dict[str, Path] = {}   # plain dict cache; lru_cache wrappers don't survive joblib pickling


def _icehg_paths() -> dict[str, Path]:
    """ICEHG records live in six subfolders (plus a MATLAB copy we ignore); RECORDS lists the real ones."""
    if not _ICEHG_PATHS:
        d = DATASET_DIRS["icehg"]
        rel = [l.strip() for l in (d / "RECORDS").read_text().splitlines() if l.strip()]
        _ICEHG_PATHS.update({Path(r).name: d / r for r in sorted(rel)})
    return _ICEHG_PATHS


def record_path(record_id: str, dataset: str) -> Path:
    """Path to a record without extension, as wfdb wants it."""
    if dataset == "icehg":
        return _icehg_paths()[record_id]
    return DATASET_DIRS[dataset] / record_id


def list_records(dataset: str) -> list[Path]:
    d = DATASET_DIRS.get(dataset)
    if d is None or not d.exists():
        return []
    if dataset == "icehg":
        return [p.with_suffix(".hea") for p in _icehg_paths().values()]
    return sorted(d.glob("*.hea"))


# -----------------------------------------------------------------------------
# 2. Cleaning and harmonisation
# -----------------------------------------------------------------------------
@dataclass
class ExclusionLog:
    """Every dropped record and why; ends up in outputs/exclusions.csv."""
    rows: list[dict] = field(default_factory=list)

    def add(self, dataset: str, record_id: str, reason: str) -> None:
        self.rows.append({"dataset": dataset, "record_id": record_id, "reason": reason})

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=["dataset", "record_id", "reason"])


# TPEHGT has no participant IDs or pair field, but repeat recordings of the same
# woman share every clinical value. Matching on these columns recovers the 18
# pregnancies behind its 26 records.
DUP_KEY_WITHIN = ["ga_delivery_wk", "age", "parity", "abortions", "weight_kg", "placenta_anterior", "smoker"]


def _dup_key(row: pd.Series, cols: list[str], floor: bool = False) -> tuple | None:
    vals = []
    for c in cols:
        v = row[c]
        if pd.isna(v):
            return None
        vals.append(int(np.floor(v)) if floor else float(v))
    return tuple(vals)


def _rec_num(record_id: str) -> int | None:
    """tpehg1007 -> 1007, icehg1089 -> 1089 (same hospital numbering); None for other datasets."""
    m = re.search(r"(\d+)$", record_id)
    return int(m.group(1)) if m and record_id.startswith(("tpehg1", "tpehg2", "tpehg3", "tpehg4", "tpehg5",
                                                          "tpehg6", "tpehg7", "tpehg8", "tpehg9", "icehg")) else None


def assign_pregnancy_ids(meta: pd.DataFrame, pair_edges: list[tuple[int, int]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """TPEHG DB + ICEHG DS: records are numbered from one hospital pool and ICEHG's
    Pair_RecID says which two numbers belong to the same pregnancy, so link
    numbers with union-find and name the pregnancy after the smallest number.
    Iceland: participant ID from the header. TPEHGT: identical clinical values ->
    same pregnancy, and a floored match to a TPEHG DB record adopts that ID."""
    meta = meta.copy()

    parent: dict[int, int] = {}

    def find(a: int) -> int:
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for a, b in pair_edges:
        union(a, b)
    pooled = meta.dataset.isin(["tpehg", "icehg"])
    meta.loc[pooled, "pregnancy_id"] = [f"ljub{find(_rec_num(r))}" for r in meta.loc[pooled, "record_id"]]

    for ds in ("tpehgt",):
        sub = meta[meta.dataset == ds]
        if sub.empty:
            continue
        key_to_id: dict[tuple, str] = {}
        n = 0
        for idx, row in sub.iterrows():
            key = _dup_key(row, DUP_KEY_WITHIN)
            if key is None or key not in key_to_id:
                n += 1
                pid = f"{ds}_preg{n:03d}"
                if key is not None:
                    key_to_id[key] = pid
            else:
                pid = key_to_id[key]
            meta.at[idx, "pregnancy_id"] = pid

    # Floor the weeks here: TPEHGT reports whole weeks, TPEHG DB decimals.
    cross_cols = ["ga_delivery_wk", "ga_recording_wk"] + DUP_KEY_WITHIN[1:]
    ref = {}
    for _, row in meta[meta.dataset == "tpehg"].iterrows():
        key = _dup_key(row, cross_cols, floor=True)
        if key is not None:
            ref.setdefault(key, row["pregnancy_id"])
    matches = []
    for idx, row in meta[meta.dataset == "tpehgt"].iterrows():
        key = _dup_key(row, cross_cols, floor=True)
        if key is not None and key in ref:
            matches.append({"record_id": row["record_id"], "dataset": row["dataset"],
                            "matched_tpehg_pregnancy": ref[key], "old_pregnancy_id": row["pregnancy_id"]})
            meta.at[idx, "pregnancy_id"] = ref[key]
    return meta, pd.DataFrame(matches, columns=["record_id", "dataset", "matched_tpehg_pregnancy", "old_pregnancy_id"])


def pregnancy_groups(meta: pd.DataFrame) -> pd.DataFrame:
    """Pregnancies (outside Iceland) that have more than one recording, with the datasets involved."""
    g = meta[meta.dataset != "ice"].groupby("pregnancy_id").agg(
        n_recordings=("record_id", "size"), record_ids=("record_id", lambda s: " ".join(s)),
        datasets=("dataset", lambda s: ", ".join(sorted({DATASET_LABEL[d] for d in s}))))
    return g[g.n_recordings > 1].reset_index()


def build_metadata(exclog: ExclusionLog) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse all headers, drop non-pregnant / labour / missing-GA records, relabel."""
    rows = []
    for ds in DATASET_DIRS:
        heas = list_records(ds)
        log(f"  {DATASET_LABEL.get(ds, ds):16s}: {len(heas):4d} header files in {DATASET_DIRS[ds]}")
        for hea in heas:
            rows.append(parse_icelandic_header(hea) if ds == "ice" else parse_ljubljana_header(hea, ds))
    if "icehg" not in DATASET_DIRS:
        log("  ICEHG DS       : not found in repository -> cohort B = TPEHG DB + TPEHGT DS")
    meta = pd.DataFrame(rows, columns=META_COLUMNS)

    # ICEHG DS re-publishes six TPEHG DB recordings under the same number (the
    # signals are identical); keep the TPEHG DB copy. Pair links are collected
    # from every ICEHG header first, dropped copies included.
    tpehg_nums = {_rec_num(r) for r in meta.loc[meta.dataset == "tpehg", "record_id"]}
    pair_edges = [(_rec_num(r), int(p)) for r, p in zip(meta.record_id, meta.pair_record_num)
                  if pd.notna(p) and _rec_num(r) is not None]

    keep = np.ones(len(meta), bool)
    for i, row in meta.iterrows():
        rt = str(row.rec_type).lower()
        if row.dataset == "icehg" and _rec_num(row.record_id) in tpehg_nums:
            exclog.add(row.dataset, row.record_id, "duplicate of a TPEHG DB recording (identical signal)"); keep[i] = False
        elif row.dataset == "tpehgt" and rt.startswith("non"):
            exclog.add(row.dataset, row.record_id, "non-pregnant recording"); keep[i] = False
        elif row.dataset == "ice" and rt.startswith("lab"):
            exclog.add(row.dataset, row.record_id, "labour recording"); keep[i] = False
        elif pd.isna(row.ga_delivery_wk):
            exclog.add(row.dataset, row.record_id, "missing gestational age at delivery"); keep[i] = False
        elif pd.isna(row.ga_recording_wk):
            exclog.add(row.dataset, row.record_id, "missing gestational age at recording"); keep[i] = False
    meta = meta[keep].reset_index(drop=True)

    # Recompute the label from GA at delivery so it means the same thing everywhere.
    meta["preterm"] = (meta.ga_delivery_wk < PRETERM_WEEKS).astype(int)
    meta["period"] = np.where(meta.ga_recording_wk < EARLY_WEEKS, "early", "late")

    meta, matches = assign_pregnancy_ids(meta, pair_edges)
    return meta, matches


# -----------------------------------------------------------------------------
# 3. Signals, pre-processing, features
# -----------------------------------------------------------------------------
_RAW_LJUBLJANA = re.compile(r"^(EHG|S)?([123])$")   # '1', 'EHG1' or 'S1'; skips the pre-filtered copies
ICE_MEDIAN_COLUMN = ["EHG9", "EHG10", "EHG11", "EHG12"]  # midline column of the 4x4 grid


def load_bipolar(record_id: str, dataset: str) -> tuple[np.ndarray, float]:
    """Return the three raw bipolar channels (n x 3, mV) and the sampling rate.

    Iceland is 16 monopolar electrodes at 200 Hz, so we build three vertical
    bipolar pairs down the midline (9-10, 10-11, 11-12), decimate to 20 Hz with
    anti-aliasing, and keep the first 30 min. Different device and electrode
    spacing, which is why that cohort is only used for external validation."""
    rec = wfdb.rdrecord(str(record_path(record_id, dataset)))
    names, x = rec.sig_name, rec.p_signal
    if dataset == "ice":
        idx = [names.index(n) for n in ICE_MEDIAN_COLUMN]
        mono = x[:, idx]
        bip = np.column_stack([mono[:, 0] - mono[:, 1], mono[:, 1] - mono[:, 2], mono[:, 2] - mono[:, 3]])
        q = int(round(rec.fs / FS_TARGET))
        bip = sps.resample_poly(bip, up=1, down=q, axis=0)
        bip = bip[: int(MAX_DURATION_S * FS_TARGET)]
        return bip, FS_TARGET
    cols = {}
    for i, n in enumerate(names):
        m = _RAW_LJUBLJANA.match(n)
        if m:
            cols[int(m.group(2))] = i
    sig = x[:, [cols[1], cols[2], cols[3]]]
    return sig, float(rec.fs)


def bandpass_and_trim(sig: np.ndarray, fs: float, band: tuple[float, float]) -> np.ndarray:
    """Zero-phase 4th-order Butterworth band-pass, then drop 180 s at each end (filter transients)."""
    sos = sps.butter(4, band, btype="bandpass", fs=fs, output="sos")
    y = sps.sosfiltfilt(sos, sig, axis=0)
    k = int(TRIM_S * fs)
    return y[k:-k] if len(y) > 2 * k else y[:0]


def sample_entropy(x: np.ndarray, m: int = SAMPEN_M, r: float = SAMPEN_R) -> float:
    """SampEn(m, r*SD). Template matches are counted with a KD-tree (Chebyshev
    distance), which is exact and far faster than the O(N^2) loop at ~29k samples."""
    x = np.asarray(x, float)
    tol = r * x.std()
    if tol == 0 or len(x) <= m + 1:
        return np.nan

    def matches(mm: int) -> int:
        emb = np.lib.stride_tricks.sliding_window_view(x, mm)
        tree = cKDTree(emb)
        return tree.count_neighbors(tree, tol, p=np.inf) - len(emb)  # minus self-matches

    b, a = matches(m), matches(m + 1)
    return float(-np.log(a / b)) if a > 0 and b > 0 else np.nan


def spectral_features(x: np.ndarray, fs: float, band: tuple[float, float]) -> dict:
    """Welch PSD within the pass-band -> median freq, peak freq, spectral entropy, LF/HF ratio."""
    f, p = sps.welch(x, fs=fs, nperseg=min(WELCH_NPERSEG, len(x)), window="hann")
    sel = (f >= band[0]) & (f <= band[1])
    f, p = f[sel], p[sel]
    total = p.sum()
    if total <= 0:
        return {"f_median": np.nan, "f_peak": np.nan, "spectral_entropy": np.nan, "lf_hf_ratio": np.nan}
    cdf = np.cumsum(p) / total
    pn = p / total
    mid = min(1.0, max(band[0], 1.0))  # LF/HF split at 1 Hz
    lf, hf = p[f < mid].sum(), p[f >= mid].sum()
    return {
        "f_median": float(f[np.searchsorted(cdf, 0.5)]),
        "f_peak": float(f[np.argmax(p)]),
        "spectral_entropy": float(-(pn[pn > 0] * np.log2(pn[pn > 0])).sum() / np.log2(len(pn))),
        "lf_hf_ratio": float(lf / hf) if hf > 0 else np.nan,
    }


def channel_features(x: np.ndarray, fs: float, band: tuple[float, float]) -> dict:
    """Core features plus the extended set (only the RQ3 'extended' arm uses the latter)."""
    feats = {"rms": float(np.sqrt(np.mean(x ** 2)))}
    feats.update(spectral_features(x, fs, band))
    feats["sampen"] = sample_entropy(x)
    feats["skewness"] = float(stats.skew(x))
    feats["kurtosis"] = float(stats.kurtosis(x))
    feats["zero_cross_rate"] = float(np.mean(np.diff(np.signbit(x)) != 0) * fs)
    dx, ddx = np.diff(x), np.diff(x, n=2)
    v0, v1, v2 = x.var(), dx.var(), ddx.var()
    mob = np.sqrt(v1 / v0) if v0 > 0 else np.nan
    feats["hjorth_mobility"] = float(mob)
    feats["hjorth_complexity"] = float(np.sqrt(v2 / v1) / mob) if v1 > 0 and mob else np.nan
    feats["line_length"] = float(np.abs(dx).sum() / (len(x) / fs))
    return feats


def extract_record(record_id: str, dataset: str) -> list[dict]:
    """Features for one record, all bands and channels. Empty list if too short after trimming."""
    sig, fs = load_bipolar(record_id, dataset)
    out = []
    for band_name, band in BANDS.items():
        y = bandpass_and_trim(sig, fs, band)
        if len(y) < MIN_DURATION_S * fs:
            return []
        for ci, ch in enumerate(CHANNELS):
            row = {"record_id": record_id, "dataset": dataset, "band": band_name, "channel": ch,
                   "duration_s": len(y) / fs}
            row.update(channel_features(y[:, ci], fs, band))
            out.append(row)
    return out


def extract_all(meta: pd.DataFrame, exclog: ExclusionLog, n_jobs: int = N_JOBS) -> pd.DataFrame:
    """Run extract_record over every record in parallel. Features are computed once, before any CV."""
    jobs = list(zip(meta.record_id, meta.dataset))
    results = Parallel(n_jobs=n_jobs, verbose=0)(delayed(extract_record)(r, d) for r, d in jobs)
    rows = []
    for (rid, ds), res in zip(jobs, results):
        if not res:
            exclog.add(ds, rid, f"shorter than {MIN_DURATION_S/60:.0f} min after trimming")
        rows.extend(res)
    return pd.DataFrame(rows)


def feature_matrix(feats_long: pd.DataFrame, band: str, channel: str, feature_set: str) -> pd.DataFrame:
    """One row per record for the given band / channel / feature set. channel='all' concatenates S1-S3."""
    cols = FEATURE_SETS[feature_set]
    sub = feats_long[feats_long.band == band]
    chans = CHANNELS if channel == "all" else [channel]
    parts = []
    for ch in chans:
        p = sub[sub.channel == ch].set_index("record_id")[cols]
        parts.append(p.add_suffix(f"_{ch}") if channel == "all" else p)
    return pd.concat(parts, axis=1)


# -----------------------------------------------------------------------------
# 3b. Check our features against the ones published with TPEHG DB
# -----------------------------------------------------------------------------
def read_fvl(path: Path) -> pd.DataFrame:
    """PhysioNet .fvl table: pipe-separated with a dashed rule on line 2."""
    df = pd.read_csv(path, sep="|", skiprows=[1], skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df["Record"] = df["Record"].str.strip()
    return df


def verify_features(feats_long: pd.DataFrame) -> pd.DataFrame:
    """Pearson r, median relative error and share within 5 % vs the published values.
    Published RMS is in uV, ours in mV."""
    fvl_dir = DATASET_DIRS["tpehg"].parent
    fvl_files = {"0.08-4": "tpehgdb_features__filter_0.08_Hz-4.0_Hz.fvl",
                 "0.3-3": "tpehgdb_features__filter_0.3_Hz-3.0_Hz.fvl",
                 "0.3-4": "tpehgdb_features__filter_0.3_Hz-4.0_Hz.fvl"}
    pairs = {"rms": "RMS", "f_median": "Fmed", "f_peak": "Fpeak", "sampen": "Samp. en."}
    rows = []
    ours = feats_long[feats_long.dataset == "tpehg"].copy()
    ours["chan_num"] = ours.channel.str[1].astype(int)
    for band, fname in fvl_files.items():
        pub = read_fvl(fvl_dir / fname)
        m = ours[ours.band == band].merge(pub, left_on=["record_id", "chan_num"], right_on=["Record", "Chann"])
        for ch in (1, 2, 3):
            mm = m[m.chan_num == ch]
            for mine, theirs in pairs.items():
                a = mm[mine].to_numpy() * (1000.0 if mine == "rms" else 1.0)
                b = mm[theirs].to_numpy(float)
                ok = np.isfinite(a) & np.isfinite(b) & (b != 0)
                rel = np.abs(a[ok] - b[ok]) / np.abs(b[ok])
                rows.append({"band": band, "channel": f"S{ch}", "feature": mine, "n": int(ok.sum()),
                             "pearson_r": float(np.corrcoef(a[ok], b[ok])[0, 1]),
                             "median_rel_error": float(np.median(rel)),
                             "frac_within_5pct": float(np.mean(rel < 0.05))})
    return pd.DataFrame(rows)


def fig_verification(feats_long: pd.DataFrame) -> None:
    pub = read_fvl(DATASET_DIRS["tpehg"].parent / "tpehgdb_features__filter_0.3_Hz-4.0_Hz.fvl")
    ours = feats_long[(feats_long.dataset == "tpehg") & (feats_long.band == "0.3-4")].copy()
    ours["Chann"] = ours.channel.str[1].astype(int)
    m = ours.merge(pub, left_on=["record_id", "Chann"], right_on=["Record", "Chann"])
    pairs = [("rms", "RMS", "RMS (uV)"), ("f_median", "Fmed", "Median frequency (Hz)"),
             ("f_peak", "Fpeak", "Peak frequency (Hz)"), ("sampen", "Samp. en.", "Sample entropy")]
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.9))
    for ax, (mine, theirs, label) in zip(axes, pairs):
        a = m[mine] * (1000 if mine == "rms" else 1)
        b = m[theirs].astype(float)
        lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.8, ls="--", zorder=1)
        ax.scatter(b, a, s=9, color=PALETTE[0], alpha=0.6, lw=0, zorder=2)
        r = np.corrcoef(a, b)[0, 1]
        ax.set_title(label)
        ax.set_xlabel("published (TPEHG DB)")
        ax.text(0.04, 0.93, f"r = {r:.3f}", transform=ax.transAxes, color=INK2, va="top")
    axes[0].set_ylabel("this pipeline")
    fig.suptitle("Feature verification, 0.3-4 Hz, all channels, 300 records x 3 channels", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig_feature_verification.png")
    plt.close(fig)


# -----------------------------------------------------------------------------
# 4. Modelling helpers shared by RQ1-3 and the external validation
# -----------------------------------------------------------------------------
def make_pipeline(model: str, sampler: str, seed: int) -> ImbPipeline:
    """impute -> scale -> (oversample) -> classifier, all fitted on the training fold only.
    imblearn's Pipeline skips the sampler at predict time, which is the leak-free way."""
    cw = "balanced" if sampler == "class_weight" else None
    if model == "logreg":
        clf = LogisticRegression(C=1.0, max_iter=5000, class_weight=cw, random_state=seed)
    elif model == "rf":
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, max_features="sqrt",
                                     class_weight=cw, random_state=seed, n_jobs=1)
    elif model == "svm":
        clf = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight=cw, random_state=seed)
    else:
        raise ValueError(model)
    steps = [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    if sampler == "smote":
        steps.append(("sampler", SMOTE(k_neighbors=5, random_state=seed)))
    elif sampler == "adasyn":
        steps.append(("sampler", ADASYN(n_neighbors=5, random_state=seed)))
    steps.append(("clf", clf))
    return ImbPipeline(steps)


def scores_of(pipe: ImbPipeline, X: np.ndarray) -> np.ndarray:
    """Probability where available, otherwise the SVM decision function (fine for AUROC)."""
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    return pipe.decision_function(X)


def _fit(pipe: ImbPipeline, X, y, sampler: str, seed: int) -> tuple[ImbPipeline, bool]:
    """ADASYN sometimes refuses to generate samples in a fold; fall back to SMOTE and flag it."""
    try:
        return pipe.fit(X, y), False
    except (RuntimeError, ValueError):
        if sampler != "adasyn":
            raise
        return make_pipeline(_model_name(pipe), "smote", seed).fit(X, y), True


def _model_name(pipe: ImbPipeline) -> str:
    return {"LogisticRegression": "logreg", "RandomForestClassifier": "rf", "SVC": "svm"}[pipe.named_steps["clf"].__class__.__name__]


def bootstrap_auroc_ci(y: np.ndarray, s: np.ndarray, groups: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    """Percentile 95 % CI, resampling pregnancies (not recordings) with replacement."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in ug}
    aucs = []
    for _ in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([members[g] for g in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], s[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def grouped_cv(X: pd.DataFrame | np.ndarray, y: np.ndarray, groups: np.ndarray, model: str, sampler: str,
               n_repeats: int = N_REPEATS, n_splits: int = N_SPLITS, seed: int = SEED, n_boot: int = N_BOOT) -> dict:
    """Repeated stratified group K-fold. Recordings from one pregnancy never straddle
    train and test. AUROC per repeat; out-of-fold scores averaged over repeats
    for the ROC curve and bootstrap CI."""
    X = np.asarray(X, float)
    n_minority_groups = len(np.unique(groups[y == 1]))
    k = max(2, min(n_splits, n_minority_groups))     # can't have more folds than preterm pregnancies
    oof = np.zeros((n_repeats, len(y)))
    aucs, fallbacks = [], 0
    for r in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed + r)
        for tr, te in cv.split(X, y, groups):
            pipe = make_pipeline(model, sampler, seed + r)
            pipe, fb = _fit(pipe, X[tr], y[tr], sampler, seed + r)
            fallbacks += fb
            oof[r, te] = scores_of(pipe, X[te])
        aucs.append(roc_auc_score(y, oof[r]))
    mean_oof = oof.mean(axis=0)
    lo, hi = bootstrap_auroc_ci(y, mean_oof, groups, n_boot, seed)
    return {"auroc_mean": float(np.mean(aucs)), "auroc_sd": float(np.std(aucs, ddof=1)) if n_repeats > 1 else 0.0,
            "auroc_pooled": float(roc_auc_score(y, mean_oof)), "ci_low": lo, "ci_high": hi,
            "n_splits": k, "adasyn_fallbacks": fallbacks, "oof": mean_oof, "aucs": aucs}


def leaky_cv(X, y, model: str, seed: int = SEED, n_splits: int = N_SPLITS) -> float:
    """The protocol Vandewiele et al. criticised: SMOTE on the whole dataset before
    splitting, folds ignoring pregnancy. Only here to quantify the inflation."""
    X = np.asarray(X, float)
    Xi = SimpleImputer(strategy="median").fit_transform(X)
    Xs, ys = SMOTE(k_neighbors=5, random_state=seed).fit_resample(Xi, y)
    oof = np.zeros(len(ys))
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(Xs, ys):
        pipe = make_pipeline(model, "none", seed).fit(Xs[tr], ys[tr])
        oof[te] = scores_of(pipe, Xs[te])
    return float(roc_auc_score(ys, oof))


def permutation_test(X, y, groups, model: str, sampler: str, n_perm: int, seed: int = SEED) -> dict:
    """Shuffle labels between pregnancies, re-run grouped CV each time.
    p = (1 + #null >= observed) / (1 + n_perm)."""
    X = np.asarray(X, float)
    observed = grouped_cv(X, y, groups, model, sampler, n_repeats=1, n_boot=10, seed=seed)["auroc_mean"]
    ug = np.unique(groups)
    glabel = np.array([y[groups == g][0] for g in ug])

    def one(i: int) -> float:
        perm = np.random.default_rng(seed + 1000 + i).permutation(glabel)
        yp = np.array([perm[np.searchsorted(ug, g)] for g in groups])
        return grouped_cv(X, yp, groups, model, sampler, n_repeats=1, n_boot=10, seed=seed + i)["auroc_mean"]

    null = np.array(Parallel(n_jobs=N_JOBS)(delayed(one)(i) for i in range(n_perm)))
    p = (1 + np.sum(null >= observed)) / (1 + n_perm)
    return {"observed": observed, "null": null, "p_value": float(p)}


# -----------------------------------------------------------------------------
# 5. Cohorts and design matrices
# -----------------------------------------------------------------------------
def cohort(meta: pd.DataFrame, name: str) -> pd.DataFrame:
    """A = TPEHG DB, B = pooled Ljubljana, C = Icelandic pregnancy recordings."""
    if name == "A":
        return meta[meta.dataset == "tpehg"]
    if name == "B":
        return meta[meta.dataset.isin(LJUBLJANA)]
    if name == "C":
        return meta[meta.dataset == "ice"]
    raise ValueError(name)


def design(meta_sub: pd.DataFrame, feats_long: pd.DataFrame, band: str, channel: str, feature_set: str,
           space: str = "ehg", period: str = "all") -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    """X, y, groups (and the metadata rows) for one configuration.
    space is one of ga | clinical | ga+clinical | ehg | ehg+clinical+ga."""
    m = meta_sub if period == "all" else meta_sub[meta_sub.period == period]
    ehg = feature_matrix(feats_long, band, channel, feature_set)
    m = m[m.record_id.isin(ehg.index)].set_index("record_id")
    blocks = []
    if "ehg" in space:
        blocks.append(ehg.loc[m.index])
    if "clinical" in space:
        blocks.append(m[CLINICAL_FEATURES])
    if "ga" in space:
        blocks.append(m[GA_FEATURE])
    X = pd.concat(blocks, axis=1)
    return X, m.preterm.to_numpy(int), m.pregnancy_id.to_numpy(), m


def cohort_summary(meta: pd.DataFrame, feats_long: pd.DataFrame) -> pd.DataFrame:
    kept = meta[meta.record_id.isin(feats_long.record_id.unique())]
    rows = []
    for name, purpose in [("A", "TPEHG DB: leak-free benchmark vs. Vandewiele et al."),
                          ("B", "pooled Ljubljana: main analysis"),
                          ("C", "EHGDB (Iceland): external validation")]:
        c = cohort(kept, name)
        preg = c.groupby("pregnancy_id").preterm.max()
        rows.append({"cohort": name, "datasets": ", ".join(DATASET_LABEL[d] for d in c.dataset.unique()),
                     "recordings": len(c), "pregnancies": c.pregnancy_id.nunique(),
                     "term_pregnancies": int((preg == 0).sum()), "preterm_pregnancies": int((preg == 1).sum()),
                     "preterm_recordings": int(c.preterm.sum()),
                     "early_recordings": int((c.period == "early").sum()), "late_recordings": int((c.period == "late").sum()),
                     "purpose": purpose})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 6. RQ1: models, imbalance handling, leakage, permutation test
# -----------------------------------------------------------------------------
def run_rq1(meta: pd.DataFrame, feats_long: pd.DataFrame, n_repeats: int, n_perm: int) -> dict:
    P = PRIMARY
    results, oof_store = [], {}
    for coh in ("A", "B"):
        X, y, g, _ = design(cohort(meta, coh), feats_long, P["band"], P["channel"], P["feature_set"])
        log(f"  cohort {coh}: {len(y)} recordings, {len(np.unique(g))} pregnancies, {y.sum()} preterm recordings")
        for model in MODELS:
            for sampler in SAMPLERS:
                res = grouped_cv(X, y, g, model, sampler, n_repeats=n_repeats)
                oof_store[(coh, model, sampler)] = (y, res["oof"], g)
                results.append({"cohort": coh, "model": model, "sampler": sampler,
                                **{k: v for k, v in res.items() if k not in ("oof", "aucs")}})
                log(f"    {model:7s} {sampler:13s} AUROC {res['auroc_mean']:.3f} "
                    f"[{res['ci_low']:.3f}, {res['ci_high']:.3f}]")
    rq1 = pd.DataFrame(results)
    rq1.to_csv(OUT / "rq1_model_comparison.csv", index=False)

    # Leaky protocol for comparison.
    XB, yB, gB, _ = design(cohort(meta, "B"), feats_long, P["band"], P["channel"], P["feature_set"])
    XA, yA, gA, _ = design(cohort(meta, "A"), feats_long, P["band"], P["channel"], P["feature_set"])
    leak_rows = []
    for coh, (X, y, g) in {"A": (XA, yA, gA), "B": (XB, yB, gB)}.items():
        for model in MODELS:
            clean = rq1[(rq1.cohort == coh) & (rq1.model == model) & (rq1.sampler == "smote")].auroc_mean.iloc[0]
            leaky = leaky_cv(X, y, model)
            leak_rows.append({"cohort": coh, "model": model, "leak_free_smote_auroc": clean, "leaky_auroc": leaky,
                              "inflation": leaky - clean})
            log(f"  leakage {coh} {model:7s}: leak-free {clean:.3f} vs leaky {leaky:.3f}")
    leak = pd.DataFrame(leak_rows)
    leak.to_csv(OUT / "rq1_leakage_comparison.csv", index=False)

    # Permutation test on the primary configuration, cohort B.
    log(f"  permutation test ({n_perm} shuffles, primary model on cohort B) ...")
    perm = permutation_test(XB, yB, gB, P["model"], P["sampler"], n_perm)
    log(f"  observed AUROC {perm['observed']:.3f}, null mean {perm['null'].mean():.3f}, p = {perm['p_value']:.4f}")
    np.save(OUT / "rq1_permutation_null.npy", perm["null"])
    with open(OUT / "rq1_permutation.json", "w") as f:
        json.dump({"observed_auroc": perm["observed"], "null_mean": float(perm["null"].mean()),
                   "null_95th": float(np.percentile(perm["null"], 95)), "p_value": perm["p_value"], "n_perm": n_perm}, f, indent=2)

    fig_rq1_models(rq1)
    fig_rq1_roc(oof_store)
    fig_rq1_permutation(perm)
    fig_rq1_leakage(leak)
    return {"rq1": rq1, "leak": leak, "perm": perm}


def _dot_ci(ax, x, m, lo, hi, color, label=None, marker="o"):
    """Point estimate with a thin CI whisker."""
    ax.plot([x, x], [lo, hi], color=color, lw=1.2, zorder=2)
    ax.plot(x, m, marker=marker, ms=6, color=color, mec="white", mew=0.8, ls="none", label=label, zorder=3)


SAMPLER_LABEL = {"none": "no correction", "class_weight": "class weights", "smote": "SMOTE", "adasyn": "ADASYN"}
MODEL_LABEL = {"logreg": "Logistic regression", "rf": "Random forest", "svm": "SVM (RBF)"}


def fig_rq1_models(rq1: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    offs = np.linspace(-0.27, 0.27, len(SAMPLERS))
    for ax, coh in zip(axes, ("A", "B")):
        d = rq1[rq1.cohort == coh]
        for mi, model in enumerate(MODELS):
            for si, sampler in enumerate(SAMPLERS):
                r = d[(d.model == model) & (d.sampler == sampler)].iloc[0]
                _dot_ci(ax, mi + offs[si], r.auroc_mean, r.ci_low, r.ci_high, PALETTE[si],
                        label=SAMPLER_LABEL[sampler] if mi == 0 else None)
                ax.text(mi + offs[si], r.ci_high + 0.012, f"{r.auroc_mean:.2f}", ha="center", va="bottom",
                        fontsize=6.5, color=INK2)
        ax.axhline(0.5, color=INK2, lw=0.8, ls="--")
        ax.set_xticks(range(len(MODELS)), [MODEL_LABEL[m] for m in MODELS])
        ax.set_title({"A": "Cohort A: TPEHG DB", "B": "Cohort B: pooled Ljubljana"}[coh])
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("AUROC (pregnancy-grouped CV)")
    axes[0].set_ylim(0.2, 1.0)
    axes[1].legend(loc="upper left", ncol=2, fontsize=8)
    fig.suptitle("RQ1: leak-free preterm vs. term classification, primary EHG features (0.3-4 Hz, S3)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig_rq1_model_comparison.png")
    plt.close(fig)


def fig_rq1_roc(oof_store: dict) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    for i, coh in enumerate(("A", "B")):
        y, s, _ = oof_store[(coh, PRIMARY["model"], PRIMARY["sampler"])]
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, color=PALETTE[i], lw=2, label=f"cohort {coh} (AUROC {roc_auc_score(y, s):.2f})")
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls="--")
    ax.set_xlabel("false-positive rate (term flagged as preterm)")
    ax.set_ylabel("sensitivity")
    ax.set_title(f"Primary model: {MODEL_LABEL[PRIMARY['model']]} + {SAMPLER_LABEL[PRIMARY['sampler']]}")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(FIG / "fig_rq1_roc_primary.png")
    plt.close(fig)


def fig_rq1_permutation(perm: dict) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3))
    ax.hist(perm["null"], bins=40, color=PALETTE[0], alpha=0.75, lw=0)
    ax.axvline(perm["observed"], color=PALETTE[1], lw=2)
    ax.set_xlim(right=max(perm["observed"] + 0.09, float(perm["null"].max()) + 0.02))
    ax.text(perm["observed"], ax.get_ylim()[1] * 0.95, f"  observed {perm['observed']:.3f}\n  p = {perm['p_value']:.3g}",
            color=INK, va="top", fontsize=8)
    ax.set_xlabel("AUROC with permuted labels (pregnancy-level shuffles)")
    ax.set_ylabel("count")
    ax.set_title(f"Permutation test, {len(perm['null'])} shuffles, cohort B")
    fig.tight_layout()
    fig.savefig(FIG / "fig_rq1_permutation.png")
    plt.close(fig)


def fig_rq1_leakage(leak: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    w = 0.36
    for ax, coh in zip(axes, ("A", "B")):
        d = leak[leak.cohort == coh].set_index("model").loc[MODELS]
        x = np.arange(len(MODELS))
        ax.bar(x - w / 2, d.leak_free_smote_auroc, w - 0.03, color=PALETTE[0], label="leak-free (SMOTE inside grouped CV)")
        ax.bar(x + w / 2, d.leaky_auroc, w - 0.03, color=PALETTE[1], label="leaky (SMOTE before splitting)")
        for xi, (a, b) in enumerate(zip(d.leak_free_smote_auroc, d.leaky_auroc)):
            ax.text(xi - w / 2, a + 0.01, f"{a:.2f}", ha="center", fontsize=7, color=INK2)
            ax.text(xi + w / 2, b + 0.01, f"{b:.2f}", ha="center", fontsize=7, color=INK2)
        ax.set_xticks(x, [MODEL_LABEL[m] for m in MODELS])
        ax.axhline(0.5, color=INK2, lw=0.8, ls="--")
        ax.set_title(f"Cohort {coh}")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("AUROC")
    axes[0].set_ylim(0.3, 1.05)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Data leakage inflates apparent performance", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig_rq1_leakage.png")
    plt.close(fig)


# -----------------------------------------------------------------------------
# 7. RQ2: does EHG add anything beyond clinical variables and GA?
# -----------------------------------------------------------------------------
SPACES = ["ga", "clinical", "ga+clinical", "ehg", "ehg+clinical+ga"]
SPACE_LABEL = {"ga": "GA at recording", "clinical": "clinical", "ga+clinical": "GA + clinical",
               "ehg": "EHG only", "ehg+clinical+ga": "EHG + clinical + GA"}


def run_rq2(meta: pd.DataFrame, feats_long: pd.DataFrame, n_repeats: int) -> pd.DataFrame:
    P = PRIMARY
    rows = []
    for coh in ("A", "B"):
        for period in PERIODS:
            for space in SPACES:
                X, y, g, _ = design(cohort(meta, coh), feats_long, P["band"], P["channel"], P["feature_set"],
                                    space=space, period=period)
                n_pre = len(np.unique(g[y == 1]))
                if n_pre < 3 or (y == 0).sum() < 3:
                    rows.append({"cohort": coh, "period": period, "space": space, "n": len(y), "n_preterm_preg": n_pre,
                                 "auroc_mean": np.nan, "ci_low": np.nan, "ci_high": np.nan})
                    continue
                res = grouped_cv(X, y, g, P["model"], P["sampler"], n_repeats=n_repeats)
                rows.append({"cohort": coh, "period": period, "space": space, "n": len(y), "n_preterm_preg": n_pre,
                             "n_features": X.shape[1], **{k: v for k, v in res.items() if k not in ("oof", "aucs")}})
                log(f"  {coh} {period:5s} {space:16s} n={len(y):3d} preterm preg={n_pre:2d} "
                    f"AUROC {res['auroc_mean']:.3f} [{res['ci_low']:.3f}, {res['ci_high']:.3f}]")
    rq2 = pd.DataFrame(rows)
    rq2.to_csv(OUT / "rq2_confounder_analysis.csv", index=False)
    fig_rq2(rq2)
    return rq2


def fig_rq2(rq2: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    for ax, period in zip(axes, PERIODS):
        for ci, coh in enumerate(("A", "B")):
            d = rq2[(rq2.period == period) & (rq2.cohort == coh)].set_index("space").loc[SPACES]
            x = np.arange(len(SPACES)) + (-0.15 if coh == "A" else 0.15)
            for xi, (_, r) in zip(x, d.iterrows()):
                if np.isnan(r.auroc_mean):
                    ax.text(xi, 0.52, "n/a", ha="center", fontsize=7, color=INK2)
                    continue
                _dot_ci(ax, xi, r.auroc_mean, r.ci_low, r.ci_high, PALETTE[ci])
            ax.plot([], [], "o", color=PALETTE[ci], label=f"cohort {coh}")
        n = rq2[(rq2.period == period) & (rq2.cohort == "B")].n.max()
        ax.set_title({"all": "all recordings", "early": "early (< 26 wk)", "late": "late (>= 26 wk)"}[period] + f", cohort B n={n}")
        ax.set_xticks(range(len(SPACES)), [SPACE_LABEL[s] for s in SPACES], rotation=25, ha="right")
        ax.axhline(0.5, color=INK2, lw=0.8, ls="--")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("AUROC (pregnancy-grouped CV)")
    axes[0].set_ylim(0.2, 1.0)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("RQ2: does EHG add to clinical variables and gestational age? (primary model)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig_rq2_feature_spaces.png")
    plt.close(fig)


# -----------------------------------------------------------------------------
# 8. RQ3: sensitivity to pre-processing choices (specification curve)
# -----------------------------------------------------------------------------
def run_rq3(meta: pd.DataFrame, feats_long: pd.DataFrame, n_repeats: int) -> pd.DataFrame:
    P = PRIMARY
    rows = []
    B = cohort(meta, "B")
    for band in BANDS:
        for channel in CHANNEL_OPTIONS:
            for fset in FEATURE_SETS:
                for period in PERIODS:
                    X, y, g, _ = design(B, feats_long, band, channel, fset, period=period)
                    n_pre = len(np.unique(g[y == 1]))
                    if n_pre < 3:
                        continue
                    res = grouped_cv(X, y, g, P["model"], P["sampler"], n_repeats=n_repeats)
                    rows.append({"band": band, "channel": channel, "feature_set": fset, "period": period,
                                 "n": len(y), "n_features": X.shape[1],
                                 "primary": band == P["band"] and channel == P["channel"] and fset == P["feature_set"] and period == P["period"],
                                 **{k: v for k, v in res.items() if k not in ("oof", "aucs")}})
        log(f"  band {band} done ({len(rows)} configurations so far)")
    rq3 = pd.DataFrame(rows).sort_values("auroc_mean").reset_index(drop=True)
    rq3.to_csv(OUT / "rq3_sensitivity.csv", index=False)
    fig_rq3(rq3)
    return rq3


def fig_rq3(rq3: pd.DataFrame) -> None:
    """Configurations sorted by AUROC on top, the choices behind each one as a dot matrix below."""
    n = len(rq3)
    fig, (ax, axm) = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True,
                                  gridspec_kw={"height_ratios": [2.2, 1.6], "hspace": 0.08})
    x = np.arange(n)
    band_color = {b: PALETTE[i] for i, b in enumerate(BANDS)}
    for xi, r in rq3.iterrows():
        ax.plot([xi, xi], [r.ci_low, r.ci_high], color=band_color[r.band], lw=0.9, alpha=0.6)
        ax.plot(xi, r.auroc_mean, "o", ms=3.2, color=band_color[r.band])
        if r.primary:
            ax.plot(xi, r.auroc_mean, "o", ms=11, mfc="none", mec=INK, mew=1.4)
            ax.annotate("primary\nconfiguration", (xi, r.auroc_mean), xytext=(-55, 35), textcoords="offset points",
                        fontsize=8, color=INK, arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
    for b, c in band_color.items():
        ax.plot([], [], "o", color=c, label=f"{b} Hz")
    ax.axhline(0.5, color=INK2, lw=0.8, ls="--")
    ax.set_ylabel("AUROC (mean, 95 % CI)")
    ax.legend(title="filter band", loc="upper left", fontsize=8, title_fontsize=8)
    ax.grid(axis="x", visible=False)
    ax.set_title(f"RQ3: specification curve over {n} pre-processing configurations, cohort B, primary model",
                 loc="left")
    rows = [("channel", CHANNEL_OPTIONS), ("feature set", list(FEATURE_SETS)), ("period", PERIODS)]
    yt, ylab = [], []
    yy = 0
    for key, options in rows:
        col = {"channel": "channel", "feature set": "feature_set", "period": "period"}[key]
        for opt in options:
            on = rq3[col] == opt
            axm.scatter(x[on], np.full(on.sum(), yy), s=10, color=INK2, lw=0)
            yt.append(yy); ylab.append(f"{key}: {opt}")
            yy += 1
        yy += 0.6
    axm.set_yticks(yt, ylab, fontsize=7.5)
    axm.set_xlabel("configuration rank (sorted by AUROC)")
    axm.grid(axis="x", visible=False)
    axm.invert_yaxis()
    fig.savefig(FIG / "fig_rq3_specification_curve.png")
    plt.close(fig)


# -----------------------------------------------------------------------------
# 9. External validation on the Icelandic database (cohort C)
# -----------------------------------------------------------------------------
def run_external(meta: pd.DataFrame, feats_long: pd.DataFrame, n_repeats: int) -> dict:
    """Fit the primary model on all of cohort B, score Iceland unchanged. Thresholds
    come from cohort B's out-of-fold scores (80 % sensitivity, and p = 0.5), never
    from the Icelandic data. Main quantity: share of term Icelandic cases flagged."""
    P = PRIMARY
    XB, yB, gB, _ = design(cohort(meta, "B"), feats_long, P["band"], P["channel"], P["feature_set"])
    XC, yC, gC, mC = design(cohort(meta, "C"), feats_long, P["band"], P["channel"], P["feature_set"])
    cvB = grouped_cv(XB, yB, gB, P["model"], P["sampler"], n_repeats=n_repeats)
    fpr, tpr, thr = roc_curve(yB, cvB["oof"])
    thr80 = float(thr[np.argmax(tpr >= 0.80)])
    pipe = make_pipeline(P["model"], P["sampler"], SEED).fit(np.asarray(XB, float), yB)
    sC = scores_of(pipe, np.asarray(XC, float))
    out = {"n_recordings": int(len(yC)), "n_pregnancies": int(len(np.unique(gC))),
           "n_preterm_recordings": int(yC.sum()), "n_preterm_pregnancies": int(len(np.unique(gC[yC == 1]))),
           "threshold_sens80_from_cohortB": thr80, "cohortB_fpr_at_sens80": float(fpr[np.argmax(tpr >= 0.80)])}
    for name, t in [("sens80", thr80), ("p50", 0.5)]:
        flagged = sC >= t
        term = yC == 0
        preg_flag = pd.Series(flagged).groupby(gC).max()
        preg_term = pd.Series(term).groupby(gC).all()
        out[f"fpr_recordings_{name}"] = float(flagged[term].mean())
        out[f"fpr_pregnancies_{name}"] = float(preg_flag[preg_term].mean())
        if yC.sum() > 0:
            out[f"sensitivity_recordings_{name}"] = float(flagged[~term].mean())
    out["auroc_iceland"] = float(roc_auc_score(yC, sC)) if 0 < yC.sum() < len(yC) else np.nan
    out["auroc_iceland_note"] = ("computed on very few preterm cases; unreliable" if yC.sum() > 0 else "no preterm cases")
    out["mean_score_cohortB_term"] = float(cvB["oof"][yB == 0].mean())
    out["mean_score_cohortB_preterm"] = float(cvB["oof"][yB == 1].mean())
    out["mean_score_iceland"] = float(sC.mean())
    with open(OUT / "external_validation.json", "w") as f:
        json.dump(out, f, indent=2)
    pd.DataFrame({"record_id": mC.index, "pregnancy_id": gC, "preterm": yC, "score": sC,
                  "flag_sens80": sC >= thr80, "flag_p50": sC >= 0.5}).to_csv(OUT / "external_validation_scores.csv", index=False)
    for k, v in out.items():
        log(f"  {k}: {v}")
    fig_external(yB, cvB["oof"], yC, sC, thr80, XB, XC)
    return out


def fig_external(yB, sB, yC, sC, thr80, XB, XC) -> None:
    """Score distributions on the left, raw feature distributions on the right (domain shift check)."""
    fig, axes = plt.subplots(1, 5, figsize=(13, 2.9))
    ax = axes[0]
    bins = np.linspace(0, 1, 26)
    ax.hist(sB[yB == 0], bins, density=True, histtype="step", lw=1.8, color=PALETTE[0], label="B term (OOF)")
    ax.hist(sB[yB == 1], bins, density=True, histtype="step", lw=1.8, color=PALETTE[1], label="B preterm (OOF)")
    ax.hist(sC, bins, density=True, histtype="step", lw=1.8, color=PALETTE[2], label="Iceland (external)")
    ax.axvline(thr80, color=INK2, lw=0.9, ls="--")
    ax.text(thr80, ax.get_ylim()[1] * 0.55, " 80 % sens.\n threshold", fontsize=6.5, color=INK2, va="top")
    ax.set_xlabel("predicted preterm probability")
    ax.set_ylabel("density")
    ax.legend(fontsize=6.5, loc="upper left")
    ax.set_title("model scores")
    for ax, col in zip(axes[1:], CORE_FEATURES):
        a, b, c = XB[col][yB == 0], XB[col][yB == 1], XC[col]
        lo, hi = np.nanpercentile(np.concatenate([a, b, c]), [1, 99])
        bins = np.linspace(lo, hi, 26)
        ax.hist(a, bins, density=True, histtype="step", lw=1.8, color=PALETTE[0])
        ax.hist(b, bins, density=True, histtype="step", lw=1.8, color=PALETTE[1])
        ax.hist(c, bins, density=True, histtype="step", lw=1.8, color=PALETTE[2])
        ax.set_title(col)
        ax.set_yticks([])
        ax.grid(False)
    fig.suptitle("External validation: pooled Ljubljana (cohort B) vs. Icelandic recordings, 0.3-4 Hz, S3",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig_external_validation.png")
    plt.close(fig)


# -----------------------------------------------------------------------------
# 10. Plain-text results summary
# -----------------------------------------------------------------------------
def txt_table(df: pd.DataFrame, index: bool = False) -> str:
    return df.round(3).to_string(index=index)


def write_report(meta: pd.DataFrame, feats_long: pd.DataFrame, exclog: pd.DataFrame, matches: pd.DataFrame) -> None:
    """Collect whatever stages have run into outputs/results.txt."""
    L = ["EHG PRETERM-BIRTH ANALYSIS: RESULTS",
         "Generated by ehg_analysis.py. Source tables are the CSV/JSON files in outputs/.", ""]

    def section(title: str, body: str) -> None:
        L.extend(["=" * 78, title, "=" * 78, body, ""])

    section("Cohorts after exclusions", txt_table(cohort_summary(meta, feats_long)))
    excl_body = txt_table(exclog.groupby(["dataset", "reason"]).size().rename("n").reset_index()) if len(exclog) else "none"
    excl_body += f"\n\nTPEHGT records matched to a TPEHG DB pregnancy on clinical variables: {len(matches)}"
    if (OUT / "pregnancy_groups.csv").exists():
        groups = pd.read_csv(OUT / "pregnancy_groups.csv")
        excl_body += (f"\nPregnancies with more than one recording (Ljubljana): {len(groups)}, "
                      f"of which {(groups.datasets == 'TPEHG DB').sum()} lie entirely within TPEHG DB "
                      f"(revealed by ICEHG DS Pair_RecID)")
    section("Exclusions", excl_body)
    per_ds = meta[meta.record_id.isin(feats_long.record_id.unique())].groupby("dataset").agg(
        recordings=("record_id", "size"), pregnancies=("pregnancy_id", "nunique"), preterm_recordings=("preterm", "sum"))
    section("Records per dataset (retained)", txt_table(per_ds.reset_index()))

    def csv_section(title: str, name: str, drop: tuple = ()) -> None:
        path = OUT / name
        if path.exists():
            section(title, txt_table(pd.read_csv(path).drop(columns=list(drop), errors="ignore")))

    csv_section("Feature verification vs. published TPEHG DB values", "feature_verification.csv")
    csv_section("RQ1: model x imbalance strategy", "rq1_model_comparison.csv", drop=("adasyn_fallbacks",))
    csv_section("RQ1: leakage comparison", "rq1_leakage_comparison.csv")
    if (OUT / "rq1_permutation.json").exists():
        section("RQ1: permutation test (primary model, cohort B)", (OUT / "rq1_permutation.json").read_text())
    csv_section("RQ2: confounder analysis", "rq2_confounder_analysis.csv")
    csv_section("RQ3: sensitivity (sorted by AUROC)", "rq3_sensitivity.csv")
    if (OUT / "external_validation.json").exists():
        section("External validation (Iceland)", (OUT / "external_validation.json").read_text())
    section("Figures", "\n".join(f"figures/{p.name}" for p in sorted(FIG.glob("*.png"))))
    (OUT / "results.txt").write_text("\n".join(L))
    log(f"  wrote {OUT / 'results.txt'}")


# -----------------------------------------------------------------------------
# 11. Stages and CLI
# -----------------------------------------------------------------------------
def stage_load() -> tuple[pd.DataFrame, pd.DataFrame, ExclusionLog]:
    log("[load] parsing headers and applying record-level exclusions")
    exclog = ExclusionLog()
    meta, matches = build_metadata(exclog)
    meta.to_csv(OUT / "metadata_clean.csv", index=False)
    matches.to_csv(OUT / "pregnancy_cross_matches.csv", index=False)
    exclog.frame().to_csv(OUT / "exclusions.csv", index=False)
    groups = pregnancy_groups(meta)
    groups.to_csv(OUT / "pregnancy_groups.csv", index=False)
    log(f"  retained {len(meta)} recordings; excluded {len(exclog.rows)}; "
        f"pregnancies with >1 recording: {len(groups)} ({(groups.datasets == 'TPEHG DB').sum()} entirely within TPEHG DB)")
    return meta, matches, exclog


def stage_features(meta: pd.DataFrame, exclog: ExclusionLog) -> pd.DataFrame:
    log(f"[features] extracting {len(BANDS)} bands x {len(CHANNELS)} channels for {len(meta)} recordings (parallel)")
    feats = extract_all(meta, exclog)
    feats.to_csv(OUT / "features_long.csv", index=False)
    exclog.frame().to_csv(OUT / "exclusions.csv", index=False)   # now includes the too-short records
    log(f"  {feats.record_id.nunique()} recordings with features; {len(feats)} rows")
    return feats


def load_cached() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta = pd.read_csv(OUT / "metadata_clean.csv")
    feats = pd.read_csv(OUT / "features_long.csv")
    excl = pd.read_csv(OUT / "exclusions.csv")
    matches = pd.read_csv(OUT / "pregnancy_cross_matches.csv")
    return meta, feats, excl, matches


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["load", "features", "verify", "rq1", "rq2", "rq3", "external", "report"],
                    help="run one stage (later stages use cached outputs)")
    ap.add_argument("--all", action="store_true", help="run every stage in order")
    ap.add_argument("--quick", action="store_true", help="smoke test: 2 CV repeats, 50 permutations, 200 bootstraps")
    ap.add_argument("--n-repeats", type=int, default=None)
    ap.add_argument("--n-perm", type=int, default=None)
    args = ap.parse_args(argv)
    if not (args.all or args.stage):
        ap.error("give --all or --stage")

    global N_BOOT
    n_repeats = args.n_repeats or (2 if args.quick else N_REPEATS)
    n_perm = args.n_perm or (50 if args.quick else N_PERM)
    if args.quick:
        N_BOOT = 200
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    np.random.seed(SEED)

    stages = ["load", "features", "verify", "rq1", "rq2", "rq3", "external", "report"] if args.all else [args.stage]

    meta = feats = excl = matches = None
    for st in stages:
        if st == "load":
            meta, matches, exclog = stage_load()
        elif st == "features":
            if meta is None:  # standalone run: reuse the load stage's tables if they exist
                if (OUT / "metadata_clean.csv").exists():
                    meta = pd.read_csv(OUT / "metadata_clean.csv")
                    matches = pd.read_csv(OUT / "pregnancy_cross_matches.csv")
                    excl = pd.read_csv(OUT / "exclusions.csv")
                    exclog = ExclusionLog(excl[~excl.reason.str.startswith("shorter")].to_dict("records"))
                else:
                    meta, matches, exclog = stage_load()
            feats = stage_features(meta, exclog)
        else:
            if feats is None:
                meta, feats, excl, matches = load_cached()
            if st == "verify":
                log("[verify] comparing features with the TPEHG DB .fvl tables")
                v = verify_features(feats)
                v.to_csv(OUT / "feature_verification.csv", index=False)
                fig_verification(feats)
                log(v.groupby("feature")[["pearson_r", "median_rel_error", "frac_within_5pct"]].mean().round(4).to_string())
            elif st == "rq1":
                log(f"[rq1] models x samplers, {n_repeats} repeats of {N_SPLITS}-fold grouped CV")
                run_rq1(meta, feats, n_repeats, n_perm)
            elif st == "rq2":
                log("[rq2] confounder analysis")
                run_rq2(meta, feats, n_repeats)
            elif st == "rq3":
                log("[rq3] sensitivity to pre-processing")
                run_rq3(meta, feats, n_repeats)
            elif st == "external":
                log("[external] validation on the Icelandic database")
                run_external(meta, feats, n_repeats)
            elif st == "report":
                log("[report]")
                excl = pd.read_csv(OUT / "exclusions.csv")
                write_report(meta, feats, excl, matches)
        if st in ("load", "features"):
            excl = exclog.frame()
    log("done.")
    # Shut the worker pool down cleanly; avoids a noisy (harmless) traceback at exit on macOS.
    try:
        from joblib.externals.loky import get_reusable_executor
        get_reusable_executor().shutdown(wait=True)
    except Exception:
        pass


if __name__ == "__main__":
    main()
