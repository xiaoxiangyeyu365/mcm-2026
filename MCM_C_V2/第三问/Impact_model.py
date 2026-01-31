"""第三问：pro dancer 与选手特征影响分析（judges vs fans）。

题目：
- Use the data including your fan vote estimates to develop a model that analyzes the impact of
  various pro dancers as well as characteristics for the celebrities (age, industry, etc).
- How much do such things impact how well a celebrity will do in the competition?
- Do they impact judges scores and fan votes in the same way?

实现思路（强调可解释）：
- 周级（week-level）两套回归：
  1) y_judge = judge_percent
  2) y_fan   = fan_vote_normalized
  特征包含：age、industry(one-hot)、season_progress、season fixed effect、pro dancer fixed effect。
  使用 GroupKFold 按 (season, celebrity) 分组交叉验证，避免同一选手不同周泄漏。

- 季级（contestant-season）回归：
  y_success = -placement（值越大越“成功”）
  特征：age、industry、pro dancer、season fixed effect。

- 影响量化：
  - baseline CV R^2
  - drop-column（特征组消融）：移除 dancer/industry/age 后 R^2 下降量
  - 系数层面：输出 dancer/industry 的 Ridge 系数（控制其它变量后相对影响）

输出到 impact_model_outputs/：
- tables/*.csv
- figures/*.png

运行：
D:/python/python.exe codeMCM/2026/c/impact_model_outputs/Impact_model.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class ModelSpec:
    name: str
    y_col: str


def _find_file(candidates: List[Path]) -> Path:
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    raise FileNotFoundError("未找到数据文件。已尝试路径：\n" + "\n".join(str(p) for p in candidates))


def _ensure_numeric(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def prepare_week_level(cleaned_path: Path, fan_path: Path) -> pd.DataFrame:
    cleaned = pd.read_csv(cleaned_path)
    fan = pd.read_csv(fan_path)

    for frame in (cleaned, fan):
        if "celebrity_name" in frame.columns:
            frame["celebrity_name"] = frame["celebrity_name"].astype(str).str.strip()
        if "ballroom_partner" in frame.columns:
            frame["ballroom_partner"] = frame["ballroom_partner"].astype(str).str.strip()
        for col in ("season", "week"):
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")

    # fan_vote_normalized
    if "fan_vote_normalized" not in fan.columns:
        raise ValueError("fan_vote_final_fixed.csv 缺少 fan_vote_normalized")

    keys = ["season", "week", "celebrity_name"]
    fan_small = fan[keys + ["fan_vote_normalized"]].drop_duplicates(keys, keep="last")

    df = cleaned.merge(fan_small, on=keys, how="left", validate="1:1")

    # judge_percent
    if "total_judge_score" not in df.columns:
        raise ValueError("cleaned 数据缺少 total_judge_score")

    df = df.copy()
    df["week_total_score"] = df.groupby(["season", "week"])["total_judge_score"].transform("sum")
    df["judge_percent"] = df["total_judge_score"] / df["week_total_score"].replace(0, np.nan)

    _ensure_numeric(df, ["judge_percent", "fan_vote_normalized", "celebrity_age_during_season", "season_progress", "placement"])

    # engineered celebrity characteristics
    df["age_sq"] = df["celebrity_age_during_season"].astype(float) ** 2

    # keep usable rows
    df = df.dropna(subset=["judge_percent", "fan_vote_normalized", "ballroom_partner", "celebrity_age_during_season", "season_progress", "season"])

    # group id for CV
    df["group_id"] = df["season"].astype(int).astype(str) + "|" + df["celebrity_name"].astype(str)

    # industry columns
    ind_cols = [c for c in df.columns if c.startswith("ind_")]
    if len(ind_cols) == 0:
        raise ValueError("未检测到 ind_* 行业列")

    # make sure industry are 0/1 numeric
    for c in ind_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # industry groupings (broader celebrity characteristic proxies)
    # Note: these are derived from one-hot industries to create denser, more interpretable groups.
    def _has(col: str) -> bool:
        return col in df.columns

    entertainment_cols = [
        c
        for c in [
            "ind_Actor/Actress",
            "ind_Musician",
            "ind_Singer/Rapper",
            "ind_TV Personality",
            "ind_Social Media Personality",
            "ind_Social media personality",
            "ind_Comedian",
            "ind_Model",
            "ind_Fashion Designer",
            "ind_Radio Personality",
            "ind_Magician",
            "ind_Producer",
        ]
        if _has(c)
    ]

    sports_cols = [
        c
        for c in [
            "ind_Athlete",
            "ind_Sports Broadcaster",
            "ind_Racing Driver",
        ]
        if _has(c)
    ]

    public_service_cols = [
        c
        for c in [
            "ind_Politician",
            "ind_Military",
            "ind_Journalist",
            "ind_News Anchor",
            "ind_Conservationist",
            "ind_Astronaut",
            "ind_Motivational Speaker",
        ]
        if _has(c)
    ]

    df["grp_entertainment"] = df[entertainment_cols].sum(axis=1) if entertainment_cols else 0.0
    df["grp_sports"] = df[sports_cols].sum(axis=1) if sports_cols else 0.0
    df["grp_public_service"] = df[public_service_cols].sum(axis=1) if public_service_cols else 0.0

    _ensure_numeric(df, ["age_sq", "grp_entertainment", "grp_sports", "grp_public_service"])

    return df


def prepare_season_level(week_df: pd.DataFrame) -> pd.DataFrame:
    # one row per (season, celebrity)
    ind_cols = [c for c in week_df.columns if c.startswith("ind_")]

    # dancer per season is usually constant; use mode
    def _mode(x: pd.Series) -> str:
        x = x.dropna().astype(str)
        if x.empty:
            return ""
        return x.value_counts().index[0]

    gcols = ["season", "celebrity_name"]
    agg: Dict[str, Tuple[str, str]] = {
        "ballroom_partner": (_mode, "ballroom_partner"),
        "celebrity_age_during_season": ("first", "celebrity_age_during_season"),
        "placement": ("first", "placement"),
    }

    # industries: first row (they are static)
    for c in ind_cols:
        agg[c] = ("first", c)

    df = (
        week_df.groupby(gcols, as_index=False)
        .agg({k: v[0] for k, v in agg.items()})
        .rename(columns={"ballroom_partner": "ballroom_partner"})
    )

    # success score: higher better
    df["success_score"] = -pd.to_numeric(df["placement"], errors="coerce")

    df = df.dropna(subset=["success_score", "ballroom_partner", "celebrity_age_during_season", "season"])
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    return df


def build_pipeline(categorical: List[str], numeric: List[str], passthrough: List[str]):
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numeric),
            ("pt", "passthrough", passthrough),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    model = Ridge(alpha=1.0, random_state=0)

    return Pipeline([("pre", pre), ("model", model)])


def cv_score_r2(df: pd.DataFrame, y_col: str, features: List[str], group_col: str) -> float:
    from sklearn.metrics import r2_score
    from sklearn.model_selection import GroupKFold

    X = df[features]
    y = df[y_col].values
    groups = df[group_col].values

    # choose splits based on number of groups
    n_groups = len(pd.unique(groups))
    n_splits = 5 if n_groups >= 5 else max(2, n_groups)

    gkf = GroupKFold(n_splits=n_splits)

    preds = np.full_like(y, fill_value=np.nan, dtype=float)

    for train_idx, test_idx in gkf.split(X, y, groups):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]

        pipe = make_model(train, y_col, features)
        pipe.fit(train[features], train[y_col].values)
        preds[test_idx] = pipe.predict(test[features])

    mask = ~np.isnan(preds)
    return float(r2_score(y[mask], preds[mask]))


def make_model(train_df: pd.DataFrame, y_col: str, features: List[str]):
    categorical = [c for c in ["season", "ballroom_partner"] if c in features]
    numeric = [c for c in ["celebrity_age_during_season", "age_sq", "season_progress"] if c in features]
    passthrough = [c for c in features if c.startswith("ind_") or c.startswith("grp_")]

    return build_pipeline(categorical=categorical, numeric=numeric, passthrough=passthrough)


def drop_column_importance_week(df: pd.DataFrame, y_col: str) -> pd.DataFrame:
    ind_cols = [c for c in df.columns if c.startswith("ind_")]
    base_features = ["season", "ballroom_partner", "celebrity_age_during_season", "season_progress"] + ind_cols

    base_r2 = cv_score_r2(df, y_col=y_col, features=base_features, group_col="group_id")

    def score_without(drop: List[str]) -> float:
        feats = [c for c in base_features if c not in drop]
        return cv_score_r2(df, y_col=y_col, features=feats, group_col="group_id")

    r2_no_dancer = score_without(["ballroom_partner"])
    r2_no_age = score_without(["celebrity_age_during_season"])
    r2_no_progress = score_without(["season_progress"])

    # industry group: remove all ind_*
    feats_no_ind = [c for c in base_features if not c.startswith("ind_")]
    r2_no_ind = cv_score_r2(df, y_col=y_col, features=feats_no_ind, group_col="group_id")

    out = pd.DataFrame(
        [
            {"target": y_col, "group": "baseline", "cv_r2": base_r2, "delta_r2_vs_base": 0.0},
            {"target": y_col, "group": "no_dancer", "cv_r2": r2_no_dancer, "delta_r2_vs_base": r2_no_dancer - base_r2},
            {"target": y_col, "group": "no_industry", "cv_r2": r2_no_ind, "delta_r2_vs_base": r2_no_ind - base_r2},
            {"target": y_col, "group": "no_age", "cv_r2": r2_no_age, "delta_r2_vs_base": r2_no_age - base_r2},
            {"target": y_col, "group": "no_season_progress", "cv_r2": r2_no_progress, "delta_r2_vs_base": r2_no_progress - base_r2},
        ]
    )
    return out


def extract_onehot_coefs(pipeline, prefix: str) -> pd.DataFrame:
    # assumes pipeline has ColumnTransformer named pre and Ridge named model
    pre = pipeline.named_steps["pre"]
    model = pipeline.named_steps["model"]

    feature_names: List[str] = []

    for name, transformer, cols in pre.transformers_:
        if name == "cat":
            enc = transformer
            enc_names = enc.get_feature_names_out(cols)
            feature_names.extend(enc_names.tolist())
        elif name == "num":
            # scaler keeps names
            feature_names.extend(list(cols))
        elif name == "pt":
            feature_names.extend(list(cols))

    coefs = pd.Series(model.coef_, index=feature_names, dtype=float)
    keep = coefs[coefs.index.str.startswith(prefix)].sort_values(ascending=False)

    out = keep.reset_index()
    out.columns = ["feature", "coef"]
    # OneHotEncoder 特征名形如 "ballroom_partner_<value>"，这里严格去掉前缀
    out["name"] = out["feature"].str.replace("^" + prefix, "", regex=True)
    return out[["name", "coef"]]


def plot_barh(df: pd.DataFrame, title: str, out_path: Path, top_n: int = 15) -> None:
    import matplotlib.pyplot as plt

    d = df.copy().head(top_n)
    d = d.iloc[::-1]

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    ax.barh(d["name"], d["coef"], color="#4c78a8")
    ax.axvline(0.0, color="#666666", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Ridge coefficient (conditional effect)")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_scatter(x: pd.DataFrame, y: pd.DataFrame, title: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    m = x.merge(y, on="name", how="inner", suffixes=("_judge", "_fan"))

    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    ax.scatter(m["coef_judge"], m["coef_fan"], s=22, alpha=0.65, color="#333333")
    ax.axhline(0.0, color="#888888", linewidth=1)
    ax.axvline(0.0, color="#888888", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Effect on judge_percent (coef)")
    ax.set_ylabel("Effect on fan_vote_normalized (coef)")
    ax.grid(True, alpha=0.25)

    # annotate extremes
    m2 = m.copy()
    m2["score"] = (m2["coef_judge"].abs() + m2["coef_fan"].abs())
    for _, r in m2.sort_values("score", ascending=False).head(10).iterrows():
        ax.annotate(str(r["name"]), (r["coef_judge"], r["coef_fan"]), fontsize=8, alpha=0.8)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def get_all_feature_coefs(pipeline) -> pd.DataFrame:
    """Return a DataFrame of (feature, coef) after preprocessing.

    Feature names follow:
    - OneHotEncoder: <col>_<category>
    - Numeric: original column names
    - Passthrough: original column names
    """

    pre = pipeline.named_steps["pre"]
    model = pipeline.named_steps["model"]

    feature_names: List[str] = []
    for name, transformer, cols in pre.transformers_:
        if name == "cat":
            enc = transformer
            feature_names.extend(enc.get_feature_names_out(cols).tolist())
        elif name == "num":
            feature_names.extend(list(cols))
        elif name == "pt":
            feature_names.extend(list(cols))

    coefs = pd.DataFrame({"feature": feature_names, "coef": model.coef_.astype(float)})
    return coefs


def _top_abs(df: pd.DataFrame, n: int) -> pd.DataFrame:
    d = df.copy()
    d["abs_coef"] = d["coef"].abs()
    return d.sort_values("abs_coef", ascending=False).head(n).drop(columns=["abs_coef"]) 


def plot_celebrity_characteristics_barh(
    judge_df: pd.DataFrame,
    fan_df: pd.DataFrame,
    title: str,
    out_path: Path,
    top_n: int = 12,
) -> None:
    """Two-panel barh: judge vs fan celebrity characteristic effects (age + industries)."""

    import matplotlib.pyplot as plt

    j = _top_abs(judge_df, top_n).copy()
    f = _top_abs(fan_df, top_n).copy()

    # reverse for barh
    j = j.iloc[::-1]
    f = f.iloc[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), sharex=False)

    axes[0].barh(j["name"], j["coef"], color="#4c78a8")
    axes[0].axvline(0.0, color="#666666", linewidth=1)
    axes[0].set_title("Judges model")
    axes[0].set_xlabel("Ridge coefficient (conditional effect)")
    axes[0].grid(True, axis="x", alpha=0.25)

    axes[1].barh(f["name"], f["coef"], color="#f58518")
    axes[1].axvline(0.0, color="#666666", linewidth=1)
    axes[1].set_title("Fans model")
    axes[1].set_xlabel("Ridge coefficient (conditional effect)")
    axes[1].grid(True, axis="x", alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path)
    plt.close(fig)


def plot_characteristics_scatter(
    judge_df: pd.DataFrame,
    fan_df: pd.DataFrame,
    title: str,
    out_path: Path,
    label_top_n: int = 12,
) -> None:
    """Scatter of celebrity characteristic coefs: judge vs fan."""

    import matplotlib.pyplot as plt

    m = judge_df.merge(fan_df, on="name", how="inner", suffixes=("_judge", "_fan"))

    fig, ax = plt.subplots(figsize=(7.0, 6.4))
    ax.scatter(m["coef_judge"], m["coef_fan"], s=28, alpha=0.65, color="#333333")
    ax.axhline(0.0, color="#888888", linewidth=1)
    ax.axvline(0.0, color="#888888", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Effect on judge_percent (coef)")
    ax.set_ylabel("Effect on fan_vote_normalized (coef)")
    ax.grid(True, alpha=0.25)

    m2 = m.copy()
    m2["score"] = m2["coef_judge"].abs() + m2["coef_fan"].abs()
    for _, r in m2.sort_values("score", ascending=False).head(label_top_n).iterrows():
        ax.annotate(str(r["name"]), (r["coef_judge"], r["coef_fan"]), fontsize=8, alpha=0.85)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    here = Path(__file__).resolve().parent
    project_root = here.parents[1]  # .../2026/c

    cleaned_path = _find_file(
        [
            project_root / "cleaned_dwts_data_V2.csv",
            project_root.parents[2] / "cleaned_dwts_data_V2.csv",
            Path("D:/codePYTHON/cleaned_dwts_data_V2.csv"),
            Path("d:/codePYTHON/cleaned_dwts_data_V2.csv"),
        ]
    )
    fan_path = _find_file(
        [
            project_root / "fan_vote_final_fixed.csv",
            project_root.parents[2] / "fan_vote_final_fixed.csv",
            Path("D:/codePYTHON/fan_vote_final_fixed.csv"),
            Path("d:/codePYTHON/fan_vote_final_fixed.csv"),
        ]
    )

    out_dir = here
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] cleaned: {cleaned_path}")
    print(f"[INFO] fan:     {fan_path}")

    week = prepare_week_level(cleaned_path, fan_path)

    # season-level
    season_df = prepare_season_level(week)

    # ---------------- Week-level models ----------------
    specs = [
        ModelSpec(name="judge", y_col="judge_percent"),
        ModelSpec(name="fan", y_col="fan_vote_normalized"),
    ]

    ind_cols = [c for c in week.columns if c.startswith("ind_")]
    base_features = ["season", "ballroom_partner", "celebrity_age_during_season", "season_progress"] + ind_cols

    group_importance_rows: List[pd.DataFrame] = []

    dancer_effects: Dict[str, pd.DataFrame] = {}
    industry_effects: Dict[str, pd.DataFrame] = {}
    # celebrity-characteristics-only model (more interpretable than dozens of sparse ind_* columns)
    char_effects: Dict[str, pd.DataFrame] = {}

    for spec in specs:
        imp = drop_column_importance_week(week, y_col=spec.y_col)
        group_importance_rows.append(imp)

        pipe = make_model(week, y_col=spec.y_col, features=base_features)
        pipe.fit(week[base_features], week[spec.y_col].values)

        coef_all = get_all_feature_coefs(pipe)

        # dancer effects
        de = extract_onehot_coefs(pipe, prefix="ballroom_partner_")
        # support (how many rows per dancer)
        support = week["ballroom_partner"].value_counts().rename("n_weeks").reset_index()
        support.columns = ["name", "n_weeks"]
        de = de.merge(support, on="name", how="left")
        de = de.sort_values("coef", ascending=False)

        # filter to reasonably frequent dancers
        de_f = de[de["n_weeks"] >= 30].copy().reset_index(drop=True)
        dancer_effects[spec.name] = de_f

        # industry effects: they are passthrough ind_*, coefficients on those columns
        ie = coef_all[coef_all["feature"].str.startswith("ind_")].copy()
        ie["name"] = ie["feature"].str.replace("^ind_", "", regex=True)
        ie = ie[["name", "coef"]]
        industry_effects[spec.name] = ie.sort_values("coef", ascending=False).reset_index(drop=True)

        # celebrity characteristics (age + industries)
        # Fit a dedicated model with engineered characteristics:
        # age, age_sq (non-linear age), and 3 coarse industry groups.
        char_features = [
            "season",
            "ballroom_partner",
            "celebrity_age_during_season",
            "age_sq",
            "season_progress",
            "grp_entertainment",
            "grp_sports",
            "grp_public_service",
        ]
        char_features = [c for c in char_features if c in week.columns]
        pipe_char = make_model(week, y_col=spec.y_col, features=char_features)
        pipe_char.fit(week[char_features], week[spec.y_col].values)
        coef_char = get_all_feature_coefs(pipe_char)

        keep_feats = [
            "celebrity_age_during_season",
            "age_sq",
            "grp_entertainment",
            "grp_sports",
            "grp_public_service",
        ]
        ce = coef_char[coef_char["feature"].isin([f for f in keep_feats if f in coef_char["feature"].values])].copy()
        rename_map = {
            "celebrity_age_during_season": "age",
            "age_sq": "age_sq",
            "grp_entertainment": "industry_group_entertainment",
            "grp_sports": "industry_group_sports",
            "grp_public_service": "industry_group_public_service",
        }
        ce["name"] = ce["feature"].map(rename_map).fillna(ce["feature"])
        ce = ce[["name", "coef"]].reset_index(drop=True)
        char_effects[spec.name] = ce

        # plots
        plot_barh(
            de_f,
            title=f"Top pro dancer effects on {spec.y_col} (Ridge, conditional)",
            out_path=fig_dir / f"fig_top_dancers_{spec.name}.png",
            top_n=15,
        )

    group_importance = pd.concat(group_importance_rows, ignore_index=True)
    group_importance.to_csv(tab_dir / "feature_group_drop_r2.csv", index=False, encoding="utf-8-sig")

    dancer_effects["judge"].to_csv(tab_dir / "dancer_effects_judge.csv", index=False, encoding="utf-8-sig")
    dancer_effects["fan"].to_csv(tab_dir / "dancer_effects_fan.csv", index=False, encoding="utf-8-sig")

    industry_effects["judge"].to_csv(tab_dir / "industry_effects_judge.csv", index=False, encoding="utf-8-sig")
    industry_effects["fan"].to_csv(tab_dir / "industry_effects_fan.csv", index=False, encoding="utf-8-sig")

    # celebrity characteristics outputs (engineered, denser groups)
    char_effects["judge"].to_csv(tab_dir / "celebrity_characteristics_effects_judge.csv", index=False, encoding="utf-8-sig")
    char_effects["fan"].to_csv(tab_dir / "celebrity_characteristics_effects_fan.csv", index=False, encoding="utf-8-sig")

    plot_celebrity_characteristics_barh(
        judge_df=char_effects["judge"],
        fan_df=char_effects["fan"],
        title="Celebrity characteristics effects (age + industries): judges vs fans",
        out_path=fig_dir / "fig_celebrity_characteristics_effects_judge_vs_fan.png",
        top_n=12,
    )

    plot_characteristics_scatter(
        judge_df=char_effects["judge"],
        fan_df=char_effects["fan"],
        title="Do celebrity characteristics impact judges and fans the same way?",
        out_path=fig_dir / "fig_celebrity_characteristics_scatter_judge_vs_fan.png",
        label_top_n=12,
    )

    # compare dancer effects across judge vs fan
    plot_scatter(
        dancer_effects["judge"][["name", "coef"]],
        dancer_effects["fan"][["name", "coef"]],
        title="Pro dancer conditional effects: judge vs fan",
        out_path=fig_dir / "fig_dancer_effect_scatter_judge_vs_fan.png",
    )

    # ---------------- Season-level success model ----------------
    # Predict success_score using only static features (no weekly performance)
    ind_cols_s = [c for c in season_df.columns if c.startswith("ind_")]
    season_features = ["season", "ballroom_partner", "celebrity_age_during_season"] + ind_cols_s

    # Use simple CV (no grouping necessary beyond i.i.d contestant-seasons)
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    Xs = season_df[season_features]
    ys = season_df["success_score"].values

    preds = np.full_like(ys, np.nan, dtype=float)
    for tr, te in kf.split(Xs):
        tr_df = season_df.iloc[tr]
        te_df = season_df.iloc[te]
        pipe = make_model(tr_df, y_col="success_score", features=season_features)
        pipe.fit(tr_df[season_features], tr_df["success_score"].values)
        preds[te] = pipe.predict(te_df[season_features])

    success_r2 = float(r2_score(ys, preds))

    # drop-column style for season model
    def season_r2_without(drop_cols: List[str]) -> float:
        feats = [c for c in season_features if c not in drop_cols]
        X = season_df[feats]
        y = season_df["success_score"].values
        preds2 = np.full_like(y, np.nan, dtype=float)
        for tr, te in kf.split(X):
            tr_df = season_df.iloc[tr]
            te_df = season_df.iloc[te]
            pipe = make_model(tr_df, y_col="success_score", features=feats)
            pipe.fit(tr_df[feats], tr_df["success_score"].values)
            preds2[te] = pipe.predict(te_df[feats])
        return float(r2_score(y, preds2))

    r2_no_dancer = season_r2_without(["ballroom_partner"])
    r2_no_age = season_r2_without(["celebrity_age_during_season"])

    feats_no_ind = [c for c in season_features if not c.startswith("ind_")]
    X = season_df[feats_no_ind]
    y = season_df["success_score"].values
    preds3 = np.full_like(y, np.nan, dtype=float)
    for tr, te in kf.split(X):
        tr_df = season_df.iloc[tr]
        te_df = season_df.iloc[te]
        pipe = make_model(tr_df, y_col="success_score", features=feats_no_ind)
        pipe.fit(tr_df[feats_no_ind], tr_df["success_score"].values)
        preds3[te] = pipe.predict(te_df[feats_no_ind])
    r2_no_ind = float(r2_score(y, preds3))

    success_imp = pd.DataFrame(
        [
            {"target": "success_score(-placement)", "group": "baseline", "cv_r2": success_r2, "delta_r2_vs_base": 0.0},
            {"target": "success_score(-placement)", "group": "no_dancer", "cv_r2": r2_no_dancer, "delta_r2_vs_base": r2_no_dancer - success_r2},
            {"target": "success_score(-placement)", "group": "no_industry", "cv_r2": r2_no_ind, "delta_r2_vs_base": r2_no_ind - success_r2},
            {"target": "success_score(-placement)", "group": "no_age", "cv_r2": r2_no_age, "delta_r2_vs_base": r2_no_age - success_r2},
        ]
    )
    success_imp.to_csv(tab_dir / "success_feature_group_drop_r2.csv", index=False, encoding="utf-8-sig")

    # Fit full season model and export dancer coefficients (frequent only)
    pipe = make_model(season_df, y_col="success_score", features=season_features)
    pipe.fit(season_df[season_features], season_df["success_score"].values)

    de = extract_onehot_coefs(pipe, prefix="ballroom_partner_")
    support = season_df["ballroom_partner"].value_counts().rename("n_seasons").reset_index()
    support.columns = ["name", "n_seasons"]
    de = de.merge(support, on="name", how="left")
    de = de.sort_values("coef", ascending=False)
    de_f = de[de["n_seasons"] >= 8].copy().reset_index(drop=True)
    de_f.to_csv(tab_dir / "dancer_effects_success.csv", index=False, encoding="utf-8-sig")

    plot_barh(
        de_f,
        title="Top pro dancer effects on success_score (-placement)",
        out_path=fig_dir / "fig_top_dancers_success.png",
        top_n=15,
    )

    # quick summary txt for writeup
    summary = {
        "n_week_rows": int(len(week)),
        "n_contestant_seasons": int(len(season_df)),
        "judge_cv_r2": float(group_importance[group_importance["target"] == "judge_percent"].query("group=='baseline'")["cv_r2"].iloc[0]),
        "fan_cv_r2": float(group_importance[group_importance["target"] == "fan_vote_normalized"].query("group=='baseline'")["cv_r2"].iloc[0]),
        "success_cv_r2": float(success_r2),
    }

    (tab_dir / "summary.json").write_text(pd.Series(summary).to_json(indent=2), encoding="utf-8")

    print("[OK] wrote outputs to:", out_dir)


if __name__ == "__main__":
    main()
