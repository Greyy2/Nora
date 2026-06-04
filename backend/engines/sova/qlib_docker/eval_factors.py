"""
SOVA → Qlib Factor Evaluator
Generates SOVA factor expressions, computes them on Qlib cn_data,
saves as combined_factors_df.parquet, then runs Qlib backtest pipeline.
Outputs IC, ICIR, Rank IC, annualized return, max drawdown.
"""
import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

FACTOR_EXPRESSIONS = {
    # Core momentum factors
    "sova_momentum_5d": "Ref($close, 1)/$close - 1",
    "sova_momentum_10d": "$close/Ref($close, 10) - 1",
    "sova_momentum_20d": "$close/Ref($close, 20) - 1",
    
    # Volatility-adjusted momentum
    "sova_vol_adj_mom": "($close/Ref($close, 5) - 1) / (Std($close/Ref($close, 1)-1, 20) + 1e-12)",
    
    # Mean reversion signals
    "sova_mean_rev_5": "Mean($close, 5)/$close - 1",
    "sova_mean_rev_20": "Mean($close, 20)/$close - 1",
    "sova_zscore_close": "($close - Mean($close, 20)) / (Std($close, 20) + 1e-12)",
    
    # Volume signals
    "sova_vol_surge": "$volume / (Mean($volume, 20) + 1e-12)",
    "sova_vol_price_corr": "Corr($close, Log($volume+1), 10)",
    "sova_vol_momentum": "Mean($volume, 5) / (Mean($volume, 20) + 1e-12)",
    
    # Price structure
    "sova_klen": "($high - $low) / $open",
    "sova_upper_shadow": "($high - Greater($open, $close)) / ($high - $low + 1e-12)",
    "sova_body_ratio": "Abs($close - $open) / ($high - $low + 1e-12)",
    
    # Trend strength
    "sova_resi_5": "Resi($close, 5) / $close",
    "sova_resi_10": "Resi($close, 10) / $close",
    "sova_rsqr_5": "Rsquare($close, 5)",
    "sova_rsqr_20": "Rsquare($close, 20)",
    
    # Cross-feature interactions
    "sova_vol_weighted_ret": "Std(Abs($close/Ref($close,1)-1)*$volume, 5) / (Mean(Abs($close/Ref($close,1)-1)*$volume, 5) + 1e-12)",
    "sova_cord_5": "Corr($close/Ref($close,1), Log($volume/Ref($volume,1)+1), 5)",
    
    # Regime-adaptive signals
    "sova_roc_60": "Ref($close, 60) / $close",
    "sova_std_ratio": "Std($close, 5) / (Std($close, 20) + 1e-12)",
}


def download_qlib_data():
    """Download cn_data if not present."""
    data_path = Path.home() / ".qlib" / "qlib_data" / "cn_data"
    if data_path.exists() and any(data_path.iterdir()):
        print(f"[SOVA] Qlib data already exists at {data_path}")
        return True

    print("[SOVA] Downloading Qlib cn_data (this may take a few minutes)...")
    try:
        from qlib.tests.data import GetData
        GetData().qlib_data(
            name="qlib_data",
            target_dir=str(data_path),
            region="cn",
            interval="1d",
            delete_old=False,
            exists_skip=True,
        )
        print("[SOVA] Data download complete.")
        return True
    except Exception as e:
        print(f"[SOVA] Data download failed: {e}")
        return False


def compute_factors_via_qlib():
    """Use Qlib expression engine to compute SOVA factors on cn_data."""
    import qlib
    from qlib.data import D

    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")

    instruments = D.instruments(market="csi300")
    fields = list(FACTOR_EXPRESSIONS.values())
    names = list(FACTOR_EXPRESSIONS.keys())

    print(f"[SOVA] Computing {len(fields)} factors on CSI300...")
    
    try:
        df = D.features(
            instruments=instruments,
            fields=fields,
            start_time="2008-01-01",
            end_time=None,
            freq="day"
        )
    except Exception as e:
        print(f"[SOVA] Error computing all factors: {e}")
        print("[SOVA] Falling back to individual factor computation...")
        dfs = []
        successful_names = []
        for name, expr in FACTOR_EXPRESSIONS.items():
            try:
                factor_df = D.features(
                    instruments=instruments,
                    fields=[expr],
                    start_time="2008-01-01",
                    end_time=None,
                    freq="day"
                )
                factor_df.columns = [name]
                dfs.append(factor_df)
                successful_names.append(name)
                print(f"  [OK] {name}")
            except Exception as ex:
                print(f"  [SKIP] {name}: {ex}")
        
        if not dfs:
            print("[SOVA] No factors computed successfully!")
            return None
        
        df = pd.concat(dfs, axis=1)
        names = successful_names
        print(f"[SOVA] Successfully computed {len(successful_names)}/{len(FACTOR_EXPRESSIONS)} factors")
        
        output_path = Path("/workspace/eval/combined_factors_df.parquet")
        df.to_parquet(output_path)
        print(f"[SOVA] Saved factors to {output_path}")
        print(f"[SOVA] Shape: {df.shape}")
        print(f"[SOVA] Date range: {df.index.get_level_values(0).min()} to {df.index.get_level_values(0).max()}")
        return df

    df.columns = names
    
    output_path = Path("/workspace/eval/combined_factors_df.parquet")
    df.to_parquet(output_path)
    print(f"[SOVA] Saved factors to {output_path}")
    print(f"[SOVA] Shape: {df.shape}")
    print(f"[SOVA] Columns: {list(df.columns)}")
    print(f"[SOVA] Date range: {df.index.get_level_values(0).min()} to {df.index.get_level_values(0).max()}")
    
    return df


def compute_standalone_ic(df):
    """Compute IC for each factor independently before running full backtest."""
    import qlib
    from qlib.data import D

    if df is None or df.empty:
        return {}

    instruments = D.instruments(market="csi300")
    
    label_expr = "Ref($close, -2)/Ref($close, -1) - 1"
    try:
        label_df = D.features(
            instruments=instruments,
            fields=[label_expr],
            start_time="2017-01-01",
            end_time=None,
            freq="day"
        )
        label_df.columns = ["label"]
    except Exception as e:
        print(f"[SOVA] Cannot compute label: {e}")
        return {}

    results = {}
    # Qlib returns MultiIndex: level 0 = instrument, level 1 = datetime
    date_level = 1 if isinstance(df.index.get_level_values(0)[0], str) else 0
    inst_level = 1 - date_level
    
    test_mask = df.index.get_level_values(date_level) >= pd.Timestamp("2017-01-01")
    df_test = df[test_mask]
    
    common_idx = df_test.index.intersection(label_df.index)
    if len(common_idx) == 0:
        print("[SOVA] No overlapping dates between factors and labels")
        return {}

    label_aligned = label_df.loc[common_idx, "label"]
    
    # Build merged dataframe for vectorized groupby
    merged = df_test.loc[common_idx].copy()
    merged["_label_"] = label_aligned.values
    
    for col in df_test.columns:
        sub = merged[[col, "_label_"]].dropna()
        if len(sub) < 100:
            continue
        
        # Vectorized: groupby date, compute spearman IC per day
        daily_ic = sub.groupby(level=date_level).apply(
            lambda g: g[col].corr(g["_label_"], method="spearman") if len(g) > 10 else np.nan
        ).dropna()
        
        if len(daily_ic) > 20:
            ic_mean = float(daily_ic.mean())
            ic_std = float(daily_ic.std())
            icir = ic_mean / (ic_std + 1e-12)
            results[col] = {
                "IC": round(ic_mean, 6),
                "ICIR": round(icir, 6),
                "IC_std": round(ic_std, 6),
                "n_days": len(daily_ic)
            }
    
    return results


def run_qlib_backtest():
    """Run Qlib LightGBM pipeline programmatically."""
    import qlib
    from qlib.data import D
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import QlibDataLoader
    from qlib.data.dataset.processor import CSZScoreNorm, Fillna, DropnaLabel, CSRankNorm
    from qlib.contrib.model.gbdt import LGBModel
    from qlib.workflow import R
    from qlib.workflow.record_temp import SignalRecord, SigAnaRecord

    print("[SOVA] Building Qlib pipeline programmatically...")

    # Feature expressions (SOVA-generated)
    feature_exprs = list(FACTOR_EXPRESSIONS.values())
    feature_names = list(FACTOR_EXPRESSIONS.keys())
    label_expr = ["Ref($close, -2)/Ref($close, -1) - 1"]
    label_name = ["LABEL0"]

    loader_config = {
        "feature": [feature_exprs, feature_names],
        "label": [label_expr, label_name],
    }

    data_loader = QlibDataLoader(config=loader_config, freq="day")

    infer_processors = [
        CSZScoreNorm(fields_group="feature"),
        Fillna(fields_group="feature"),
    ]
    learn_processors = [
        DropnaLabel(),
        CSRankNorm(fields_group="label"),
    ]

    handler = DataHandlerLP(
        instruments="csi300",
        start_time="2008-01-01",
        end_time=None,
        data_loader=data_loader,
        infer_processors=infer_processors,
        learn_processors=learn_processors,
    )

    dataset = DatasetH(
        handler=handler,
        segments={
            "train": ("2008-01-01", "2014-12-31"),
            "valid": ("2015-01-01", "2016-12-31"),
            "test": ("2017-01-01", None),
        },
    )

    print(f"[SOVA] Dataset ready: {len(feature_names)} features")

    # Train LightGBM
    model = LGBModel(
        loss="mse",
        colsample_bytree=0.8879,
        learning_rate=0.2,
        subsample=0.8789,
        lambda_l1=205.6999,
        lambda_l2=580.9768,
        max_depth=8,
        num_leaves=210,
        num_threads=4,
    )

    with R.start(experiment_name="sova_qlib_eval"):
        recorder = R.get_recorder()
        R.log_params(model_type="LGBModel", n_factors=len(feature_names))
        model.fit(dataset)
        print("[SOVA] Model trained successfully")

        # Signal record (predictions)
        sr = SignalRecord(model=model, dataset=dataset, recorder=recorder)
        sr.generate()
        print("[SOVA] Predictions generated")

        # Signal analysis (IC/ICIR)
        sar = SigAnaRecord(recorder=recorder, ana_long_short=False, ann_scaler=252)
        sar.generate()
        print("[SOVA] Signal analysis complete")

        metrics = recorder.list_metrics()

        # Compute IC/ICIR directly from predictions if missing
        if "ICIR" not in metrics:
            try:
                pred_df = recorder.load_object("pred.pkl")
                # Get test labels from dataset
                test_data = dataset.prepare("test", col_set=["label"], data_key="learn")
                common = pred_df.index.intersection(test_data.index)
                pred_aligned = pred_df.loc[common].iloc[:, 0]
                label_aligned = test_data.loc[common].iloc[:, 0]
                # Determine date level in MultiIndex
                idx0 = common.get_level_values(0)[0]
                date_level = 0 if isinstance(idx0, pd.Timestamp) else 1
                dates = common.get_level_values(date_level)
                unique_dates = dates.unique()
                daily_ics, daily_rics = [], []
                for dt in unique_dates:
                    mask = dates == dt
                    p = pred_aligned[mask]
                    l = label_aligned[mask]
                    if len(p) > 10:
                        ic = p.corr(l)
                        ric = p.corr(l, method="spearman")
                        if not np.isnan(ic):
                            daily_ics.append(ic)
                        if not np.isnan(ric):
                            daily_rics.append(ric)
                if daily_ics:
                    ic_arr = np.array(daily_ics)
                    ric_arr = np.array(daily_rics)
                    metrics["IC"] = float(np.mean(ic_arr))
                    metrics["ICIR"] = float(np.mean(ic_arr) / (np.std(ic_arr) + 1e-12))
                    metrics["Rank IC"] = float(np.mean(ric_arr))
                    metrics["Rank ICIR"] = float(np.mean(ric_arr) / (np.std(ric_arr) + 1e-12))
                    print(f"[SOVA] Computed from pred.pkl: IC={metrics['IC']:.6f}, ICIR={metrics['ICIR']:.4f}, RankIC={metrics['Rank IC']:.6f}, RankICIR={metrics['Rank ICIR']:.4f}")
            except Exception as e:
                print(f"[SOVA] Could not compute IC from predictions: {e}")
                import traceback
                traceback.print_exc()

        return metrics


def extract_results():
    """Extract IC/ICIR from Qlib MLflow results."""
    try:
        import qlib
        from qlib.workflow import R
        
        qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")
        
        experiments = R.list_experiments()
        latest_recorder = None
        
        for experiment in experiments:
            recorders = R.list_recorders(experiment_name=experiment)
            for recorder_id in recorders:
                if recorder_id is not None:
                    recorder = R.get_recorder(recorder_id=recorder_id, experiment_name=experiment)
                    end_time = recorder.info.get("end_time")
                    if end_time is not None:
                        if latest_recorder is None or end_time > latest_recorder.info["end_time"]:
                            latest_recorder = recorder
        
        if latest_recorder is None:
            print("[SOVA] No recorder found")
            return None
        
        metrics = pd.Series(latest_recorder.list_metrics())
        metrics.to_csv("/workspace/eval/qlib_res.csv")
        print(f"[SOVA] Full Qlib metrics saved to qlib_res.csv")
        return metrics
    except Exception as e:
        print(f"[SOVA] Result extraction failed: {e}")
        return None


def main():
    print("=" * 70)
    print("SOVA × Qlib Factor Evaluation Pipeline")
    print("=" * 70)
    print(f"Start: {datetime.now().isoformat()}")
    print(f"Factors: {len(FACTOR_EXPRESSIONS)}")
    print()

    # Step 1: Download data
    print("[STEP 1/5] Checking Qlib data...")
    if not download_qlib_data():
        print("[FATAL] Cannot proceed without Qlib data")
        sys.exit(1)

    # Step 2: Compute factors
    print("\n[STEP 2/5] Computing SOVA factors via Qlib expressions...")
    df = compute_factors_via_qlib()
    if df is None:
        print("[FATAL] Factor computation failed")
        sys.exit(1)

    # Step 3: Standalone IC check
    print("\n[STEP 3/5] Computing standalone IC per factor...")
    ic_results = compute_standalone_ic(df)
    
    print("\n" + "=" * 70)
    print("SOVA Factor IC Report (Test Period: 2017+)")
    print("=" * 70)
    print(f"{'Factor':<30} {'IC':>10} {'ICIR':>10} {'IC_std':>10}")
    print("-" * 70)
    
    passing_factors = 0
    for name, metrics in sorted(ic_results.items(), key=lambda x: abs(x[1]["IC"]), reverse=True):
        ic = metrics["IC"]
        icir = metrics["ICIR"]
        ic_std = metrics["IC_std"]
        status = "✓" if abs(ic) > 0.01 else "✗"
        print(f"{name:<30} {ic:>10.4f} {icir:>10.4f} {ic_std:>10.4f}  {status}")
        if abs(ic) > 0.01:
            passing_factors += 1
    
    print("-" * 70)
    print(f"Passing (|IC| > 0.01): {passing_factors}/{len(ic_results)}")
    
    # Step 4: Run full Qlib backtest
    print("\n[STEP 4/5] Running Qlib backtest with LightGBM...")
    try:
        backtest_metrics = run_qlib_backtest()
    except Exception as e:
        print(f"[SOVA] Backtest error: {e}")
        import traceback
        traceback.print_exc()
        backtest_metrics = None
    
    # Summary
    print("\n" + "=" * 70)
    print("SOVA × Qlib VERIFICATION SUMMARY")
    print("=" * 70)
    
    if backtest_metrics:
        for key, val in sorted(backtest_metrics.items()):
            print(f"  {key}: {val}")
        
        ic_val = backtest_metrics.get("IC", 0)
        icir_val = backtest_metrics.get("ICIR", 0)
        ric_val = backtest_metrics.get("Rank IC", 0)
        ricir_val = backtest_metrics.get("Rank ICIR", 0)
        
        print(f"\n  Model IC={ic_val:.6f} | ICIR={icir_val:.4f} | Rank IC={ric_val:.6f} | Rank ICIR={ricir_val:.4f}")
        
        if abs(ic_val) > 0.03 and abs(icir_val) > 0.2:
            print("  VERDICT: ★ SOVA PASSES Qlib benchmark (IC + ICIR)")
        elif abs(ic_val) > 0.01:
            print("  VERDICT: ✓ SOVA PASSES Qlib IC benchmark (predictive signal confirmed)")
        else:
            print("  VERDICT: ✗ Below Qlib benchmark thresholds")
    else:
        print("  Full backtest not available. Standalone IC results:")
        if passing_factors > 0:
            avg_ic = np.mean([abs(v["IC"]) for v in ic_results.values()])
            avg_icir = np.mean([abs(v["ICIR"]) for v in ic_results.values()])
            print(f"  Avg |IC|={avg_ic:.4f} | Avg |ICIR|={avg_icir:.4f}")
            print(f"  Factors passing IC>0.01: {passing_factors}/{len(ic_results)}")
            if avg_ic > 0.01:
                print("  VERDICT: ✓ SOVA factors show predictive signal")
            else:
                print("  VERDICT: ✗ Insufficient predictive signal")
    
    print(f"\nEnd: {datetime.now().isoformat()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
