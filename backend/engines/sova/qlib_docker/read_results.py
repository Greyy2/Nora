"""
Read Qlib experiment results and print metrics summary.
"""
import sys
import pandas as pd


def read_results():
    try:
        import qlib
        from qlib.workflow import R

        qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")

        experiments = R.list_experiments()
        latest = None
        latest_exp = None

        for exp in experiments:
            recorders = R.list_recorders(experiment_name=exp)
            for rid, rec in recorders.items():
                end_time = rec.info.get("end_time")
                if end_time is not None:
                    if latest is None or end_time > latest.info["end_time"]:
                        latest = rec
                        latest_exp = exp

        if latest is None:
            print("No completed experiment found.")
            return

        print(f"Experiment: {latest_exp}")
        print(f"Recorder: {latest.info.get('id', 'N/A')}")
        print()

        metrics = latest.list_metrics()
        df = pd.Series(metrics)
        df.to_csv("/workspace/eval/qlib_res.csv")

        print("=" * 50)
        print("QLIB EXPERIMENT RESULTS")
        print("=" * 50)

        key_metrics = [
            "IC", "ICIR", "Rank IC", "Rank ICIR",
            "1day.excess_return_with_cost.annualized_return",
            "1day.excess_return_with_cost.max_drawdown",
            "1day.excess_return_with_cost.information_ratio",
            "1day.excess_return_without_cost.annualized_return",
        ]

        for m in key_metrics:
            if m in metrics:
                print(f"  {m}: {metrics[m]:.6f}")

        print()
        print("All metrics:")
        for k, v in sorted(metrics.items()):
            print(f"  {k}: {v}")

        # Verdict
        ic = metrics.get("IC", 0)
        icir = metrics.get("ICIR", 0)
        ann_ret = metrics.get("1day.excess_return_with_cost.annualized_return", 0)

        print()
        print("=" * 50)
        print("VERDICT")
        print("=" * 50)
        print(f"  IC:   {ic:.6f}  {'PASS' if abs(ic) > 0.01 else 'FAIL'} (threshold: |IC| > 0.01)")
        print(f"  ICIR: {icir:.6f}  {'PASS' if abs(icir) > 0.5 else 'FAIL'} (threshold: |ICIR| > 0.5)")
        print(f"  AnnRet: {ann_ret:.6f}  {'PASS' if ann_ret > 0 else 'FAIL'} (threshold: > 0)")

        if abs(ic) > 0.01 and abs(icir) > 0.5 and ann_ret > 0:
            print("\n  ★ OVERALL: PASS — SOVA factors meet Qlib benchmark standards")
        elif abs(ic) > 0.01:
            print("\n  ◆ OVERALL: PARTIAL — Predictive signal exists but below full benchmark")
        else:
            print("\n  ✗ OVERALL: FAIL — Insufficient predictive signal")

    except Exception as e:
        print(f"Error reading results: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    read_results()
