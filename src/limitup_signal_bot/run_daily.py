from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
import json
import os
import pickle
from pathlib import Path
import time
import warnings
from zoneinfo import ZoneInfo

import numpy as np

from .data_fetch import default_start_date, fetch_history, to_raw_frame, to_raw_matrix
from .features import batch_to_seq_features, batch_to_tab_features
from .notify import send_bark
from .universe import Stock, load_universe


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CN_TZ = ZoneInfo("Asia/Shanghai")


def log(message: str) -> None:
    timestamp = datetime.now(CN_TZ).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def load_model_bundle(model_dir: Path) -> tuple[object, np.ndarray, np.ndarray]:
    with (model_dir / "model.pkl").open("rb") as f:
        model = pickle.load(f)
    scaler = np.load(model_dir / "scaler.npz")
    return model, scaler["mean"].astype(np.float32), scaler["std"].astype(np.float32)


def predict_score(model: object, x_tab: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    x = ((x_tab - mean) / std).astype(np.float32)
    return float(model.predict_proba(x)[0, 1])


def build_feature(raw_matrix: np.ndarray, window: int) -> np.ndarray:
    if raw_matrix.shape[0] < window:
        raise ValueError(f"not enough trading rows: got {raw_matrix.shape[0]}, need {window}")
    raw_window = raw_matrix[-window:, :][None, :, :].astype(np.float32)
    x_seq = batch_to_seq_features(raw_window)
    return batch_to_tab_features(raw_window, x_seq)


def score_stock(
    stock: Stock,
    start_date: str,
    end_date: str,
    adjust: str,
    data_source: str,
    window: int,
    up_bundle: tuple[object, np.ndarray, np.ndarray],
    down_bundle: tuple[object, np.ndarray, np.ndarray],
) -> dict[str, object]:
    df = fetch_history(stock.ticker, start_date=start_date, end_date=end_date, adjust=adjust, source=data_source)
    raw_matrix = to_raw_matrix(df)
    x_tab = build_feature(raw_matrix, window)
    up_model, up_mean, up_std = up_bundle
    down_model, down_mean, down_std = down_bundle
    up_score = predict_score(up_model, x_tab, up_mean, up_std)
    down_score = predict_score(down_model, x_tab, down_mean, down_std)
    raw_frame = to_raw_frame(df)
    latest = raw_frame.iloc[-1].to_dict()
    return {
        "ticker": stock.ticker,
        "name": stock.name,
        "latest_date": latest["date"].date().isoformat(),
        "latest_close": float(latest["close"]),
        "latest_pct_chg": float(latest["pct_chg"]),
        "latest_turnover": float(latest["turnover"]),
        "up_score": up_score,
        "down_risk_score": down_score,
        "combined_score": up_score * (1.0 - down_score),
        "rows": int(raw_matrix.shape[0]),
    }


def format_notification(rows: list[dict[str, object]], run_date: str, min_up_score: float, max_down_risk: float) -> tuple[str, str]:
    title = f"A股涨停候选 {run_date}"
    if not rows:
        return title, f"今日没有满足 up>={min_up_score:.2f}, risk<={max_down_risk:.2f} 的候选。"
    lines = [
        f"规则: up>={min_up_score:.2f}, risk<={max_down_risk:.2f}",
        "Top candidates:",
    ]
    for idx, row in enumerate(rows, start=1):
        lines.append(
            f"{idx}. {row['ticker']} {row['name']} "
            f"up={row['up_score']:.3f} risk={row['down_risk_score']:.3f} "
            f"score={row['combined_score']:.3f}"
        )
    return title, "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch latest A-share data and push daily limit-up candidates.")
    parser.add_argument("--universe", default=PROJECT_ROOT / "config/universe.xlsx", type=Path)
    parser.add_argument("--up-model-dir", default=PROJECT_ROOT / "models/up_window60", type=Path)
    parser.add_argument("--down-model-dir", default=PROJECT_ROOT / "models/down_window60", type=Path)
    parser.add_argument("--output-dir", default=PROJECT_ROOT / "outputs", type=Path)
    parser.add_argument("--window", default=60, type=int)
    parser.add_argument("--lookback-calendar-days", default=260, type=int)
    parser.add_argument("--end-date", default="")
    parser.add_argument("--adjust", default="")
    parser.add_argument("--data-source", default="daily", choices=["daily", "hist", "auto"])
    parser.add_argument("--top-n", default=10, type=int)
    parser.add_argument("--min-up-score", default=0.60, type=float)
    parser.add_argument("--max-down-risk", default=0.10, type=float)
    parser.add_argument("--bark-url", default=os.environ.get("BARK_URL", ""))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", default=0, type=int, help="Debug only: process the first N universe tickers.")
    parser.add_argument("--sleep-seconds", default=0.05, type=float)
    parser.add_argument("--progress-every", default=25, type=int)
    parser.add_argument("--workers", default=8, type=int)
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    args = parse_args()
    now = datetime.now(CN_TZ)
    end_date = args.end_date or now.strftime("%Y%m%d")
    start_date = default_start_date(datetime.strptime(end_date, "%Y%m%d").date(), args.lookback_calendar_days)

    log("Starting daily limit-up signal job")
    log(
        "Config: "
        f"window={args.window}, start_date={start_date}, end_date={end_date}, "
        f"top_n={args.top_n}, min_up={args.min_up_score:.2f}, max_risk={args.max_down_risk:.2f}, "
        f"data_source={args.data_source}, workers={args.workers}, dry_run={args.dry_run}"
    )

    log(f"Loading universe from {args.universe}")
    universe = load_universe(args.universe)
    if args.limit:
        universe = universe[: args.limit]
        log(f"Debug limit enabled: processing first {len(universe)} tickers")
    log(f"Universe loaded: {len(universe)} tickers")
    log(f"Loading up model from {args.up_model_dir}")
    up_bundle = load_model_bundle(args.up_model_dir)
    log(f"Loading down-risk model from {args.down_model_dir}")
    down_bundle = load_model_bundle(args.down_model_dir)
    log("Models loaded")

    rows: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    started = time.monotonic()
    workers = max(1, min(args.workers, len(universe) or 1))

    def run_one(stock: Stock) -> dict[str, object]:
        return score_stock(
            stock=stock,
            start_date=start_date,
            end_date=end_date,
            adjust=args.adjust,
            data_source=args.data_source,
            window=args.window,
            up_bundle=up_bundle,
            down_bundle=down_bundle,
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_stock = {executor.submit(run_one, stock): stock for stock in universe}
        for idx, future in enumerate(as_completed(future_to_stock), start=1):
            stock = future_to_stock[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                errors.append({"ticker": stock.ticker, "name": stock.name, "error": repr(exc)})
                log(f"WARNING: failed {stock.ticker} {stock.name}: {exc!r}")
            if args.progress_every > 0 and (idx == 1 or idx % args.progress_every == 0 or idx == len(universe)):
                elapsed = time.monotonic() - started
                speed = idx / elapsed if elapsed > 0 else 0.0
                eta = (len(universe) - idx) / speed if speed > 0 else 0.0
                log(
                    f"Progress: {idx}/{len(universe)} tickers, "
                    f"scored={len(rows)}, errors={len(errors)}, elapsed={elapsed:.1f}s, eta={eta:.1f}s"
                )
            if args.sleep_seconds > 0 and idx < len(universe):
                time.sleep(args.sleep_seconds)

    log("Scoring finished; ranking candidates")
    rows.sort(key=lambda row: row["combined_score"], reverse=True)
    candidates = [
        row
        for row in rows
        if row["up_score"] >= args.min_up_score and row["down_risk_score"] <= args.max_down_risk
    ][: args.top_n]
    fallback_used = False
    if not candidates:
        candidates = rows[: args.top_n]
        fallback_used = True
        log("No ticker matched strict thresholds; falling back to top ranked rows")
    log(f"Selected {len(candidates)} candidates; fallback_used={fallback_used}")
    for idx, row in enumerate(candidates[: min(5, len(candidates))], start=1):
        log(
            f"Top {idx}: {row['ticker']} {row['name']} "
            f"up={row['up_score']:.4f}, risk={row['down_risk_score']:.4f}, "
            f"score={row['combined_score']:.4f}, latest={row['latest_date']}"
        )

    run_date = now.strftime("%Y-%m-%d")
    title, body = format_notification(candidates, run_date, args.min_up_score, args.max_down_risk)
    notification = {"sent": False, "dry_run": args.dry_run}
    if not args.dry_run:
        log("Sending Bark notification")
        notification = send_bark(args.bark_url, title=title, body=body)
        log(f"Bark response: {notification}")
    else:
        log("Dry-run enabled; skipping Bark notification")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "run_at": now.isoformat(),
        "start_date": start_date,
        "end_date": end_date,
        "window": args.window,
        "universe_file": str(args.universe),
        "universe_size": len(universe),
        "scored": len(rows),
        "errors": errors,
        "fallback_used": fallback_used,
        "rules": {
            "rank_score": "up_score * (1 - down_risk_score)",
            "min_up_score": args.min_up_score,
            "max_down_risk": args.max_down_risk,
            "top_n": args.top_n,
        },
        "top": candidates,
        "notification": notification,
    }
    output_path = args.output_dir / f"signals_{run_date}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    log(f"Wrote output: {output_path}")
    log("Daily limit-up signal job finished")
    print(json.dumps({"output": str(output_path), "top": candidates, "errors": errors[:10], "notification": notification}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
