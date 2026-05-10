# Limit-up Signal Bot

Standalone daily A-share limit-up signal bot.

It reads the fixed stock universe from `config/universe.xlsx`, dynamically fetches recent daily bars through AkShare, computes the same dimensionless `window_60` features used in research, scores every stock with the bundled LightGBM models, and pushes the top candidates through Bark.

## Strategy

- Universe: stocks in `config/universe.xlsx`.
- Data: fetched dynamically with `akshare.stock_zh_a_hist`; no dependency on local historical `.npy` files.
- Default source in automation: `akshare.stock_zh_a_daily`, fetched concurrently with 8 workers for speed.
- Model: LightGBM raw/window_60 up model plus LightGBM raw/window_60 down-risk model.
- Rank score: `up_score * (1 - down_risk_score)`.
- Default filter: `up_score >= 0.60` and `down_risk_score <= 0.10`.
- Default output: top 10 candidates.

## Local Run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

PYTHONPATH=src python -m limitup_signal_bot.run_daily --dry-run --limit 5 --workers 4 --progress-every 1
```

Send a Bark notification locally:

```bash
export BARK_URL="https://api.day.app/<your-device-key>"
PYTHONPATH=src python -m limitup_signal_bot.run_daily
```

`BARK_URL` may also contain placeholders:

```text
https://api.day.app/<your-device-key>/{title}/{body}
```

## GitHub Actions

The workflow is in `.github/workflows/daily-signal.yml`.

It runs at `08:10 UTC` on weekdays, which is `16:10 Asia/Shanghai`, after A-share close.

Before enabling the action, add this repository secret:

```text
BARK_URL=https://api.day.app/<your-device-key>
```

You can also run it manually from the Actions tab with `workflow_dispatch`.

## Files

- `config/universe.xlsx`: current 834-stock universe.
- `models/up_window60`: bundled limit-up LightGBM model and scaler.
- `models/down_window60`: bundled limit-down risk model and scaler.
- `src/limitup_signal_bot`: fetch, feature, predict, notify code.
- `outputs`: JSON output from each run.

## Notes

AkShare data availability can lag after market close. If a vendor delay happens, the bot still scores the latest available trading day in the fetched data and includes `latest_date` in the output.

The data fetcher supports `--data-source daily`, `--data-source hist`, and `--data-source auto`. The GitHub Action uses `daily` because it is much faster and more stable for the full universe. Increase `--workers` cautiously if the vendor endpoint is healthy; decrease it if you see throttling or many connection errors.
