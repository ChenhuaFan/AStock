# Limit-up Signal Bot

Standalone intraday A-share limit-up signal bot.

It reads the fixed stock universe from `config/universe.xlsx`, dynamically fetches recent market data through AkShare, computes the same dimensionless `window_60` features used in research, scores every stock with the bundled LightGBM models, and pushes the top candidates through Bark.

## Strategy

- Universe: stocks in `config/universe.xlsx`.
- Data: fetched dynamically; no dependency on local historical `.npy` files.
- Default source in automation: `intraday`, which fetches historical daily bars and appends a temporary current-day bar aggregated from 1-minute data through 14:30.
- Model: LightGBM raw/window_60 up model plus LightGBM raw/window_60 down-risk model.
- Rank score: `up_score * (1 - down_risk_score)`.
- Default filter: `up_score >= 0.60` and `down_risk_score <= 0.10`.
- Default actionability filter: exclude stocks whose latest daily gain is `>= 9.8%`, because they are already near limit-up.
- Default output: top 10 candidates.

## Local Run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

PYTHONPATH=src python -m limitup_signal_bot.run_daily \
  --dry-run \
  --limit 5 \
  --data-source intraday \
  --intraday-time 14:30 \
  --workers 4 \
  --progress-every 1
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

It runs at `06:30 UTC` on weekdays, which is `14:30 Asia/Shanghai`, for the final 30 minutes decision window.

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

In intraday mode, the bot uses complete historical daily bars plus a temporary current-day daily bar aggregated from minute data between `09:30` and `--intraday-time`. The temporary bar is an approximation for decision-making before the close.

The data fetcher supports `--data-source intraday`, `--data-source daily`, `--data-source hist`, and `--data-source auto`. Increase `--workers` cautiously if the vendor endpoint is healthy; decrease it if you see throttling or many connection errors.
