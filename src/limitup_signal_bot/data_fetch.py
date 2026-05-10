from __future__ import annotations

from datetime import date, timedelta
import time

import akshare as ak
import numpy as np
import pandas as pd


RAW_COLUMNS = ["open", "close", "high", "low", "volume", "amount", "amplitude", "pct_chg", "change", "turnover"]


def default_start_date(end: date, calendar_days: int = 260) -> str:
    return (end - timedelta(days=calendar_days)).strftime("%Y%m%d")


def _exchange_symbol(symbol: str) -> str:
    if symbol.startswith("6"):
        return f"sh{symbol}"
    if symbol.startswith(("4", "8")):
        return f"bj{symbol}"
    return f"sz{symbol}"


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "振幅": "amplitude",
        "涨跌幅": "pct_chg",
        "涨跌额": "change",
        "换手率": "turnover",
    }
    return df.rename(columns=rename).copy()


def to_raw_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = _rename_columns(df)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date")
    for column in ["open", "close", "high", "low", "volume", "amount", "turnover"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if "amplitude" not in frame.columns or "pct_chg" not in frame.columns or "change" not in frame.columns:
        prev_close = frame["close"].shift(1)
        frame["change"] = frame["close"] - prev_close
        frame["pct_chg"] = (frame["close"] / prev_close - 1.0) * 100.0
        frame["amplitude"] = (frame["high"] - frame["low"]) / prev_close * 100.0
        frame = frame.iloc[1:].copy()
    else:
        for column in ["amplitude", "pct_chg", "change"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # stock_zh_a_daily reports volume in shares and turnover as a fraction.
    if frame["volume"].median() > 10_000_000:
        frame["volume"] = frame["volume"] / 100.0
    if frame["turnover"].median() < 0.5:
        frame["turnover"] = frame["turnover"] * 100.0

    frame = frame.dropna(subset=RAW_COLUMNS)
    return frame


def to_raw_matrix(df: pd.DataFrame) -> np.ndarray:
    frame = to_raw_frame(df)
    return frame[RAW_COLUMNS].to_numpy(dtype=np.float32)


def fetch_history(symbol: str, start_date: str, end_date: str, adjust: str = "", retries: int = 2) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
            )
        except Exception as exc:
            last_error = exc
            time.sleep(0.5 + attempt * 0.5)

    try:
        return ak.stock_zh_a_daily(
            symbol=_exchange_symbol(symbol),
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )
    except Exception as fallback_error:
        raise RuntimeError(f"hist failed: {last_error!r}; daily failed: {fallback_error!r}") from fallback_error
