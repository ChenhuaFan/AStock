from __future__ import annotations

from datetime import date, datetime, timedelta
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


def _fetch_daily(symbol: str, start_date: str, end_date: str, adjust: str, retries: int) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return ak.stock_zh_a_daily(
                symbol=_exchange_symbol(symbol),
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
            )
        except Exception as exc:
            last_error = exc
            time.sleep(0.5 + attempt * 0.5)
    raise RuntimeError(f"daily failed: {last_error!r}")


def _normalize_intraday_time(intraday_time: str) -> str:
    if len(intraday_time.split(":")) == 2:
        return f"{intraday_time}:00"
    return intraday_time


def _fetch_intraday_minutes(
    symbol: str,
    trading_day: date,
    intraday_time: str,
    adjust: str,
    retries: int,
) -> pd.DataFrame:
    end_time = _normalize_intraday_time(intraday_time)
    start_date = f"{trading_day:%Y-%m-%d} 09:30:00"
    end_date = f"{trading_day:%Y-%m-%d} {end_time}"

    fallback_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            fallback = ak.stock_zh_a_minute(
                symbol=_exchange_symbol(symbol),
                period="1",
                adjust=adjust,
            )
            fallback["day"] = pd.to_datetime(fallback["day"], errors="coerce")
            start_dt = pd.Timestamp(start_date)
            end_dt = pd.Timestamp(end_date)
            filtered = fallback[(fallback["day"] >= start_dt) & (fallback["day"] <= end_dt)].copy()
            if not filtered.empty:
                return filtered
            fallback_error = ValueError("stock_zh_a_minute returned no rows for requested window")
        except Exception as exc:
            fallback_error = exc
            time.sleep(0.5 + attempt * 0.5)

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return ak.stock_zh_a_hist_min_em(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                period="1",
                adjust=adjust,
            )
        except Exception as exc:
            last_error = exc
            time.sleep(0.5 + attempt * 0.5)
    raise RuntimeError(f"minute failed: {fallback_error!r}; hist_min failed: {last_error!r}")


def _append_intraday_bar(
    daily_df: pd.DataFrame,
    minute_df: pd.DataFrame,
    trading_day: date,
) -> pd.DataFrame:
    daily_frame = to_raw_frame(daily_df)
    daily_frame = daily_frame[daily_frame["date"].dt.date < trading_day].copy()
    if daily_frame.empty:
        raise ValueError("not enough previous daily rows to append intraday bar")
    if minute_df.empty:
        raise ValueError("empty intraday minute data")

    minutes = minute_df.rename(
        columns={
            "时间": "datetime",
            "day": "datetime",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
        }
    ).copy()
    minutes["datetime"] = pd.to_datetime(minutes["datetime"], errors="coerce")
    minutes = minutes.dropna(subset=["datetime"]).sort_values("datetime")
    minutes = minutes[minutes["datetime"].dt.date == trading_day].copy()
    for column in ["open", "close", "high", "low", "volume", "amount"]:
        minutes[column] = pd.to_numeric(minutes[column], errors="coerce")
    minutes = minutes.dropna(subset=["open", "close", "high", "low", "volume", "amount"])
    if minutes.empty:
        raise ValueError("empty intraday minute data after cleaning")

    prev_close = float(daily_frame.iloc[-1]["close"])
    open_ = float(minutes.iloc[0]["open"])
    close = float(minutes.iloc[-1]["close"])
    high = float(minutes["high"].max())
    low = float(minutes["low"].min())
    volume = float(minutes["volume"].sum())
    amount = float(minutes["amount"].sum())
    unit_ratio = float(
        np.nanmedian(
            minutes["amount"].to_numpy(dtype=float)
            / np.maximum(
                minutes["volume"].to_numpy(dtype=float) * minutes["close"].to_numpy(dtype=float),
                1e-12,
            )
        )
    )
    if unit_ratio < 20.0:
        volume = volume / 100.0

    daily_raw = _rename_columns(daily_df)
    daily_raw["date"] = pd.to_datetime(daily_raw["date"], errors="coerce")
    daily_raw = daily_raw.dropna(subset=["date"]).sort_values("date")
    daily_raw = daily_raw[daily_raw["date"].dt.date < trading_day]
    outstanding_share = float(daily_raw.iloc[-1].get("outstanding_share", 0.0)) if not daily_raw.empty else 0.0
    turnover = (volume * 10000.0 / outstanding_share) if outstanding_share > 0 else 0.0

    intraday_row = {
        "date": pd.Timestamp(trading_day),
        "open": open_,
        "close": close,
        "high": high,
        "low": low,
        "volume": volume,
        "amount": amount,
        "amplitude": (high - low) / prev_close * 100.0 if abs(prev_close) > 1e-12 else 0.0,
        "pct_chg": (close / prev_close - 1.0) * 100.0 if abs(prev_close) > 1e-12 else 0.0,
        "change": close - prev_close,
        "turnover": turnover,
    }
    return pd.concat([daily_frame, pd.DataFrame([intraday_row])], ignore_index=True)


def fetch_intraday_history(
    symbol: str,
    start_date: str,
    end_date: str,
    adjust: str,
    intraday_time: str,
    retries: int,
) -> pd.DataFrame:
    trading_day = datetime.strptime(end_date, "%Y%m%d").date()
    daily_df = _fetch_daily(symbol, start_date, end_date, adjust, retries)
    minute_df = _fetch_intraday_minutes(symbol, trading_day, intraday_time, adjust, retries)
    return _append_intraday_bar(daily_df, minute_df, trading_day)


def fetch_history(
    symbol: str,
    start_date: str,
    end_date: str,
    adjust: str = "",
    source: str = "daily",
    intraday_time: str = "14:30",
    retries: int = 1,
) -> pd.DataFrame:
    if source not in {"daily", "hist", "auto", "intraday"}:
        raise ValueError(f"Unsupported data source: {source}")

    if source == "daily":
        return _fetch_daily(symbol, start_date, end_date, adjust, retries)

    if source == "intraday":
        return fetch_intraday_history(symbol, start_date, end_date, adjust, intraday_time, retries)

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

    if source == "hist":
        raise RuntimeError(f"hist failed: {last_error!r}")

    try:
        return ak.stock_zh_a_daily(
            symbol=_exchange_symbol(symbol),
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )
    except Exception as fallback_error:
        raise RuntimeError(f"hist failed: {last_error!r}; daily failed: {fallback_error!r}") from fallback_error
