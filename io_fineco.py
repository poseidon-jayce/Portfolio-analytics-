from __future__ import annotations

import pandas as pd

FINECO_HEADER_ROW = 4  # Excel row 5 (0-based)
SHEET_NAME = "Movimenti Dossier Titoli"


def load_isin_ticker_map(path_csv: str) -> dict[str, str]:
    m = pd.read_csv(path_csv)
    m["isin"] = m["isin"].astype(str).str.strip()
    m["ticker"] = m["ticker"].astype(str).str.strip()
    return dict(zip(m["isin"], m["ticker"], strict=False))


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=True).dt.tz_localize(None)


def load_fineco_movements_xlsx(
    xlsx_path: str,
    isin_to_ticker: dict[str, str],
) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name=SHEET_NAME, header=FINECO_HEADER_ROW)

    # Rename Italian columns → English
    df = df.rename(
        columns={
            "Descrizione": "description",
            "Titolo": "name",
            "Isin": "isin",
            "Segno": "sign",
            "Divisa": "currency",
            "Data valuta": "value_date",
            "Operazione": "operation_date",
            "Controvalore": "amount",
            "Quantita": "quantity",
            "Prezzo": "price",
        }
    )

    df = df.dropna(subset=["description"]).copy()

    df["date"] = _to_datetime(df["value_date"]).fillna(
        _to_datetime(df["operation_date"])
    )
    df = df.dropna(subset=["date"]).copy()
    df["date"] = df["date"].dt.normalize()

    for c in ["description", "name", "isin", "sign", "currency"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    df["amount_eur"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    df["quantity"] = pd.to_numeric(df.get("quantity"), errors="coerce").fillna(0.0)
    df["price"] = pd.to_numeric(df.get("price"), errors="coerce").fillna(0.0)

    fee_cols = [
        "Commissioni Fondi Sw/Ingr/Uscita",
        "Commissioni Fondi Banca Corrispondente",
        "Spese Fondi Sgr",
        "Commissioni amministrato",
    ]
    present_fee_cols = [c for c in fee_cols if c in df.columns]
    if present_fee_cols:
        df["fees_eur"] = (
            df[present_fee_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .sum(axis=1)
        )
    else:
        df["fees_eur"] = 0.0

    df["ticker"] = df["isin"].map(isin_to_ticker)

    desc = df["description"].str.upper()
    sign = df["sign"].str.upper()

    df["type"] = "UNKNOWN"
    df.loc[desc.eq("DIVIDENDO"), "type"] = "DIVIDEND"

    is_trade = desc.str.contains("COMPRAVENDITA", na=False)
    df.loc[is_trade & sign.eq("A"), "type"] = "BUY"
    df.loc[is_trade & sign.eq("V"), "type"] = "SELL"

    # Cash flow convention
    df["cash_flow_eur"] = 0.0
    df.loc[df["type"].eq("DIVIDEND"), "cash_flow_eur"] = df.loc[
        df["type"].eq("DIVIDEND"), "amount_eur"
    ]
    df.loc[df["type"].eq("BUY"), "cash_flow_eur"] = -(
        df.loc[df["type"].eq("BUY"), "amount_eur"]
        + df.loc[df["type"].eq("BUY"), "fees_eur"]
    )
    df.loc[df["type"].eq("SELL"), "cash_flow_eur"] = +(
        df.loc[df["type"].eq("SELL"), "amount_eur"]
        - df.loc[df["type"].eq("SELL"), "fees_eur"]
    )

    df["shares_delta"] = 0.0
    df.loc[df["type"].eq("BUY"), "shares_delta"] = df.loc[
        df["type"].eq("BUY"), "quantity"
    ]
    df.loc[df["type"].eq("SELL"), "shares_delta"] = -df.loc[
        df["type"].eq("SELL"), "quantity"
    ]

    ledger = df[
        [
            "date",
            "type",
            "isin",
            "ticker",
            "name",
            "currency",
            "quantity",
            "price",
            "amount_eur",
            "fees_eur",
            "shares_delta",
            "cash_flow_eur",
        ]
    ].rename(columns={"currency": "trade_ccy"})

    ledger = ledger.sort_values(["date", "type", "ticker"]).reset_index(drop=True)
    return ledger