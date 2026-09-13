"""Sanitized double-entry ledger snapshot adapter.

The adapter reads a local-only, sanitized snapshot that models core double-entry
entities:
 - ledgers
 - accounts
 - transactions
 - entries
 - optional reversals

It enforces accounting invariants before feature extraction:
 - at least two entries per transaction
 - positive integer base-unit amounts
 - debit total equals credit total
 - single currency per transaction and consistency with ledger/account currency

Labels are operational risk proxies derived from settlement/reversal outcomes and are
not confirmed fraud ground truth.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DOUBLE_ENTRY_SNAPSHOT_SCHEMA_VERSION = "double-entry-ledger-snapshot-v1"

DOUBLE_ENTRY_FEATURE_NAMES = (
    "total_amount_base_units",
    "entry_count",
    "debit_entry_count",
    "credit_entry_count",
    "debit_asset_entries",
    "debit_liability_entries",
    "debit_revenue_entries",
    "debit_expense_entries",
    "credit_asset_entries",
    "credit_liability_entries",
    "credit_revenue_entries",
    "credit_expense_entries",
    "unique_account_count",
    "prior_account_tx_count_1h",
    "prior_account_tx_count_24h",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "has_session_reference",
    "has_external_reference",
    "tx_type_card_payment",
    "tx_type_bank_transfer",
    "tx_type_cash_withdrawal",
    "tx_type_fee",
    "tx_type_adjustment",
    "tx_type_other",
)

KNOWN_TRANSACTION_TYPES = (
    "card_payment",
    "bank_transfer",
    "cash_withdrawal",
    "fee",
    "adjustment",
)

ACCOUNT_TYPE_BUCKETS = ("asset", "liability", "revenue", "expense")

TRANSACTION_REQUIRED_COLUMNS = (
    "transaction_id",
    "ledger_id",
    "transaction_type",
    "created_at",
    "currency",
    "settlement_state",
    "session_reference",
    "external_reference",
)

ACCOUNT_REQUIRED_COLUMNS = ("account_id", "ledger_id", "account_type", "currency")
LEDGER_REQUIRED_COLUMNS = ("ledger_id", "currency")
ENTRY_REQUIRED_COLUMNS = (
    "entry_id",
    "transaction_id",
    "account_id",
    "entry_type",
    "amount_base_units",
    "currency",
)
REVERSAL_REQUIRED_COLUMNS = ("reversal_id", "transaction_id", "reversed_at", "reason_code")


class SnapshotContractError(ValueError):
    """Raised when snapshot files violate ledger contract invariants."""


@dataclasses.dataclass(frozen=True)
class SnapshotDataset:
    features: pd.DataFrame
    labels: np.ndarray
    metadata: dict[str, Any]


def _resolve_snapshot_path(dataset_path: str) -> Path:
    path = Path(dataset_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...], name: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SnapshotContractError(f"{name} is missing required columns: {', '.join(missing)}")


def _load_csv(snapshot_root: Path, filename: str, required_columns: tuple[str, ...], table_name: str) -> pd.DataFrame:
    file_path = snapshot_root / filename
    if not file_path.exists():
        raise SnapshotContractError(f"Missing required snapshot file: {file_path}")
    frame = pd.read_csv(file_path, dtype=str)
    if frame.empty:
        raise SnapshotContractError(f"Snapshot file {file_path} is empty")
    _require_columns(frame, required_columns, table_name)
    return frame


def _parse_iso8601(value: str, context: str) -> datetime:
    raw = str(value).strip()
    if not raw:
        raise SnapshotContractError(f"Missing timestamp for {context}")
    normalized = raw.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as err:
        raise SnapshotContractError(f"Invalid timestamp {raw!r} for {context}") from err
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_entry_amount(value: str, context: str) -> int:
    raw = str(value).strip()
    if raw == "":
        raise SnapshotContractError(f"Missing amount_base_units for {context}")
    if "." in raw:
        raise SnapshotContractError(
            f"amount_base_units must be integer base units for {context}; got decimal value {raw!r}"
        )
    try:
        amount = int(raw)
    except ValueError as err:
        raise SnapshotContractError(f"Invalid integer amount_base_units {raw!r} for {context}") from err
    if amount <= 0:
        raise SnapshotContractError(f"amount_base_units must be > 0 for {context}; got {amount}")
    return amount


def _normalize_side(value: str, context: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in {"debit", "credit"}:
        raise SnapshotContractError(f"Invalid entry_type {value!r} for {context}; expected debit or credit")
    return normalized


def _normalize_account_type(value: str) -> str:
    account_type = str(value).strip().lower()
    if account_type in ACCOUNT_TYPE_BUCKETS:
        return account_type
    return "other"


def _as_presence_flag(value: Any) -> int:
    text = str(value).strip().lower()
    if text in {"", "none", "null", "0", "false", "no"}:
        return 0
    return 1


def load_double_entry_snapshot(dataset_path: str) -> SnapshotDataset:
    """Load and validate a sanitized double-entry ledger snapshot."""
    snapshot_root = _resolve_snapshot_path(dataset_path)
    if not snapshot_root.exists():
        raise SnapshotContractError(f"Snapshot directory does not exist: {snapshot_root}")
    if not snapshot_root.is_dir():
        raise SnapshotContractError(f"Snapshot path is not a directory: {snapshot_root}")

    ledgers = _load_csv(snapshot_root, "ledgers.csv", LEDGER_REQUIRED_COLUMNS, "ledgers")
    accounts = _load_csv(snapshot_root, "accounts.csv", ACCOUNT_REQUIRED_COLUMNS, "accounts")
    transactions = _load_csv(
        snapshot_root, "transactions.csv", TRANSACTION_REQUIRED_COLUMNS, "transactions"
    )
    entries = _load_csv(snapshot_root, "entries.csv", ENTRY_REQUIRED_COLUMNS, "entries")

    reversals_path = snapshot_root / "reversals.csv"
    reversals = pd.DataFrame(columns=REVERSAL_REQUIRED_COLUMNS)
    if reversals_path.exists():
        reversals = pd.read_csv(reversals_path, dtype=str)
        if not reversals.empty:
            _require_columns(reversals, REVERSAL_REQUIRED_COLUMNS, "reversals")

    ledgers = ledgers.copy()
    accounts = accounts.copy()
    transactions = transactions.copy()
    entries = entries.copy()

    for table, key in (
        (ledgers, "ledger_id"),
        (accounts, "account_id"),
        (transactions, "transaction_id"),
        (entries, "entry_id"),
    ):
        if table[key].duplicated().any():
            duplicated = table.loc[table[key].duplicated(), key].iloc[0]
            raise SnapshotContractError(f"Duplicate {key} detected: {duplicated}")

    ledger_currency = {
        str(row["ledger_id"]).strip(): str(row["currency"]).strip().upper()
        for _, row in ledgers.iterrows()
    }
    if not ledger_currency:
        raise SnapshotContractError("No ledgers found in snapshot")

    account_lookup: dict[str, tuple[str, str, str]] = {}
    for _, row in accounts.iterrows():
        account_id = str(row["account_id"]).strip()
        ledger_id = str(row["ledger_id"]).strip()
        currency = str(row["currency"]).strip().upper()
        if ledger_id not in ledger_currency:
            raise SnapshotContractError(f"Account {account_id} references unknown ledger_id {ledger_id}")
        if ledger_currency[ledger_id] != currency:
            raise SnapshotContractError(
                f"Account {account_id} currency {currency} does not match ledger {ledger_id} currency {ledger_currency[ledger_id]}"
            )
        account_lookup[account_id] = (
            ledger_id,
            _normalize_account_type(row["account_type"]),
            currency,
        )

    entries_by_transaction: dict[str, list[dict[str, Any]]] = {}
    for _, row in entries.iterrows():
        transaction_id = str(row["transaction_id"]).strip()
        account_id = str(row["account_id"]).strip()
        context = f"entry_id={row['entry_id']} transaction_id={transaction_id}"
        if account_id not in account_lookup:
            raise SnapshotContractError(f"{context} references unknown account_id {account_id}")

        side = _normalize_side(str(row["entry_type"]), context)
        amount_base_units = _parse_entry_amount(str(row["amount_base_units"]), context)
        currency = str(row["currency"]).strip().upper()
        account_currency = account_lookup[account_id][2]
        if currency != account_currency:
            raise SnapshotContractError(
                f"{context} currency {currency} does not match account {account_id} currency {account_currency}"
            )

        entries_by_transaction.setdefault(transaction_id, []).append(
            {
                "entry_id": str(row["entry_id"]).strip(),
                "account_id": account_id,
                "side": side,
                "amount_base_units": amount_base_units,
                "currency": currency,
            }
        )

    reversal_transaction_ids: set[str] = set()
    if not reversals.empty:
        for _, row in reversals.iterrows():
            transaction_id = str(row["transaction_id"]).strip()
            if transaction_id:
                _parse_iso8601(str(row["reversed_at"]), f"reversal for transaction_id={transaction_id}")
                reversal_transaction_ids.add(transaction_id)

    records: list[dict[str, Any]] = []
    account_history: dict[str, list[datetime]] = {account_id: [] for account_id in account_lookup}

    sorted_transactions = transactions.sort_values(
        by=["created_at", "transaction_id"], kind="mergesort"
    )
    for _, tx_row in sorted_transactions.iterrows():
        transaction_id = str(tx_row["transaction_id"]).strip()
        ledger_id = str(tx_row["ledger_id"]).strip()
        tx_currency = str(tx_row["currency"]).strip().upper()
        transaction_type = str(tx_row["transaction_type"]).strip().lower()
        settlement_state = str(tx_row["settlement_state"]).strip().lower()
        created_at = _parse_iso8601(str(tx_row["created_at"]), f"transaction_id={transaction_id}")

        if ledger_id not in ledger_currency:
            raise SnapshotContractError(
                f"transaction_id={transaction_id} references unknown ledger_id {ledger_id}"
            )
        if ledger_currency[ledger_id] != tx_currency:
            raise SnapshotContractError(
                f"transaction_id={transaction_id} currency {tx_currency} does not match ledger currency {ledger_currency[ledger_id]}"
            )

        tx_entries = entries_by_transaction.get(transaction_id, [])
        if len(tx_entries) < 2:
            raise SnapshotContractError(
                f"transaction_id={transaction_id} must contain at least two entries; got {len(tx_entries)}"
            )

        currencies = {entry["currency"] for entry in tx_entries}
        if len(currencies) != 1:
            raise SnapshotContractError(
                f"transaction_id={transaction_id} has mixed entry currencies {sorted(currencies)}; mixed currencies are disallowed"
            )
        if tx_currency not in currencies:
            raise SnapshotContractError(
                f"transaction_id={transaction_id} currency {tx_currency} does not match entry currency {next(iter(currencies))}"
            )

        debit_total = 0
        credit_total = 0
        debit_counts = {bucket: 0 for bucket in ACCOUNT_TYPE_BUCKETS}
        credit_counts = {bucket: 0 for bucket in ACCOUNT_TYPE_BUCKETS}

        involved_accounts: set[str] = set()
        for entry in tx_entries:
            account_id = entry["account_id"]
            involved_accounts.add(account_id)
            account_type = account_lookup[account_id][1]
            amount_base_units = int(entry["amount_base_units"])
            if entry["side"] == "debit":
                debit_total += amount_base_units
                if account_type in debit_counts:
                    debit_counts[account_type] += 1
            else:
                credit_total += amount_base_units
                if account_type in credit_counts:
                    credit_counts[account_type] += 1

        if debit_total != credit_total:
            raise SnapshotContractError(
                f"transaction_id={transaction_id} is unbalanced: debit_total={debit_total} credit_total={credit_total}"
            )
        if debit_total <= 0:
            raise SnapshotContractError(
                f"transaction_id={transaction_id} has non-positive amount after balancing"
            )

        prior_1h = 0
        prior_24h = 0
        for account_id in sorted(involved_accounts):
            history = account_history.setdefault(account_id, [])
            prior_1h += sum(1 for ts in history if ts >= created_at - timedelta(hours=1))
            prior_24h += sum(1 for ts in history if ts >= created_at - timedelta(hours=24))

        tx_type_flags = {
            f"tx_type_{known}": 1 if transaction_type == known else 0
            for known in KNOWN_TRANSACTION_TYPES
        }
        tx_type_flags["tx_type_other"] = 1 if transaction_type not in KNOWN_TRANSACTION_TYPES else 0

        row_features = {
            "total_amount_base_units": int(debit_total),
            "entry_count": int(len(tx_entries)),
            "debit_entry_count": int(sum(1 for e in tx_entries if e["side"] == "debit")),
            "credit_entry_count": int(sum(1 for e in tx_entries if e["side"] == "credit")),
            "debit_asset_entries": int(debit_counts["asset"]),
            "debit_liability_entries": int(debit_counts["liability"]),
            "debit_revenue_entries": int(debit_counts["revenue"]),
            "debit_expense_entries": int(debit_counts["expense"]),
            "credit_asset_entries": int(credit_counts["asset"]),
            "credit_liability_entries": int(credit_counts["liability"]),
            "credit_revenue_entries": int(credit_counts["revenue"]),
            "credit_expense_entries": int(credit_counts["expense"]),
            "unique_account_count": int(len(involved_accounts)),
            "prior_account_tx_count_1h": int(prior_1h),
            "prior_account_tx_count_24h": int(prior_24h),
            "hour_of_day": int(created_at.hour),
            "day_of_week": int(created_at.weekday()),
            "is_weekend": int(created_at.weekday() >= 5),
            "has_session_reference": _as_presence_flag(tx_row["session_reference"]),
            "has_external_reference": _as_presence_flag(tx_row["external_reference"]),
            **tx_type_flags,
        }

        for account_id in involved_accounts:
            account_history.setdefault(account_id, []).append(created_at)

        proxy_label = int(
            transaction_id in reversal_transaction_ids
            or settlement_state in {"failed", "reversed"}
        )

        records.append(
            {
                "transaction_id": transaction_id,
                "features": row_features,
                "proxy_label": proxy_label,
                "created_at": created_at.isoformat(),
            }
        )

    if not records:
        raise SnapshotContractError("No valid transactions found in snapshot")

    features = pd.DataFrame([record["features"] for record in records], columns=list(DOUBLE_ENTRY_FEATURE_NAMES))
    labels = np.array([int(record["proxy_label"]) for record in records], dtype=int)

    metadata = {
        "schema_version": DOUBLE_ENTRY_SNAPSHOT_SCHEMA_VERSION,
        "label_definition": "proxy_risk=1 if settlement_state indicates failure/reversal or reversal record exists",
        "label_caveat": "Proxy risk labels are operational outcomes, not confirmed fraud ground truth.",
        "feature_names": list(DOUBLE_ENTRY_FEATURE_NAMES),
        "transaction_ids_order": [record["transaction_id"] for record in records],
        "record_count": int(len(records)),
        "positive_label_count": int(labels.sum()),
        "currency_contract": "single currency per transaction; mixed currencies disallowed in this milestone",
        "double_entry_contract": ">=2 entries, positive integer amounts, debit_total == credit_total",
    }
    return SnapshotDataset(features=features, labels=labels, metadata=metadata)
