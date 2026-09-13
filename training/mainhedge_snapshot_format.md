# Mainhedge Ledger Snapshot (Sanitized, Local-Only)

This format models a **double-entry ledger snapshot** for local Ledger ML training.

## Hard data boundary

- Do **not** commit or upload real Mainhedge records.
- Use synthetic or sanitized values only.
- Never include credentials, customer identifiers, PII, raw metadata blobs, or secrets.

## Schema version

- `mainhedge-ledger-snapshot-v1`

## Required files

Place files under one directory (passed as `LEDGERML_DATASET_PATH`):

1. `ledgers.csv`
2. `accounts.csv`
3. `transactions.csv`
4. `entries.csv`
5. `reversals.csv` (optional)

## Table schemas

### `ledgers.csv`

| column | type | notes |
|---|---|---|
| `ledger_id` | string | fake/sanitized identifier |
| `currency` | string | ISO-like currency code, e.g. `USD` |

### `accounts.csv`

| column | type | notes |
|---|---|---|
| `account_id` | string | fake/sanitized identifier |
| `ledger_id` | string | must exist in `ledgers.csv` |
| `account_type` | string | expected: `asset`, `liability`, `revenue`, `expense` |
| `currency` | string | must match account ledger currency |

### `transactions.csv`

| column | type | notes |
|---|---|---|
| `transaction_id` | string | fake/sanitized identifier |
| `ledger_id` | string | must exist in `ledgers.csv` |
| `transaction_type` | string | e.g. `card_payment`, `bank_transfer`, `fee` |
| `created_at` | RFC3339 timestamp | used for temporal features |
| `currency` | string | must match ledger and entries currency |
| `settlement_state` | string | used only for proxy labels, never as model feature |
| `session_reference` | string/empty | converted to safe presence flag |
| `external_reference` | string/empty | converted to safe presence flag |

### `entries.csv`

| column | type | notes |
|---|---|---|
| `entry_id` | string | fake/sanitized identifier |
| `transaction_id` | string | must exist in `transactions.csv` |
| `account_id` | string | must exist in `accounts.csv` |
| `entry_type` | string | `debit` or `credit` |
| `amount_base_units` | integer string | positive integer only (no decimals) |
| `currency` | string | must match account and transaction currency |

### `reversals.csv` (optional)

| column | type | notes |
|---|---|---|
| `reversal_id` | string | fake/sanitized identifier |
| `transaction_id` | string | transaction marked as reversed |
| `reversed_at` | RFC3339 timestamp | parsed for validation |
| `reason_code` | string | operational metadata |

## Enforced invariants

- At least two entries per transaction.
- Positive integer `amount_base_units`.
- Debit total equals credit total for every transaction.
- Single currency per transaction (mixed currencies rejected in this milestone).

## Label semantics

- `proxy_label = 1` when settlement state is `failed`/`reversed` **or** reversal row exists.
- These are **operational risk proxies**, not confirmed fraud truth labels.

## Leakage guardrails

Outcome fields such as settlement state and reversal flags are used only to derive labels.
They are not included directly as model features.
