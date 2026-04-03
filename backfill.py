#!/usr/bin/env python3
"""Backfill historical inclusion time data from Xatu.

Queries sample points every 6 hours over the past 6 months, computing
fee-viable inclusion time metrics for each. Results are persisted to
data/history.json and can be resumed if interrupted.

Usage:
    python3 backfill.py              # backfill all missing points
    python3 backfill.py --days 30    # only last 30 days
"""

import argparse
import json
import os
import sys
import time as _time
from datetime import datetime, timedelta, timezone

import pandas as pd

from compute import (
    build_lookups,
    compute_viable_times,
    create_xatu,
    parse_blocks,
    parse_transactions,
    query_mempool_batched,
    query_mev_slots,
    summarize,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
HISTORY_FILE = os.path.join(DATA_DIR, "history.json")
SAMPLE_INTERVAL_HOURS = 6
BLOCKS_PER_SAMPLE = 20
TX_SAMPLE_SIZE = 200
QUERY_DELAY = 0.2  # seconds between samples


def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return []


def save_history(data):
    os.makedirs(DATA_DIR, exist_ok=True)
    data.sort(key=lambda x: x["timestamp"])
    tmp = HISTORY_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    os.replace(tmp, HISTORY_FILE)


def compute_sample(xatu, target_time):
    """Compute inclusion metrics for ~20 blocks around target_time."""
    ts = target_time.strftime("%Y-%m-%d %H:%M:%S")

    # Blocks in a 10-minute window around target
    raw_blocks = xatu.execute_query(
        f"""
        SELECT slot, slot_start_date_time, execution_payload_base_fee_per_gas,
               execution_payload_block_number, execution_payload_gas_used,
               execution_payload_gas_limit
        FROM default.canonical_beacon_block
        WHERE meta_network_name = 'mainnet'
          AND slot_start_date_time >= toDateTime64('{ts}', 3) - INTERVAL 5 MINUTE
          AND slot_start_date_time <= toDateTime64('{ts}', 3) + INTERVAL 5 MINUTE
        ORDER BY slot
        """,
        columns="slot,slot_start_date_time,base_fee_per_gas,block_number,gas_used,gas_limit",
    )
    if raw_blocks is None or raw_blocks.empty:
        return None

    blocks = parse_blocks(raw_blocks)
    (basefee_by_slot, slot_times, block_to_slot, sorted_slots,
     gas_used_by_slot, gas_limit_by_slot) = build_lookups(blocks)

    # Take middle N blocks
    mid = len(blocks) // 2
    start = max(0, mid - BLOCKS_PER_SAMPLE // 2)
    sample_blocks = blocks.iloc[start : start + BLOCKS_PER_SAMPLE]
    if len(sample_blocks) < 5:
        return None

    block_numbers = sample_blocks["block_number"].tolist()
    block_list = ",".join(str(b) for b in block_numbers)

    # Transactions
    raw_txs = xatu.execute_query(
        f"""
        SELECT block_number, transaction_hash, from_address, nonce,
               max_fee_per_gas, max_priority_fee_per_gas, gas_price,
               gas_limit, transaction_type
        FROM default.canonical_execution_transaction
        WHERE meta_network_name = 'mainnet'
          AND block_number IN ({block_list})
        """,
        columns=(
            "block_number,transaction_hash,from_address,nonce,"
            "max_fee_per_gas,max_priority_fee_per_gas,gas_price,"
            "gas_limit,transaction_type"
        ),
    )
    if raw_txs is None or raw_txs.empty:
        return None

    txs = parse_transactions(raw_txs)
    n_sample = min(TX_SAMPLE_SIZE, len(txs))
    if n_sample < 10:
        return None
    txs_sample = txs.sample(n=n_sample, random_state=42)

    # MEV-boost slots
    earliest = sample_blocks["slot_start_date_time"].min()
    latest = sample_blocks["slot_start_date_time"].max()
    mev_slots = query_mev_slots(xatu, earliest, latest)

    # Mempool first-seen times
    mempool_df = query_mempool_batched(
        xatu, txs_sample["transaction_hash"].tolist(), earliest, latest
    )
    if mempool_df is None:
        return None

    merged = txs_sample.merge(
        mempool_df, left_on="transaction_hash", right_on="tx_hash", how="left"
    )
    public_txs = merged[merged["first_seen_time"].notna()]
    if len(public_txs) < 5:
        return None

    viable_times = compute_viable_times(
        public_txs, basefee_by_slot, slot_times, block_to_slot, sorted_slots,
        gas_used_by_slot, gas_limit_by_slot, mev_slots,
    )
    stats = summarize(viable_times)
    if stats is None:
        return None

    stats["timestamp"] = target_time.strftime("%Y-%m-%dT%H:%M:%S")
    stats["basefee_gwei"] = round(
        float(sample_blocks["base_fee_per_gas"].median()) / 1e9, 2
    )
    stats["slot"] = int(sample_blocks["slot"].iloc[len(sample_blocks) // 2])
    return stats


def main():
    parser = argparse.ArgumentParser(description="Backfill historical inclusion data")
    parser.add_argument("--days", type=int, default=180, help="Lookback in days")
    args = parser.parse_args()

    xatu = create_xatu()
    history = load_history()
    existing = {d["timestamp"] for d in history}

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=args.days)

    # Generate sample timestamps every SAMPLE_INTERVAL_HOURS
    sample_times = []
    t = start.replace(minute=0, second=0, microsecond=0)
    while t < now - timedelta(hours=1):
        sample_times.append(t)
        t += timedelta(hours=SAMPLE_INTERVAL_HOURS)

    to_compute = [
        st for st in sample_times
        if st.strftime("%Y-%m-%dT%H:%M:%S") not in existing
    ]

    if not to_compute:
        print(f"All {len(sample_times)} points already computed.")
        return

    total = len(to_compute)
    est_min = total * 2.5 / 60
    print(f"Computing {total} / {len(sample_times)} sample points")
    print(f"Estimated time: ~{est_min:.0f} minutes")
    print()

    added = 0
    skipped = 0
    for i, target in enumerate(to_compute):
        label = target.strftime("%Y-%m-%d %H:%M")
        sys.stdout.write(f"\r  [{i+1}/{total}] {label} ...")
        sys.stdout.flush()

        try:
            result = compute_sample(xatu, target)
            if result:
                history.append(result)
                added += 1
            else:
                skipped += 1
        except Exception as e:
            sys.stdout.write(f" error: {e}\n")
            skipped += 1

        # Incremental save every 25 points
        if added > 0 and added % 25 == 0:
            save_history(history)
            sys.stdout.write(" [saved]")

        _time.sleep(QUERY_DELAY)

    if added > 0:
        save_history(history)

    print(f"\n\nDone. Added {added}, skipped {skipped}. Total: {len(history)} points.")


if __name__ == "__main__":
    main()
