"""Shared computation logic for fee-viable inclusion times."""

import os
from bisect import bisect_left, bisect_right

import pandas as pd
import pyxatu


def create_xatu():
    """Create PyXatu client, using env variables if set, else config file."""
    if os.environ.get("CLICKHOUSE_URL"):
        return pyxatu.PyXatu(use_env_variables=True)
    return pyxatu.PyXatu()

BEACON_GENESIS_TIME = 1606824023
MEMPOOL_BATCH_SIZE = 200  # Xatu rejects queries >~30KB; 200 hashes ≈ 14KB
# Builder revenue filter: skip slots where the builder earns < $0.01 from
# including the tx.  revenue = effective_priority * gas_limit (in wei).
ETH_PRICE_USD = 1800
MIN_BUILDER_REVENUE_USD = 0.01
MIN_REVENUE_WEI = int(MIN_BUILDER_REVENUE_USD / ETH_PRICE_USD * 1e18)


def time_to_slot(dt):
    """Convert datetime to beacon chain slot number."""
    return int((dt.timestamp() - BEACON_GENESIS_TIME) // 12)


def parse_blocks(blocks):
    """Parse raw block query result into typed DataFrame + lookup dicts."""
    blocks["slot"] = blocks["slot"].astype(int)
    blocks["base_fee_per_gas"] = pd.to_numeric(
        blocks["base_fee_per_gas"], errors="coerce"
    )
    blocks["block_number"] = blocks["block_number"].astype(int)
    blocks["slot_start_date_time"] = pd.to_datetime(blocks["slot_start_date_time"])
    if "gas_used" in blocks.columns:
        blocks["gas_used"] = pd.to_numeric(blocks["gas_used"], errors="coerce").fillna(0)
    if "gas_limit" in blocks.columns:
        blocks["gas_limit"] = pd.to_numeric(
            blocks["gas_limit"], errors="coerce"
        ).fillna(30_000_000)
    return blocks


def build_lookups(blocks):
    """Build slot→basefee, slot→time, block→slot, slot→gas dicts from parsed blocks."""
    basefee_by_slot = dict(zip(blocks["slot"], blocks["base_fee_per_gas"]))
    slot_times = dict(zip(blocks["slot"], blocks["slot_start_date_time"]))
    block_to_slot = dict(zip(blocks["block_number"], blocks["slot"]))
    sorted_slots = sorted(basefee_by_slot.keys())
    gas_used_by_slot = (
        dict(zip(blocks["slot"], blocks["gas_used"]))
        if "gas_used" in blocks.columns
        else {}
    )
    gas_limit_by_slot = (
        dict(zip(blocks["slot"], blocks["gas_limit"]))
        if "gas_limit" in blocks.columns
        else {}
    )
    return (
        basefee_by_slot, slot_times, block_to_slot, sorted_slots,
        gas_used_by_slot, gas_limit_by_slot,
    )


def parse_transactions(txs):
    """Parse fee columns, compute fee_cap/priority_cap, nonce-dedup."""
    for col in ("max_fee_per_gas", "max_priority_fee_per_gas", "gas_price"):
        txs[col] = pd.to_numeric(txs[col], errors="coerce").fillna(0)
    txs["nonce"] = pd.to_numeric(txs["nonce"], errors="coerce").fillna(0).astype(int)
    txs["block_number"] = txs["block_number"].astype(int)
    if "gas_limit" in txs.columns:
        txs["gas_limit"] = pd.to_numeric(
            txs["gas_limit"], errors="coerce"
        ).fillna(21_000).astype(int)

    # Exclude type-3 blob transactions (different inclusion dynamics)
    if "transaction_type" in txs.columns:
        txs["transaction_type"] = pd.to_numeric(
            txs["transaction_type"], errors="coerce"
        ).fillna(2).astype(int)
        txs = txs[txs["transaction_type"] != 3].copy()

    is_legacy = txs["max_fee_per_gas"] <= 0
    txs["fee_cap"] = txs["max_fee_per_gas"].where(~is_legacy, txs["gas_price"])
    txs["priority_cap"] = txs["gas_price"].where(
        is_legacy, txs["max_priority_fee_per_gas"]
    )
    txs = txs[txs["fee_cap"] > 0]

    # Nonce dedup: keep only first (lowest nonce) tx per sender
    txs = txs.sort_values(["nonce", "block_number"])
    txs = txs.drop_duplicates(subset="from_address", keep="first")
    return txs


def query_mev_slots(xatu, earliest_time, latest_time):
    """Query which slots used MEV-boost (had relay payloads delivered).

    Returns a set of slot numbers, or None if no data is available
    (e.g., relay data hasn't caught up yet). None disables the filter.
    """
    r = xatu.execute_query(
        f"""
        SELECT DISTINCT slot
        FROM default.mev_relay_proposer_payload_delivered
        WHERE meta_network_name = 'mainnet'
          AND slot_start_date_time >= toDateTime64('{earliest_time}', 3)
          AND slot_start_date_time <= toDateTime64('{latest_time}', 3)
        """,
        columns="slot",
    )
    if r is None or r.empty:
        return None  # no data — disable filter rather than exclude all slots
    return set(int(s) for s in r["slot"])


def query_mempool_batched(xatu, tx_hashes, earliest_time, latest_time):
    """Query mempool first-seen times in batches of MEMPOOL_BATCH_SIZE."""
    results = []
    for i in range(0, len(tx_hashes), MEMPOOL_BATCH_SIZE):
        batch = tx_hashes[i : i + MEMPOOL_BATCH_SIZE]
        hash_list = ",".join(f"'{h}'" for h in batch)
        df = xatu.execute_query(
            f"""
            SELECT
                hash AS tx_hash,
                min(event_date_time) AS first_seen_time
            FROM default.mempool_transaction
            WHERE meta_network_name = 'mainnet'
              AND event_date_time >= toDateTime64('{earliest_time}', 3) - INTERVAL 30 MINUTE
              AND event_date_time <= toDateTime64('{latest_time}', 3)
              AND hash IN ({hash_list})
            GROUP BY hash
            """,
            columns="tx_hash,first_seen_time",
        )
        if df is not None and not df.empty:
            results.append(df)

    if not results:
        return None

    mempool_df = pd.concat(results)
    mempool_df["first_seen_time"] = pd.to_datetime(mempool_df["first_seen_time"])
    return mempool_df


def compute_viable_times(public_txs, basefee_by_slot, slot_times, block_to_slot,
                         sorted_slots, gas_used_by_slot=None,
                         gas_limit_by_slot=None, mev_slots=None):
    """Compute fee-viable inclusion time for each public transaction.

    A slot is viable if:
      1. Block was built via MEV-boost (builder had full mempool visibility)
      2. fee_cap >= basefee AND builder revenue >= $0.01
      3. The block had enough remaining gas for the transaction

    Returns list of viable times in seconds (only txs with viable_time > 0).
    """
    viable_times = []
    check_gas = bool(gas_used_by_slot and gas_limit_by_slot)
    check_mev = mev_slots is not None

    for _, tx in public_txs.iterrows():
        first_seen = tx["first_seen_time"]
        block_num = int(tx["block_number"])
        fee_cap = tx["fee_cap"]
        priority_cap = tx["priority_cap"]
        tx_gas = int(tx["gas_limit"]) if "gas_limit" in tx.index else 21_000

        inclusion_slot = block_to_slot.get(block_num)
        if inclusion_slot is None:
            continue

        inclusion_time = slot_times[inclusion_slot]
        raw_time = (inclusion_time - first_seen).total_seconds()
        if raw_time <= 0:
            continue

        first_seen_slot = time_to_slot(first_seen)
        lo = bisect_left(sorted_slots, first_seen_slot)
        hi = bisect_right(sorted_slots, inclusion_slot)
        slots_in_range = sorted_slots[lo:hi]

        # Total possible slots (including missed ones with no block)
        total_slots = max(1, inclusion_slot - first_seen_slot + 1)

        if not slots_in_range:
            viable_times.append(raw_time)
        else:
            viable_count = 0
            for s in slots_in_range:
                if check_mev and s not in mev_slots:
                    continue  # solo validator — limited mempool
                bf = basefee_by_slot[s]
                eff_prio = min(priority_cap, fee_cap - bf) if fee_cap >= bf else 0
                if fee_cap >= bf and eff_prio * tx_gas >= MIN_REVENUE_WEI:
                    if check_gas:
                        remaining = (
                            gas_limit_by_slot.get(s, 30_000_000)
                            - gas_used_by_slot.get(s, 0)
                        )
                        if remaining < tx_gas:
                            continue  # block too full for this tx
                    viable_count += 1
            viable_time = raw_time * (viable_count / total_slots)
            if viable_time > 0:
                viable_times.append(viable_time)

    return viable_times


def summarize(viable_times):
    """Compute median/mean/p90/p99 from a list of viable times."""
    if not viable_times:
        return None
    s = sorted(viable_times)
    n = len(s)
    return {
        "median": round(s[n // 2], 1),
        "mean": round(sum(viable_times) / n, 1),
        "p90": round(s[int(n * 0.9)], 1),
        "p99": round(s[min(int(n * 0.99), n - 1)], 1),
        "num_txs": n,
    }
