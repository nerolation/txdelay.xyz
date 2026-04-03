import json
import os
import time
from datetime import datetime, timezone

import pyxatu
from flask import Flask, Response, jsonify, render_template, request

from compute import (
    build_lookups,
    compute_viable_times,
    parse_blocks,
    parse_transactions,
    query_mempool_batched,
    query_mev_slots,
    summarize,
)

app = Flask(__name__)

CACHE_TTL = 60  # seconds
HISTORY_CACHE_TTL = 300  # 5 min — history.json rarely changes
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
HISTORY_FILE = os.path.join(DATA_DIR, "history.json")

_xatu = None
_cache = {"data": None, "timestamp": 0}
_history_cache = {"data": None, "timestamp": 0, "mtime": 0}


def get_xatu():
    global _xatu
    if _xatu is None:
        _xatu = pyxatu.PyXatu(use_env_variables=True)
    return _xatu


def compute_inclusion_data():
    """Query Xatu for recent blocks and compute fee-viable inclusion times."""
    xatu = get_xatu()

    raw_blocks = xatu.execute_query(
        """
        SELECT slot, slot_start_date_time,
               execution_payload_base_fee_per_gas,
               execution_payload_block_number,
               execution_payload_gas_used,
               execution_payload_gas_limit
        FROM default.canonical_beacon_block
        WHERE meta_network_name = 'mainnet'
          AND slot_start_date_time >= now() - INTERVAL 30 MINUTE
        ORDER BY slot
        """,
        columns="slot,slot_start_date_time,base_fee_per_gas,block_number,gas_used,gas_limit",
    )
    if raw_blocks is None or raw_blocks.empty:
        return None

    blocks = parse_blocks(raw_blocks)
    (basefee_by_slot, slot_times, block_to_slot, sorted_slots,
     gas_used_by_slot, gas_limit_by_slot) = build_lookups(blocks)

    analysis_blocks = blocks.tail(50)
    block_numbers = analysis_blocks["block_number"].tolist()
    block_list = ",".join(str(b) for b in block_numbers)

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
    num_filtered_txs = len(txs)
    sample_size = min(1000, num_filtered_txs)
    txs_sample = txs.sample(n=sample_size, random_state=42)

    earliest = analysis_blocks["slot_start_date_time"].min()
    latest = analysis_blocks["slot_start_date_time"].max()
    mev_slots = query_mev_slots(xatu, earliest, latest)

    mempool_df = query_mempool_batched(
        xatu, txs_sample["transaction_hash"].tolist(), earliest, latest
    )
    if mempool_df is None:
        return None

    merged = txs_sample.merge(
        mempool_df, left_on="transaction_hash", right_on="tx_hash", how="left"
    )
    public_txs = merged[merged["first_seen_time"].notna()]
    if public_txs.empty:
        return None

    viable_times = compute_viable_times(
        public_txs, basefee_by_slot, slot_times, block_to_slot, sorted_slots,
        gas_used_by_slot, gas_limit_by_slot, mev_slots,
    )
    stats = summarize(viable_times)
    if stats is None:
        return None

    return {
        "median_viable_time": stats["median"],
        "mean_viable_time": stats["mean"],
        "p90_viable_time": stats["p90"],
        "p99_viable_time": stats["p99"],
        "num_blocks": len(block_numbers),
        "num_txs": num_filtered_txs,
        "latest_basefee_gwei": round(blocks["base_fee_per_gas"].iloc[-1] / 1e9, 2),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }


def get_cached_data():
    now = time.time()
    if now - _cache["timestamp"] < CACHE_TTL and _cache["data"] is not None:
        return _cache["data"]
    try:
        data = compute_inclusion_data()
        if data is not None:
            _cache["data"] = data
            _cache["timestamp"] = now
        return _cache["data"]
    except Exception as e:
        print(f"Error computing inclusion data: {e}")
        return _cache["data"]


def get_history():
    """Return cached history, reloading only when the file changes."""
    now = time.time()
    if not os.path.exists(HISTORY_FILE):
        return []
    mtime = os.path.getmtime(HISTORY_FILE)
    if (now - _history_cache["timestamp"] < HISTORY_CACHE_TTL
            and _history_cache["mtime"] == mtime
            and _history_cache["data"] is not None):
        return _history_cache["data"]
    with open(HISTORY_FILE) as f:
        data = json.load(f)
    _history_cache.update(data=data, timestamp=now, mtime=mtime)
    return data


# ── Routes ──


@app.route("/")
def index():
    data = get_cached_data()
    return render_template("index.html", data=data)


@app.route("/chart")
def chart():
    return render_template("chart.html")


@app.route("/api/data")
def api_data():
    data = get_cached_data()
    if data is None:
        return jsonify({"error": "No data available"}), 500
    return jsonify(data)


@app.route("/api/history")
def api_history():
    data = get_history()
    since = request.args.get("since")
    if since:
        data = [d for d in data if d["timestamp"] >= since]
    body = json.dumps(data, separators=(",", ":"))
    return Response(body, content_type="application/json",
                    headers={"Cache-Control": "public, max-age=300"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
