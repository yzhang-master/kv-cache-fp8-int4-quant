import json
from pathlib import Path

from kvquant.bench import run_benchmark


def test_benchmark_measures_accuracy_storage_and_attention(tmp_path: Path) -> None:
    output = tmp_path / "smoke.json"
    result = run_benchmark(output, sequence_length=48, heads=2, head_dim=16, repeats=1)
    saved = json.loads(output.read_text())
    assert len(saved["results"]) == 10
    assert 0 <= saved["reference_long_range_top1_rate"] <= 1
    assert all(row["attention_ms"] > 0 for row in saved["results"])
    quantized = [row for row in saved["results"] if row["format"] in {"fp8_e4m3", "int4"}]
    assert len(quantized) == 8
    assert all(row["storage_bytes"] < saved["results"][0]["storage_bytes"] for row in quantized)
    assert all(0 <= row["attention_top1_agreement"] <= 1 for row in quantized)
    int4_token = next(
        row
        for row in quantized
        if row["format"] == "int4" and row["granularity"] == "token" and row["sink_tokens"] == 4
    )
    fp8_token = next(
        row
        for row in quantized
        if row["format"] == "fp8_e4m3" and row["granularity"] == "token" and row["sink_tokens"] == 4
    )
    assert int4_token["storage_bytes"] < fp8_token["storage_bytes"]
    assert result["scope"].startswith("Synthetic")
