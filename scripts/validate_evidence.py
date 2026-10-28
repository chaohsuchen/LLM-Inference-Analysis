"""Validate the published evidence independently of the original GPU run folders."""
import csv
import gzip
import hashlib
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.bench import summarize


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    root = ROOT / 'docs/results'
    hashes = json.loads((root / 'SHA256SUMS.json').read_text())
    for name, digest in hashes.items():
        require(hashlib.sha256((root / name).read_bytes()).hexdigest() == digest,
                f'Hash mismatch: {name}')
    groups = defaultdict(list)
    with gzip.open(root / 'requests.jsonl.gz', 'rt') as f:
        for line in f:
            row = json.loads(line)
            require(not {'prompt', 'sample', 'error'} & row.keys(), 'Private fields in evidence')
            require(row['ok'] and row['output_tokens'] == 128, 'Failed or unexpected output')
            require(0 <= row['dispatch_s'] <= row['first_content_s'] <= row['last_content_s'] <= row['completion_s'],
                    'Invalid event order')
            key = (row['repeat'], row['config'], row['workload'], row['concurrency'])
            groups[key].append(row)
    with (root / 'cells.csv').open() as f:
        rows = list(csv.DictReader(f))
    configs = ['serial','batch','cache','chunk512','chunk2048','combined']
    expected = set(itertools.product(range(3), configs, ['unique','shared','mixed'], [1,8,32]))
    require(set(groups) == expected and len(rows) == 162, 'Incomplete matrix')
    seen = set()
    for row in rows:
        key = (int(row['repeat']), row['config'], row['workload'], int(row['concurrency']))
        require(key not in seen, f'Duplicate summary: {key}')
        seen.add(key)
        records = groups[key]
        require(len(records) == 128 and len({r['id'] for r in records}) == 128, f'Invalid request count: {key}')
        require(row['measurement_source'] == 'live_vllm_http', 'Invalid measurement provenance')
        computed = summarize(records, float(row['wall_s']), float(row['ttft_slo_s']))
        for metric, value in computed.items():
            original = row[metric]
            require((value is None and original == '') or
                    (value is not None and math.isclose(value, float(original), rel_tol=1e-10, abs_tol=1e-10)),
                    f'Metric mismatch: {key} {metric}')
    require(seen == expected, 'Summary matrix differs from request matrix')
    # Inputs and measured output lengths must match across configurations.
    for repeat, workload, concurrency in itertools.product(range(3), ['unique','shared','mixed'], [1,8,32]):
        inputs = [{r['id']:(r['input_tokens'],r['server_prompt_tokens'],r['output_tokens'])
                   for r in groups[(repeat,cfg,workload,concurrency)]} for cfg in configs]
        require(all(x == inputs[0] for x in inputs), 'Unmatched paired input/output lengths')
    print(f'Validated {len(rows)} cells and {sum(map(len, groups.values())):,} requests; hashes and request-derived metrics match.')


if __name__ == '__main__':
    main()
