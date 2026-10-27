"""Refresh the frontend snapshot from existing Phase 1 reports (no inference)."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def rows(name):
    with (ROOT / 'reports' / name).open() as stream:
        return [{key: float(value) if key != 'model' and key != 'target' else value
                 for key, value in row.items()} for row in csv.DictReader(stream)]

def main():
    engines = rows('fd001_final_ml_predictions.csv')
    histories = {}
    with (ROOT / 'data/test_FD001.txt').open() as stream:
        for line in stream:
            values = [float(v) for v in line.split()]
            histories.setdefault(str(int(values[0])), []).append([int(values[1]), *[values[4+i] for i in (2, 4, 7, 11, 12, 14)]])
    assert len(engines) == 100
    assert all(histories[str(int(e['engine_id']))][-1][0] == e['last_observed_cycle'] for e in engines)
    payload = dict(engines=engines, histories=histories, metrics=rows('fd001_final_ml_metrics.csv')[0], models=rows('grouped_cv_model_comparison.csv'))
    (ROOT / 'app/data.json').write_text(json.dumps(payload, separators=(',', ':')))
    print(f'Exported {len(engines)} engines with verified cycle histories.')

if __name__ == '__main__':
    main()
