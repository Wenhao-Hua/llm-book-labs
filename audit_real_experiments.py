"""Audit saved evidence independently of model execution, without torch imports."""
import argparse
import hashlib
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, default=BASE / 'real-experiments/results')
    p.add_argument('--output', type=Path, default=BASE / 'checks/real-experiment-audit.json')
    args = p.parse_args()
    reports = {k: json.loads((args.results / (k + '.json')).read_text(encoding='utf8')) for k in ['retrieval', 'finetune', 'cache']}
    checks = []
    def check(name, passed, detail):
        checks.append({'name': name, 'passed': bool(passed), 'detail': detail})
    data = json.loads((BASE / 'real-experiments/data.json').read_text(encoding='utf8'))
    check('input provenance', all(r['environment']['data_sha256'] == sha(BASE / 'real-experiments/data.json') and r['environment']['script_sha256'] == sha(BASE / 'real_experiments.py') for r in reports.values()), 'Script and teaching data match saved execution manifests.')
    check('pinned revisions', all(len(r['model']['revision']) == 40 and r['model']['files'] for r in reports.values()), 'Each report fixes a model revision and file hashes.')
    r = reports['retrieval']['result']
    docs = {d['id']: d for d in data['documents']}
    records = r['records']
    check('retrieval population', r['embedding_shape'] == [len(docs), 384] and len(records) == len(data['queries']), 'No query omitted from the report.')
    eligible = all(all(docs[item['id']]['current'] and q['role'] in docs[item['id']]['roles'] for item in q['filtered_ranking']) for q in records)
    check('returned candidates', eligible, 'Checks output filters, not upstream data isolation.')
    recalculated = {k: sum(len(set(q['relevant']) & {x['id'] for x in q['filtered_ranking'][:k]}) / len(q['relevant']) for q in records) / len(records) for k in [1, 3]}
    check('recall recalculation', all(math.isclose(recalculated[k], r[f'mean_filtered_recall_at_{k}']) for k in [1, 3]), recalculated)
    check('pooling control', r['padding_max_abs_difference'] < 1e-5, r['padding_max_abs_difference'])
    f = reports['finetune']['result']
    train = data['classification']['train']; holdout = data['classification']['holdout']
    check('split and seeds', not ({x['text'] for x in train} & {x['text'] for x in holdout}) and [x['seed'] for x in f['runs']] == [7, 17, 29], 'Exact text disjointness only; not independent population sampling.')
    valid = True
    for run in f['runs']:
        final = run['final_holdout']
        preds = [max(range(len(row)), key=lambda i: row[i]) for row in final['logits']]
        valid &= preds == final['predictions'] and sum(p == x['label'] for p, x in zip(preds, holdout)) == final['correct'] and len(preds) == len(holdout)
    check('classification score recalculation', valid, [r['final_holdout']['correct'] for r in f['runs']])
    check('full schedule and encoder update', all(len(r['trajectory']) == 20 and r['encoder_weight_max_change'] > 0 for r in f['runs']) and f['config']['steps'] == 60, 'Three fixed schedules; actual encoder weight changes recorded.')
    check('save reload evidence', all(r['reload_max_abs_difference'] < 1e-6 for r in f['runs']), 'Numerical reload assertion ran during each actual training execution.')
    checkpoint_status = []
    for run in f['runs']:
        file = args.results / f"finetune-seed-{run['seed']}" / 'model.safetensors'
        checkpoint_status.append({'seed': run['seed'], 'present': file.exists(), 'hash_matches': sha(file) == run['checkpoint_sha256'] if file.exists() else None})
    check('available checkpoint integrity', all(r['hash_matches'] is not False for r in checkpoint_status), checkpoint_status)
    c = reports['cache']['result']
    expected = 2 * 12 * c['cache_sequence_length'] * 12 * 64 * 4
    check('KV storage arithmetic', c['cache_tensor_bytes'] == expected == c['formula_tensor_bytes'], expected)
    check('generation equivalence', all(t['cached']['token_ids'] == t['full_prefix']['token_ids'] and len(t['cached']['token_ids']) == 16 for t in c['trials']), 'Three cached/full-prefix token comparisons.')
    check('timing arithmetic', all(len(m['step_seconds']) == 16 and all(x > 0 for x in m['step_seconds']) and math.isclose(sum(m['step_seconds']), m['total_seconds']) for t in c['trials'] for m in t.values()), 'Raw step timings retained; no cross-machine speed guarantee.')
    answer_calculations = {'chapter03_loss': math.log(2), 'chapter03_grad_w': (-.5-1)/2, 'chapter03_grad_b': 0, 'chapter03_new_w': .15, 'chapter04_unmasked': [8/3, 8/3], 'chapter04_masked': [1, 1], 'chapter06_available': 4096-300-100-600-512, 'chapter09_coverage': 12/20, 'chapter09_risk': 3/12, 'chapter10_lora_parameters': 4*(1024+2048), 'chapter10_ratio': 4*(1024+2048)/(1024*2048), 'chapter11_loss': math.log(2), 'chapter11_margin_gradient': -.2/2, 'chapter12_cache_mib': 512*3, 'chapter14_final_box': [(40-20)*2+10,(60-30)*2+20,(140-20)*2+10,(160-30)*2+20]}
    report = {'passed': all(c['passed'] for c in checks), 'checks': checks, 'answer_calculations': answer_calculations, 'scope': 'Saved-evidence integrity and arithmetic checks; not an independent expert review or fresh-machine replication.'}
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({'passed': report['passed'], 'checks': len(checks), 'failed': [c['name'] for c in checks if not c['passed']]}, ensure_ascii=False))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
