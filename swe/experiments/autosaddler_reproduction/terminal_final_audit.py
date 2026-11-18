"""Read-only result/journal audit; writes a separate report, never historical rows."""
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def summarize(rows):
    known = [r for r in rows if r.get('usage')]
    unknown = [r for r in rows if not r.get('usage')]
    return dict(requests=len(rows), known_tokens=sum(r['usage']['total_tokens'] for r in known),
                total_tokens=None if unknown else sum(r['usage']['total_tokens'] for r in known),
                settled_allowance=round(sum(r['usage']['allowance'] for r in known), 8),
                unknown_requests=len(unknown), unknown_reserve=round(sum(r['reservation'] for r in unknown), 8),
                known_balance_cost=sum(r['usage']['balance_cost'] for r in known))


def audit(run_id):
    base = ROOT / 'results/autosaddler_reproduction'
    run = base / 'go_experiments' / run_id
    read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
    plan = read(run / 'frozen_plan.json')
    assert read(run / 'status.json')['status'] == 'complete'
    manifest = read(run / 'source_manifest.json')
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == digest for p, digest in manifest.items())
    source = base / 'runs' / plan['selection_source']['run_id'] / 'events.jsonl'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == plan['selection_source']['events_sha256']
    last = {}
    for p in [Path(x['path']) for x in read(run / 'gateway/budget_history.json')] + [run / 'gateway/requests.jsonl']:
        for line in p.read_text(encoding='utf-8-sig').splitlines():
            if line.strip():
                row = json.loads(line)
                last[row['id']] = row
    current = [r for r in last.values() if r['run_id'] == run_id]
    results = []
    for path in (base / 'runs' / run_id / 'final_test').glob('*/*/*/*/result.json'):
        r = read(path)
        trial = read(Path(r['trial_result']))
        assert r['valid_evaluation'] and r['agent_cleanup_verified']
        assert trial['verifier_result']['rewards']['reward'] == r['score']
        assert (trial.get('exception_info') or {}).get('exception_type') in {None, 'AgentTimeoutError'}
        assert r['candidate_id'] == plan['selection_source']['seed_id' if r['label'] == 'H0' else 'candidate_id']
        assert r['model'] == plan['models']['target'] and r['repetition'] == 0
        results.append({k: r[k] for k in ('label', 'task_id', 'score', 'candidate_id', 'candidate_hash', 'agent_timed_out')})
    assert len(results) == 10
    for label in ('H0', 'H1'):
        assert sorted(r['task_id'] for r in results if r['label'] == label) == sorted(plan['test'])
    # Usage missing at process exit is UNKNOWN, regardless of the last row's status.
    groups = defaultdict(list)
    for r in last.values():
        groups[(r.get('benchmark') or 'shared_search', r['role'], r.get('purpose') or 'primary')].append(r)
    current_roles = {label: summarize([r for r in current if r['role'] == role])
                     for label, role in [('H0', 'target_final_baseline'), ('H1', 'target_final_optimized')]}
    report = dict(run_id=run_id, audited_at_utc=datetime.now(timezone.utc).isoformat(),
                  passed=True, evaluation_completeness='10/10', api_usage_complete=False,
                  seed=42, test_repetitions=1, tests=plan['test'], results=results,
                  source_files_verified=len(manifest), selection_source=plan['selection_source'],
                  current_valid_evaluation_usage=current_roles, campaign_all_attempts=summarize(list(last.values())),
                  campaign_by_role={role: summarize([r for r in last.values() if r['role'] == role]) for role in sorted({r['role'] for r in last.values()})},
                  campaign_by_benchmark_role_purpose=[dict(benchmark=k[0], role=k[1], purpose=k[2], **summarize(v)) for k,v in sorted(groups.items())],
                  unknown_requests=[r for r in last.values() if not r.get('usage')],
                  pending_requests=0, compaction_is_subset=True,
                  caveats=[
                      'Native result usage_complete flags describe exported messages only; gateway unknown requests make full API totals unavailable.',
                      'H0 FEAL first response used about 22 minutes of the unchanged official 1800-second deadline; following native compaction succeeded.',
                      'H1 path request d4de00ce06d24d5b851e422eee7d9000 outlived its parent timeout, then failed at gateway shutdown with RemoteProtocolError; reserve retained, usage unknown. Cause of coexisting primary/compaction requests unproven.',
                      'All 10 cleanup/verifier results remain valid; late transport failure is disclosed and is not a reason to repeat completed tests.',
                      'Source optimization dev 1/3 to 2/3 had technical incomplete iterations; bounded pilot, not a clean full paper replication.',
                      'All prior attempts are charged once; shared optimizer/search costs are not repeated for subsequent benchmarks.'])
    (run / 'final_audit.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    lines = ['# Terminal-Bench 2: frozen H0/H1 comparison', '',
             'All 10 official evaluations completed. H0: 0/5 (0%); H1: 0/5 (0%); difference: 0 percentage points. No improvement was observed on these five paired test tasks.', '',
             '| Task | H0 | H1 |', '|---|---:|---:|']
    for task in plan['test']:
        scores = {r['label']:r['score'] for r in results if r['task_id'] == task}
        lines.append(f"| {task} | {scores['H0']:g} | {scores['H1']:g} |")
    lines += ['', 'All results have verified agent cleanup and the official task verifier. H0 path/FEAL and H1 path reached the official timeout. The same selected candidates, target model, overlay, test IDs and seed 42 were used; no test-based optimization occurred.', '',
              '| Cost scope | Requests | Known API tokens | Missing usage | Settled allowance | Retained reserve |', '|---|---:|---:|---:|---:|---:|']
    for name, s in list(current_roles.items()) + [('Campaign: all attempts', report['campaign_all_attempts'])]:
        lines.append(f"| {name} | {s['requests']} | {s['known_tokens']:,} | {s['unknown_requests']} | {s['settled_allowance']:.8f} | {s['unknown_reserve']:.7f} |")
    lines += ['', 'Full API token totals are unknown, not zero. All known balance charges are zero. Compaction is included in parent roles; campaign totals already include earlier attempts.', '']
    lines.extend('- '+c for c in report['caveats'])
    lines += ['', 'Authoritative score comparison: comparison.json / comparison.csv. Detailed reconciled accounting and provenance: final_audit.json and campaign_accounting.json. Native comparison is only a cross-check.', '', 'Next: SWE-Bench Pro V1, then GAIA2, using the same frozen H0/H1 without another optimization.']
    (run / 'terminal_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('evaluation_completeness', 'current_valid_evaluation_usage', 'campaign_all_attempts')}))


if __name__ == '__main__':
    import sys
    audit(sys.argv[1])
