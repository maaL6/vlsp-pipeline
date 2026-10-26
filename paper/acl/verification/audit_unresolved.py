"""Summarise final document diagnostics; verify output identity without inference."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--delivery-root', type=Path, required=True)
    ap.add_argument('--pred', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    report = {'policy': 'Count units once per failure category; categories overlap. '
              'Diagnostic flags do not measure semantic omissions.', 'directions': {}}
    for direction, target in [('en-vi', 'vi'), ('vi-en', 'en')]:
        path = args.delivery_root / f'work_public_{direction}/logs/document_status.jsonl'
        docs = [json.loads(line) for line in path.open(encoding='utf-8')]
        assert len(docs) == len({d['sample_id'] for d in docs}) == 50
        categories, flags = Counter(), Counter()
        for doc in docs:
            pred = args.pred / target / (doc['sample_id'] + '.md')
            assert hashlib.sha256(pred.read_bytes()).hexdigest() == doc['output_sha256'], pred
            for unit in doc['unresolved_units']:
                categories.update({h.split(':')[0] for h in unit['hard']})
            flags.update(doc['flags'])
        report['directions'][direction] = {
            'diagnostics_path': str(path),
            'diagnostics_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'documents': len(docs), 'all_output_hashes_match': True,
            'unresolved_documents': sum(bool(d['unresolved_units']) for d in docs),
            'unresolved_units': sum(len(d['unresolved_units']) for d in docs),
            'unresolved_unit_categories': dict(categories),
            'unresolved_empty_nodes': sum(len(d['unresolved_empty']) for d in docs),
            'source_fallbacks': flags.get('untranslated_source_fallback', 0),
            'document_hard_failure_categories': dict(Counter(
                h.split(':')[0] for d in docs for h in d['doc_hard_failures'])),
            'affected_document_ids': [d['sample_id'] for d in docs if d['unresolved_units']],
        }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
