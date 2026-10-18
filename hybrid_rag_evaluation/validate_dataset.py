#!/usr/bin/env python3
"""Validate evaluation labels and corpus integrity without third-party dependencies."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent
    parser.add_argument('--dataset', type=Path, default=base / 'evaluation.jsonl')
    parser.add_argument('--corpus-root', type=Path, default=base / 'reference_corpus')
    parser.add_argument('--manifest', type=Path, default=base / 'corpus_manifest.json')
    parser.add_argument('--indexed-sources', type=Path, help='Optional JSON array of corpus-relative source paths exported from the live index; one per chunk, duplicates allowed')
    args = parser.parse_args()
    errors = []
    def check(ok, message):
        if not ok:
            errors.append(message)
    def safe(path):
        p = PurePosixPath(path)
        if p.is_absolute() or '..' in p.parts or '\\' in path:
            raise ValueError('Unsafe source path: ' + path)
        resolved = (args.corpus_root / path).resolve()
        if not resolved.is_relative_to(args.corpus_root.resolve()):
            raise ValueError('Source escapes corpus root: ' + path)
        return resolved
    try:
        rows = [json.loads(l) for l in args.dataset.read_text().splitlines() if l.strip()]
        provenance = json.loads(args.manifest.read_text())
        original = json.loads((args.corpus_root / 'manifest.json').read_text())
        faqs = [json.loads(l) for l in (args.corpus_root / 'faqs/faqs.jsonl').read_text().splitlines() if l.strip()]
        logical = {r['document_id']: r for r in original}
        for i, f in enumerate(faqs, 1):
            logical[f['id']] = dict(document_id=f['id'], path='faqs/faqs.jsonl', product=f['product'], type='faq', line=i, answer=f['answer'])
        expected_files = {f['path'] for f in provenance['files']}
        actual_files = {f.relative_to(args.corpus_root).as_posix() for f in args.corpus_root.rglob('*') if f.is_file()}
        check(actual_files == expected_files, f'Corpus file inventory differs: missing={sorted(expected_files-actual_files)}, extra={sorted(actual_files-expected_files)}')
        for f in provenance['files']:
            p = safe(f['path'])
            check(p.is_file(), 'Missing corpus file: ' + f['path'])
            if p.is_file():
                check(hashlib.sha256(p.read_bytes()).hexdigest() == f['sha256'], 'Corpus hash mismatch: ' + f['path'])
        check(len(logical) == 105, 'Expected 105 logical documents')
        check(len(faqs) == 60, 'Expected 60 FAQ records')
        check(Counter(r['type'] for r in original) == {'manual': 5, 'troubleshooting': 40}, 'Manual/troubleshooting coverage differs')
        categories = Counter(r['category'] for r in rows)
        check(categories == {'direct_answerable':10,'paraphrased':10,'exact_error_code':5,'unanswerable':5}, 'Category coverage must be 10/10/5/5')
        check(len(rows)==30, 'Expected 30 questions')
        check(len({r['question_id'] for r in rows})==len(rows), 'Duplicate question IDs')
        check(len({r['question'].strip().casefold() for r in rows})==len(rows), 'Duplicate question text')
        check(all(re.fullmatch(r'EVAL_\d{3}',r['question_id']) for r in rows), 'Invalid question ID format')
        check(Counter(r['expected_answerability'] for r in rows)=={'answerable':25,'unanswerable':5}, 'Answerability coverage differs')
        groups = defaultdict(list)
        source_types = set()
        gold_ids = set()
        codes = set()
        for r in rows:
            qid = r['question_id']
            check(r['label_status']=='verified_against_archive', qid+': unverified label')
            check(isinstance(r['expected_facts'],list) and len(r['expected_facts'])>0 and all(isinstance(f,str) and f.strip() for f in r['expected_facts']), qid+': missing expected facts')
            check(r['escalation_expectation'] in ('required','conditional','not_required'),qid+': invalid escalation expectation')
            check(type(r['should_escalate']) is bool and r['should_escalate']==(r['escalation_expectation']=='required'),qid+': inconsistent immediate escalation flag')
            unanswerable = r['category']=='unanswerable'
            check((r['expected_answerability']=='unanswerable')==unanswerable,qid+': inconsistent answerability')
            check(not r['gold_sources'] if unanswerable else bool(r['gold_sources']),qid+': invalid gold source presence')
            if unanswerable:
                check(r['escalation_expectation']=='required' and bool(r['label_notes']),qid+': missing abstention rationale')
            for role in ('gold_sources','context_sources'):
                for s in r[role]:
                    p = safe(s['path'])
                    check(p.is_file(),qid+': missing source '+s['path'])
                    check(s['source_id'] in logical,qid+': unknown source ID '+s['source_id'])
                    if s['source_id'] not in logical or not p.is_file():
                        continue
                    expected = logical[s['source_id']]
                    check(s['path']==expected['path'] and s['product']==expected['product']==r['product'] and s['type']==expected['type'],qid+': source identity mismatch')
                    text = expected['answer'] if expected['type']=='faq' else p.read_text()
                    if expected['type']=='faq':
                        check(s.get('line')==expected['line'],qid+': FAQ line mismatch')
                    else:
                        check(s['source_id'] in text,qid+': ID absent from document')
                    check(bool(s['evidence']) and s['evidence'] in text,qid+': evidence excerpt not found')
                    if role=='gold_sources':
                        gold_ids.add(s['source_id'])
                        source_types.add(s['type'])
            if r['category'] in ('direct_answerable','paraphrased'):
                groups[r['intent_group']].append(r)
            if r['category']=='exact_error_code':
                found = re.findall(r'\bE\d{3}\b',r['question'])
                check(len(found)==1,qid+': expected one exact code')
                codes.update(found)
                check(any(found and found[0] in s['evidence'] for s in r['gold_sources']),qid+': code not supported by evidence')
        check(codes=={'E204','E310','E118','E407','E512'},'Expected all five documented error codes')
        check(len(groups)==10,'Expected ten paired intent groups')
        for group, pair in groups.items():
            check(Counter(r['category'] for r in pair)=={'direct_answerable':1,'paraphrased':1},group+': invalid pair')
            if len(pair)==2:
                check(pair[0]['expected_facts']==pair[1]['expected_facts'] and pair[0]['gold_sources']==pair[1]['gold_sources'],group+': pair labels differ')
        products = {r['product'] for r in original}
        check(set(r['product'] for r in rows)==products,'Product coverage differs')
        for product in products:
            check(Counter(r['category'] for r in rows if r['product']==product)=={'direct_answerable':2,'paraphrased':2,'exact_error_code':1,'unanswerable':1},product+': category coverage differs')
        check(source_types=={'manual','troubleshooting','faq'},'Gold sources must cover all document types')
        live = {'status':'not_checked'}
        if args.indexed_sources:
            indexed = json.loads(args.indexed_sources.read_text())
            if not isinstance(indexed,list) or not all(isinstance(p,str) for p in indexed):
                raise ValueError('--indexed-sources must contain a JSON array of relative path strings')
            allowed = {r['path'] for r in logical.values()}
            unknown = set(indexed)-allowed
            missing = allowed-set(indexed)
            check(not unknown, 'Index has non-allowlisted sources (possible leakage): '+str(sorted(unknown)))
            check(not missing, 'Index is missing corpus source files: '+str(sorted(missing)))
            live = {'status':'path_inventory_checked','chunks_reported':len(indexed),'unique_files':len(set(indexed)), 'note':'Does not prove chunk text or FAQ-record completeness; inspect content separately.'}
        report = {'status':'PASS' if not errors else 'FAIL','questions':len(rows),'categories':dict(categories),'products':len(products),'unique_gold_source_ids':len(gold_ids),'logical_corpus_documents':len(logical),'indexable_files':len({r['path'] for r in logical.values()}),'live_index':live,'errors':errors}
    except (OSError, ValueError, KeyError, TypeError) as e:
        report = {'status':'FAIL','errors':errors+[str(e)]}
    print(json.dumps(report,indent=2))
    return 0 if report['status']=='PASS' else 1

if __name__=='__main__':
    sys.exit(main())
