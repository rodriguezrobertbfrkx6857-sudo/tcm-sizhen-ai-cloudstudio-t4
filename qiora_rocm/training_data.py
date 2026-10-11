"""Private revision_03 inputs; no truncation, label leakage or network fallback."""
import hashlib
import json
import math
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def safe_path(path):
    p = Path(path).resolve()
    if 'test2' in str(p).casefold():
        raise ValueError('SEALED_TEST2_ACCESS_FORBIDDEN')
    return p


def rows(path, expected_sha, role, selection=None):
    p = safe_path(path)
    raw = p.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError('DATA_SHA_MISMATCH')
    rr = [json.loads(line) for line in raw.decode('utf8').splitlines() if line.strip()]
    if selection is not None:
        if not isinstance(selection,list) or not selection or selection!=sorted(set(selection)) or any(type(i) is not int or not 0<=i<len(rr) for i in selection):
            raise ValueError('INVALID_FROZEN_ROW_SELECTION')
        rr = [rr[i] for i in selection]
    if not rr or len({r['sample_id'] for r in rr}) != len(rr):
        raise ValueError('EMPTY_OR_DUPLICATE_IDS')
    seen = set()
    for r in rr:
        if role == 'TRAIN':
            if r.get('train_role') != role or r.get('split') != role:
                raise ValueError('SPLIT_MISMATCH')
        elif r.get('role') not in ('DEV', 'DEV_PUBLIC_DIAGNOSTIC') or any(r.get(k) == 'TRAIN' for k in ('train_role','split')):
            raise ValueError('SPLIT_MISMATCH')
        if role == 'TRAIN':
            if r.get('training_use') is False or r.get('is_revealed_diagnostic'):
                raise ValueError('EVALUATION_IN_TRAIN')
            if not all(r.get(k) for k in ('question', 'answer', 'fact_cluster_id', 'quality_tier', 'source_type', 'work_id')):
                raise ValueError('MISSING_TRAIN_PROVENANCE')
            if r.get('medical_Gold') and not r.get('doctor_verified'):
                raise ValueError('UNVERIFIED_GOLD')
            weight = r.get('loss_weight', 1.)
            if not isinstance(weight, (float, int)) or not math.isfinite(weight) or not 0 < weight <= 1:
                raise ValueError('INVALID_LOSS_WEIGHT')
            key = digest([r['question'], r['answer'], r.get('sources', [])])
            if key in seen:
                raise ValueError('EXACT_DUPLICATE_SUPERVISION')
            seen.add(key)
        for source in r.get('sources', []):
            validate_source(source)
    return rr


def validate_source(s):
    required = {'doc_id', 'work_id', 'revision', 'span_start', 'span_end', 'text_sha256', 'evidence_text', 'source_type'}
    if not required <= s.keys() or not all(s[k] for k in ('doc_id', 'work_id', 'revision')):
        raise ValueError('INCOMPLETE_EVIDENCE_SOURCE')
    if hashlib.sha256(s['evidence_text'].encode('utf8')).hexdigest() != s['text_sha256']:
        raise ValueError('EVIDENCE_SHA_MISMATCH')
    if type(s['span_start']) is not int or type(s['span_end']) is not int or s['span_start'] < 0 or s['span_end'] - s['span_start'] != len(s['evidence_text']):
        raise ValueError('INVALID_EVIDENCE_SPAN')


def clusters(rr):
    return {x for r in rr for x in [r['fact_cluster_id'], *r.get('additional_fact_cluster_ids', [])]}


def isolate(train, dev):
    if clusters(train) & clusters(dev):
        raise ValueError('TRAIN_DEV_FACT_LEAKAGE')
    # Work and historical semantic contamination are reported, never hidden.
    return dict(fact_cluster_overlap=0, work_overlap=len({r['work_id'] for r in train} & {r.get('work_id') for r in dev}),
                historical_parent_semantic_contamination='UNKNOWN', independent_acceptance=False)


def public_prompt(row):
    q = row['question']
    if row.get('options'):
        q += '\n' + '\n'.join(k + '. ' + str(v) for k, v in row['options'].items()) + '\n仅输出正确选项字母，多选可输出多个字母。'
    if row.get('context'):
        q += '\n给定文字：' + row['context']
    if row.get('sources'):
        q += '\n给定来源（仅这些片段可核查）：\n' + json.dumps(row['sources'], ensure_ascii=False)
    return q


def prompt_ids(tok, row):
    text = tok.apply_chat_template([dict(role='user', content=public_prompt(row))], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    return tok.encode(text, add_special_tokens=False)


def encode(tok, row, seq_len):
    text = public_prompt(row) + row['answer']
    if not row['answer'].strip() or any(s in text for s in ('<|im_start|>', '<|im_end|>', '<|endoftext|>')):
        raise ValueError('EMPTY_OR_CONTROL_TOKEN')
    p = prompt_ids(tok, row)
    a = tok.encode(row['answer'], add_special_tokens=False)
    if not a or tok.eos_token_id in a:
        raise ValueError('EMPTY_OR_EARLY_EOS_TARGET')
    a += [tok.eos_token_id]
    if len(p) + len(a) > seq_len:
        raise ValueError('WOULD_TRUNCATE_SUPERVISION')
    return dict(input_ids=p + a, labels=[-100] * len(p) + a, prompt_tokens=len(p),
                input_tokens=len(p) + len(a), supervised_tokens=len(a))
