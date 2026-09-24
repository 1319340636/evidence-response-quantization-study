"""Fixed, outcome-independent protocol and fail-closed file gates."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

PROTOCOL = 'hf-repeatability-v1-20260911'
PARENT_HASH = 'bbe3e66dc7e53e0fa08c656716e113f63b8c5232de8b9fa895de6046b861b9f1'
MODELS = ('qwen35_9b', 'ministral3_8b', 'olmo3_7b')
ROUTES = ('FP16', 'GPTQ_INT4', 'AWQ_INT4')
MAPPINGS = ('original', 'cycle_1', 'cycle_2')
CONFIG = 'configs/hf_repeatability_v1.json'
RUN_ROOT = 'runs/hf_repeatability_v1'
EXPECTED_COUNTS = {'owners':480, 'hard':480, 'probability':480, 'errors':0, 'full':480}


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def code_sha(path):
    """Source-only canonical LF hash, permits transport line-ending changes."""
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def write_sealed(path, value):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload=json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n'
    # Never replace prior state, including a partially written manifest.
    with path.open('x', encoding='utf-8', newline='\n') as f: f.write(payload)
    with Path(str(path)+'.sha256').open('x', encoding='ascii') as f: f.write(file_sha(path)+'\n')


def read_sealed(path):
    path=Path(path)
    if file_sha(path)!=Path(str(path)+'.sha256').read_text().strip():
        raise ValueError(f'SHA-256 mismatch: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def select_cases(rows):
    if len(rows)!=1200 or len({r['case_id'] for r in rows})!=1200:
        raise ValueError('parent must have 1200 unique quartets')
    if Counter(r['negative_label'] for r in rows)!={'NOT ENOUGH INFO':237, 'REFUTES':963}:
        raise ValueError('parent label population drifted')
    key=lambda case:hashlib.sha256((PROTOCOL+'|'+case).encode()).hexdigest()
    selected=[]
    for label,n in [('NOT ENOUGH INFO',24),('REFUTES',96)]:
        selected+=sorted((r['case_id'] for r in rows if r['negative_label']==label), key=key)[:n]
    return sorted(selected, key=key)


def matrix():
    for repeat in (1,2,3):
        for mapping in MAPPINGS:
            for model in MODELS:
                for route in ROUTES: yield repeat,mapping,model,route


def claim_fresh_directory(path):
    Path(path).mkdir(parents=True,exist_ok=False)


def validate_units(units, ids):
    if len(ids)!=len(set(ids)) or len(units)!=4*len(ids): raise ValueError('incomplete quartet units')
    if len({u.owner_key for u in units})!=len(units): raise ValueError('duplicate owners')
    if {u.case_id for u in units}!=set(ids) or any(u.control!='full' for u in units):
        raise ValueError('unit population/control drifted')
    for case in ids:
        if sorted(u.cell_index for u in units if u.case_id==case)!=[0,1,2,3]:
            raise ValueError('quartet cell set drifted')


def validate_score(scores, choice):
    if set(scores)!={'A','B','C'} or not all(isinstance(v,(float,int)) and math.isfinite(v) for v in scores.values()):
        raise ValueError('invalid or nonfinite choice log-scores')
    if choice not in scores or scores[choice]!=max(scores.values()):
        raise ValueError('hard choice disagrees with argmax')


def cell_path(root, cell):
    repeat,mapping,model,route=cell
    return Path(root)/RUN_ROOT/f'repeat_{repeat}'/mapping/f'{model}_{route}'


def verify_complete(directory, config_hash, cell):
    directory=Path(directory); done=read_sealed(directory/'complete.json')
    if tuple(done.get(k) for k in ('repeat','mapping','model','route'))!=tuple(cell):
        raise ValueError('completed cell identity drifted')
    if done.get('config_sha256')!=config_hash or done.get('counts')!=EXPECTED_COUNTS:
        raise ValueError('completed config or structural counts drifted')
    if not done.get('invocation_id') or not isinstance(done.get('pid'),int):
        raise ValueError('missing process identity')
    files=done.get('files',{})
    required={'run.sqlite3','observations.json','observations.json.sha256','invocation.json',
        'invocation.json.sha256','execution.json','execution.json.sha256','audit.json',
        'export/manifest.json','export/records.jsonl'}
    if not required.issubset(files): raise ValueError('missing required file hashes')
    for relative,digest in files.items():
        p=(directory/relative).resolve()
        if not p.is_relative_to(directory.resolve()) or file_sha(p)!=digest:
            raise ValueError('completed file hash/path drifted')
    return done
