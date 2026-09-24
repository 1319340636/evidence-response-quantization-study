"""Fresh-process HF audit, using unchanged runtime/prompt/scoring primitives."""
from __future__ import annotations
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import uuid

from .hf_repeatability import (CONFIG, RUN_ROOT, PROTOCOL, PARENT_HASH, MODELS, ROUTES,
    MAPPINGS, EXPECTED_COUNTS, file_sha, code_sha, write_sealed, read_sealed,
    select_cases, matrix, cell_path, claim_fresh_directory, validate_units, validate_score)
from .hf_runtime import HFNextTokenRuntime, HFTokenAudit, audit_choice_boundary, HF_DIRECT_LOGIT_METHOD
from .hf_runtime_audit import (build_hf_runtime_audit, load_hf_runtime_audit,
    write_hf_runtime_audit, require_matching_hf_runtime_audit)
from .prompts import load_prompt_contract, render_prompt
from .scoring import score_choice_argmax, HARD_LABEL_METHOD
from .vitaminc import id_manifest_sha256
from .vitaminc_inputs import load_vitaminc_split, build_vitaminc_units
from .vitaminc_store_v4 import VitaminCRunIdentityV4, VitaminCRunStoreV4

ARTIFACTS={
 ('qwen35_9b','GPTQ_INT4'):'runs/qwen35_9b_gptq/artifact_audit.json',
 ('qwen35_9b','AWQ_INT4'):'runs/hf_awq_artifacts_v1/qwen35_9b/artifact_audit.json',
 ('ministral3_8b','GPTQ_INT4'):'runs/ministral_hf_artifacts_v1/gptq/artifact_audit.json',
 ('ministral3_8b','AWQ_INT4'):'runs/ministral_hf_artifacts_v1/awq/artifact_audit.json',
 ('olmo3_7b','GPTQ_INT4'):'runs/olmo3_hf_artifacts_v1/formal/gptq/artifact_audit.json',
 ('olmo3_7b','AWQ_INT4'):'runs/olmo3_hf_artifacts_v1/formal/awq/artifact_audit.json',
}

def now(): return datetime.now(timezone.utc).isoformat()

def historical(root, model, route, mapping='original'):
    return root/f'runs/vitaminc_hf_balanced_mapping_v2/formal/{mapping}/{model}_{route}/audit.json'

def source_closure(root):
    pending=['hf_repeatability_run','hf_repeatability_analysis']; seen=set();result={}
    while pending:
        name=pending.pop()
        if name in seen: continue
        seen.add(name); p=root/f'src/qer_fv/{name}.py'
        result[p.relative_to(root).as_posix()]=code_sha(p)
        for node in ast.walk(ast.parse(p.read_text(encoding='utf-8-sig'))):
            if isinstance(node,ast.ImportFrom) and node.level==1 and node.module:
                dep=node.module.split('.')[0]
                if (root/f'src/qer_fv/{dep}.py').exists(): pending.append(dep)
    result['scripts/run_hf_repeatability.py']=code_sha(root/'scripts/run_hf_repeatability.py')
    return result

def freeze(root):
    parent_path=root/'data/manifests/vitaminc_dose_subset_ids.txt'
    parent_ids=parent_path.read_text().splitlines()
    if len(parent_ids)!=1200 or id_manifest_sha256(parent_ids)!=PARENT_HASH:
        raise ValueError('parent ID hash mismatch')
    metadata_path=root/'data/manifests/vitaminc_strict_quartets.jsonl'
    lookup={r['case_id']:r for r in (json.loads(x) for x in metadata_path.read_text(encoding='utf-8').splitlines())}
    ids=select_cases([lookup[i] for i in parent_ids])
    config={'protocol':PROTOCOL,'created_utc':now(),'parent_ids_sha256':PARENT_HASH,
        'ids':ids,'ids_sha256':id_manifest_sha256(ids), 'input_order':'case_id ascending, cell_index 0..3',
        'cases':[{k:lookup[i][k] for k in ('case_id','page','negative_label')} for i in ids],
        'pages':len({lookup[i]['page'] for i in ids}), 'expected_inputs_per_cell':480,'repeats':3,
        'models':list(MODELS),'routes':list(ROUTES),'mappings':list(MAPPINGS),
        'engineering_absolute_logscore_threshold':1e-6,
        'no_new_significance_tests':True,'no_requantization':True,
        'files':{p:file_sha(root/p) for p in ['configs/freeze_v1.json','configs/prompt_contract_v2.json',
            'data/manifests/vitaminc_dose_subset_ids.txt','data/manifests/vitaminc_strict_quartets.jsonl',
            'data/manifests/vitaminc_splits_summary.json']},
        'source_lf_sha256':source_closure(root)}
    write_sealed(root/CONFIG,config)
    print(json.dumps({'frozen_cases':len(ids),'pages':config['pages'],'inputs':38880,'config_sha256':file_sha(root/CONFIG)}))

def load_config(root):
    config=read_sealed(root/CONFIG)
    if config['protocol']!=PROTOCOL or len(config['ids'])!=120 or id_manifest_sha256(config['ids'])!=config['ids_sha256']:
        raise ValueError('repeat protocol identity mismatch')
    if select_cases(_parent_rows(root))!=config['ids']: raise ValueError('selection changed')
    if config['models']!=list(MODELS) or config['routes']!=list(ROUTES) or config['mappings']!=list(MAPPINGS):
        raise ValueError('condition matrix drifted')
    for p,digest in config['files'].items():
        if file_sha(root/p)!=digest: raise ValueError('source input drifted: '+p)
    for p,digest in config['source_lf_sha256'].items():
        if code_sha(root/p)!=digest: raise ValueError('executable source drifted: '+p)
    return config

def _parent_rows(root):
    ids=(root/'data/manifests/vitaminc_dose_subset_ids.txt').read_text().splitlines()
    if id_manifest_sha256(ids)!=PARENT_HASH: raise ValueError('parent hash mismatch')
    lookup={r['case_id']:r for r in (json.loads(x) for x in (root/'data/manifests/vitaminc_strict_quartets.jsonl').read_text(encoding='utf-8').splitlines())}
    return [lookup[i] for i in ids]

def units_for(root, config):
    parent=load_vitaminc_split(root,split='dose_subset')
    lookup={q.case_id:q for q in parent}
    selected=[lookup[i] for i in config['ids']]
    for q,meta in zip(selected,config['cases']):
        if (q.case_id,q.page,q.negative_label)!=(meta['case_id'],meta['page'],meta['negative_label']):
            raise ValueError('raw quartet metadata differs from selection')
    units=[u for u in build_vitaminc_units(selected) if u.control=='full']
    validate_units(units, config['ids'])
    return units

def packages(route):
    names=['gptqmodel','torch','torchvision','transformers']
    if route=='AWQ_INT4': names.append('compressed-tensors')
    return {n:importlib.metadata.version(n) for n in names}

def artifact_args(root,model,route):
    if route=='FP16':return {}
    return {('gptq_artifact_audit_path' if route=='GPTQ_INT4' else 'awq_artifact_audit_path'):root/ARTIFACTS[model,route]}

def gpu_snapshot():
    return subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total,driver_version','--format=csv,noheader'],text=True).strip()

def numerical_context():
    import torch
    names=['CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','CUBLAS_WORKSPACE_CONFIG','NVIDIA_TF32_OVERRIDE',
           'CUDA_LAUNCH_BLOCKING','CUDNN_LOGLEVEL_DBG','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE',
           'TORCH_BLAS_PREFER_CUBLASLT','OMP_NUM_THREADS','MKL_NUM_THREADS','PYTHONHASHSEED',
           'PYTORCH_CUDA_ALLOC_CONF','PYTORCH_ALLOC_CONF']
    return {'torch':torch.__version__,'cuda_build':torch.version.cuda,
        'cudnn_version':torch.backends.cudnn.version(),'cudnn_benchmark':torch.backends.cudnn.benchmark,
        'cudnn_deterministic':torch.backends.cudnn.deterministic,
        'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
        'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
        'fp16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
        'float32_matmul_precision':torch.get_float32_matmul_precision(),
        'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),
        'environment':{k:os.environ.get(k) for k in names}}

def require_same_execution(current,expected):
    if current!=expected:raise ValueError('numerical environment or actual model execution drifted')

def preflight(root):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0,1':raise ValueError('preflight requires CUDA_VISIBLE_DEVICES=0,1')
    config=load_config(root)
    base=root/RUN_ROOT;base.mkdir(parents=True,exist_ok=True)
    if (base/'preflight.json').exists(): raise FileExistsError('preflight already sealed; verify instead')
    units_for(root,config)
    freeze_config=json.loads((root/'configs/freeze_v1.json').read_text())
    raw=freeze_config['vitaminc']
    if file_sha(root/raw['test_path']).lower()!=raw['test_sha256'].lower():raise ValueError('raw test SHA mismatch')
    certs={};identities={}
    for model in MODELS:
        for route in ROUTES:
            p=historical(root,model,route);old=load_hf_runtime_audit(p)
            probe=HFTokenAudit(old.choice_token_ids,old.generation_input_ids_sha256,
                old.generation_prompt_sha256,old.chat_template_sha256,old.direct_logit_method)
            fresh=build_hf_runtime_audit(model_path=old.model_path,model_key=model,precision=route,
                package_versions=packages(route),load_policy=old.load_policy,
                prompt_contract_path=root/'configs/prompt_contract_v2.json',token_audit=probe,
                **artifact_args(root,model,route))
            for mapping in MAPPINGS:
                h=historical(root,model,route,mapping)
                require_matching_hf_runtime_audit(fresh,h)
                certs[h.relative_to(root).as_posix()]=file_sha(h)
            identities[f'{model}_{route}']=fresh.to_record()
            print(f'preflight weights/software/certificates OK: {model} {route}',flush=True)
    write_sealed(base/'preflight.json',{'config_sha256':file_sha(root/CONFIG),
        'created_utc':now(),'gpu_snapshot':gpu_snapshot(),'historical_certificates':certs,
        'identities':identities,'numerical_context':numerical_context(),'raw_test_sha256':file_sha(root/raw['test_path']),
        'raw_test_path':raw['test_path'], 'model_files_verified':True})

def require_preflight(root):
    value=read_sealed(root/RUN_ROOT/'preflight.json')
    if value['config_sha256']!=file_sha(root/CONFIG):raise ValueError('preflight config mismatch')
    if gpu_snapshot()!=value['gpu_snapshot']:raise ValueError('GPU/driver changed')
    require_same_execution(numerical_context(),value['numerical_context'])
    for p,digest in value['historical_certificates'].items():
        if file_sha(root/p)!=digest:raise ValueError('historical audit changed')
    if file_sha(root/value['raw_test_path'])!=value['raw_test_sha256']:raise ValueError('raw inputs changed')
    return value

def structure(path,expected=480):
    with sqlite3.connect(f'file:{Path(path).as_posix()}?mode=ro',uri=True) as db:
        controls=dict(db.execute('SELECT control,COUNT(*) FROM owners GROUP BY control'))
        h=db.execute("SELECT COUNT(*),COALESCE(SUM(status='error'),0) FROM hard_results").fetchone()
        p=db.execute("SELECT COUNT(*),COALESCE(SUM(status='error'),0) FROM probability_results").fetchone()
        n=db.execute('SELECT COUNT(*) FROM owners').fetchone()[0]
    out={'owners':n,'hard':h[0],'probability':p[0],'errors':h[1]+p[1],'full':controls.get('full',0)}
    if controls!={'full':expected} or n!=expected or h!=(expected,0) or p!=(expected,0):
        raise ValueError('cell structural gate failed: '+str(out))
    return out

def collect(runtime,certificate,contract,units,directory,cell,config_hash):
    directory=Path(directory);db_path=directory/'run.sqlite3'
    if db_path.exists():raise FileExistsError('never reuse a database for a repeat')
    repeat,mapping,model,route=cell
    identity=VitaminCRunIdentityV4(protocol_version=f'{PROTOCOL}:{repeat}:{mapping}',
        split='repeatability_subset',split_sha256=config_hash,
        audit_certificate_sha256=certificate.certificate_sha256,
        prompt_contract_sha256=certificate.prompt_contract_sha256,model_key=model,quantization=route,
        expected_full=len(units),expected_no_evidence=0,expected_knowledge_only=0)
    rows=[]
    with VitaminCRunStoreV4(db_path,identity) as store:
        for index,u in enumerate(units,1):
            rendered=render_prompt(contract,u.prompt_input,control='full',mapping_variant=mapping)
            scores=runtime.score_messages(rendered.messages,certificate.choice_token_ids)
            validate_score(scores.choice_logprobs,scores.scored_choice)
            if not all(math.isfinite(v) for v in scores.choice_probabilities.values()):raise ValueError('nonfinite probability')
            scored=score_choice_argmax(rendered.choice_to_label,scores.choice_logprobs)
            if scored.choice!=scores.scored_choice:raise ValueError('scoring argmax mismatch')
            metadata={'case_id':u.case_id,'page':u.page,'control':'full','cell_index':u.cell_index,
                'claim_index':u.claim_index,'evidence_index':u.evidence_index,'sample_id':u.sample_id,
                'negative_label':u.prompt_input.metadata['negative_label'],'mapping_variant':mapping}
            common={'choice_logprobs':dict(scores.choice_logprobs),'scored_choice':scored.choice,
                'scored_label':scored.label,'choice_to_label':dict(rendered.choice_to_label),
                'generation_input_ids_sha256':scores.generation_input_ids_sha256,
                'generation_prompt_sha256':scores.generation_prompt_sha256,'prompt_sha256':rendered.prompt_sha256,
                'token_ids':dict(certificate.choice_token_ids)}
            store.put_hard_success(u.owner_key,metadata=metadata,payload={**common,'hard_label_method':HARD_LABEL_METHOD})
            store.put_probability_success(u.owner_key,metadata=metadata,payload={**common,
                'choice_probabilities':dict(scores.choice_probabilities),'direct_logit_method':scores.direct_logit_method,
                'label_probability_mass':scores.label_probability_mass})
            rows.append({'owner_key':u.owner_key,**metadata,**common})
            if index%16==0: print(f'completed={index}/{len(units)} errors=0',flush=True)
        store.export_complete(directory/'export')
    structure(db_path,len(units))
    return rows

def run_cell(root,cell):
    if cell not in set(matrix()):raise ValueError('unknown repeat condition')
    config=load_config(root);require_preflight(root)
    directory=cell_path(root,cell);claim_fresh_directory(directory)
    repeat,mapping,model,route=cell
    invocation={'invocation_id':str(uuid.uuid4()),'pid':os.getpid(),'repeat':repeat,'mapping':mapping,
        'model':model,'route':route,'started_utc':now(),'config_sha256':file_sha(root/CONFIG),
        'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'gpu_snapshot':gpu_snapshot()}
    if invocation['cuda_visible_devices']!='0,1':raise ValueError('CUDA_VISIBLE_DEVICES must remain 0,1')
    write_sealed(directory/'invocation.json',invocation)
    units=units_for(root,config)
    old=load_hf_runtime_audit(historical(root,model,route,mapping))
    runtime=HFNextTokenRuntime.from_pretrained(old.model_path,route,model_key=model)
    contract=load_prompt_contract(root/'configs/prompt_contract_v2.json')
    probe=render_prompt(contract,units[0].prompt_input,control='full',mapping_variant=mapping)
    token_audit=audit_choice_boundary(runtime.tokenizer,probe.messages,tuple(probe.choice_to_label))
    cert=build_hf_runtime_audit(model_path=old.model_path,model_key=model,precision=route,
        package_versions=packages(route),load_policy=runtime.load_policy,
        prompt_contract_path=root/'configs/prompt_contract_v2.json',token_audit=token_audit,
        **artifact_args(root,model,route))
    require_matching_hf_runtime_audit(cert,historical(root,model,route,mapping))
    write_hf_runtime_audit(cert,directory/'audit.json')
    # Capture actual module placement, not only the requested auto/balanced policy.
    loaded=runtime._loaded_model
    forward=getattr(loaded,'model',loaded) if route=='GPTQ_INT4' else loaded
    device_map=getattr(forward,'hf_device_map',getattr(loaded,'hf_device_map',None))
    execution={'actual_device_map':{str(k):str(v) for k,v in (device_map or {}).items()},
        'eval_training_flag':getattr(forward,'training',None),'packages':packages(route),
        'requested_load_policy':dict(runtime.load_policy),'gpu_snapshot':gpu_snapshot(),
        'numerical_context':numerical_context(),
        'parameter_dtype_devices':dict(Counter(f'{p.dtype}/{p.device}' for p in forward.parameters())),
        'module_classes':dict(Counter(f'{type(m).__module__}.{type(m).__qualname__}' for m in forward.modules())),
        'attention_implementation':str(getattr(getattr(forward,'config',None),'_attn_implementation',None))}
    if execution['eval_training_flag'] is not False:raise ValueError('model is not in eval mode')
    if repeat>1:
        first=cell_path(root,(1,mapping,model,route))
        from .hf_repeatability import verify_complete
        verify_complete(first,file_sha(root/CONFIG),(1,mapping,model,route))
        require_same_execution(execution,read_sealed(first/'execution.json'))
    write_sealed(directory/'execution.json',execution)
    rows=collect(runtime,cert,contract,units,directory,cell,file_sha(root/CONFIG))
    write_sealed(directory/'observations.json',{'rows':rows})
    files={p.relative_to(directory).as_posix():file_sha(p) for p in directory.rglob('*') if p.is_file()}
    write_sealed(directory/'complete.json',{**invocation,'finished_utc':now(),
        'counts':structure(directory/'run.sqlite3'),'files':files})
    print('cell structural gate passed',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['freeze','preflight','cell']);p.add_argument('--project-root',type=Path,required=True)
    p.add_argument('--repeat',type=int);p.add_argument('--mapping');p.add_argument('--model');p.add_argument('--route')
    a=p.parse_args();root=a.project_root.resolve()
    if a.mode=='freeze':freeze(root)
    elif a.mode=='preflight':preflight(root)
    else:run_cell(root,(a.repeat,a.mapping,a.model,a.route))

if __name__=='__main__':main()
