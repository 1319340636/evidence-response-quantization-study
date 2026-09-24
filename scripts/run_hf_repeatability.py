"""Serial isolated repeatability queue; no partial-outcome analysis."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback

from qer_fv.hf_repeatability import (CONFIG,RUN_ROOT,matrix,cell_path,verify_complete,
    file_sha,read_sealed,write_sealed)
from qer_fv.hf_repeatability_run import load_config,require_preflight,structure,now


def cell_state(root,cell,config_hash):
    directory=cell_path(root,cell)
    if not directory.exists():return 'new'
    if not (directory/'complete.json').exists():
        raise ValueError(f'incomplete prior process at {directory}; no automatic mixing/resume')
    verify_complete(directory,config_hash,cell)
    structure(directory/'run.sqlite3')
    return 'complete'


def command_for(root,cell):
    repeat,mapping,model,route=cell
    return [sys.executable,'-m','qer_fv.hf_repeatability_run','cell','--project-root',str(root),
        '--repeat',str(repeat),'--mapping',mapping,'--model',model,'--route',route]


def finish(root,config_hash):
    runs=[];gates=[];executions={}
    for cell in matrix():
        directory=cell_path(root,cell)
        done=verify_complete(directory,config_hash,cell)
        structure(directory/'run.sqlite3')
        invocation=read_sealed(directory/'invocation.json')
        for k in ('repeat','mapping','model','route','pid','invocation_id','config_sha256'):
            if done[k]!=invocation[k]:raise ValueError('invocation/completion mismatch')
        execution=read_sealed(directory/'execution.json')
        key=cell[1:]
        if key in executions and executions[key]!=execution:
            raise ValueError('actual module placement or runtime environment differs across repeats')
        executions[key]=execution
        rows=read_sealed(directory/'observations.json')['rows']
        runs.append({**invocation,'rows':rows})
        gates.append({'cell':list(cell),'complete_sha256':file_sha(directory/'complete.json'),'counts':done['counts']})
    if len({r['invocation_id'] for r in runs})!=81:raise ValueError('duplicate process invocation IDs')
    # No scientific calculation has been performed before the full gate above.
    base=root/RUN_ROOT
    from qer_fv.hf_repeatability_analysis import summarize_runs
    result=summarize_runs(runs,expected_cases=120)
    write_sealed(base/'structural_gate.json',{'gate_passed':True,'cells':gates,
        'config_sha256':config_hash,'inputs':38880,'completed_utc':now()})
    write_sealed(base/'repeatability_report.json',result)
    report='# Fixed-artifact HF repeatability audit\n\n'
    report+='Population: 120 quartets; 27 conditions; three independent processes; 38,880 inputs.\n\n'
    report+='This is a run-to-run numerical audit, not a new significance test or full-population replication.\n\n'
    for key in ('classification','engineering_gate','log_score_span','hard_label_flips','limitations'):
        if key in result:report+=f'## {key}\n\n```json\n'+json.dumps(result[key],ensure_ascii=False,indent=2,allow_nan=False)+'\n```\n\n'
    with (base/'repeatability_report.md').open('x',encoding='utf-8',newline='\n') as f:f.write(report)
    with (base/'repeatability_report.md.sha256').open('x') as f:f.write(file_sha(base/'repeatability_report.md')+'\n')
    write_sealed(base/'done.json',{'completed_utc':now(),'config_sha256':config_hash,
        'report_sha256':file_sha(base/'repeatability_report.json'),'structural_gate_sha256':file_sha(base/'structural_gate.json')})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,required=True);parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();root=args.project_root.resolve()
    config=load_config(root);config_hash=file_sha(root/CONFIG)
    if args.dry_run:
        print(json.dumps({'cells':list(matrix()),'inputs_per_cell':480,'total_inputs':38880,
                          'pages':config['pages'],'config_sha256':config_hash},indent=2));return
    require_preflight(root)
    base=root/RUN_ROOT;lock=base/'queue.lock';lock.mkdir(exist_ok=False)
    (lock/'pid').write_text(str(os.getpid()))
    child=None;current='preflight'
    def stop(signum,frame): raise RuntimeError(f'queue interrupted by signal {signum}')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        if (base/'failed.json').exists() or (base/'done.json').exists():raise ValueError('queue already terminal; no automatic rerun')
        (base/'launcher.pid').write_text(str(os.getpid()))
        (base/'logs').mkdir(exist_ok=True)
        # Check all existing cells before creating anything new.
        for cell in matrix():cell_state(root,cell,config_hash)
        for cell in matrix():
            current='/'.join(map(str,cell))
            (base/'current.txt').write_text(current+'\n')
            if cell_state(root,cell,config_hash)=='complete':continue
            log=base/'logs'/('_'.join(map(str,cell))+'.log')
            with log.open('x',encoding='utf-8') as stream:
                env=dict(os.environ,PYTHONPATH=str(root/'src'),CUDA_VISIBLE_DEVICES='0,1',PYTHONUNBUFFERED='1')
                child=subprocess.Popen(command_for(root,cell),cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT)
                (base/'runner.pid').write_text(str(child.pid))
                code=child.wait();child=None
            if code!=0:raise RuntimeError(f'cell exited {code}; log={log}')
            if cell_state(root,cell,config_hash)!='complete':raise ValueError('runner exited without completion')
            with (base/'status.tsv').open('a',encoding='utf-8') as stream:stream.write(f'{current}\tstructural-ok\t{now()}\n')
            print(f'completed {current}',flush=True)
        current='final-structure-and-repeatability-analysis'
        (base/'current.txt').write_text(current+'\n')
        finish(root,config_hash)
        (base/'current.txt').write_text('done\n')
        print('all81 cells, structural gate and repeatability analysis completed',flush=True)
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        if not (base/'failed.json').exists():
            write_sealed(base/'failed.json',{'stage':current,'time':now(),'error':str(error),'traceback':traceback.format_exc()})
        raise
    finally:
        (lock/'pid').unlink(missing_ok=True);lock.rmdir()

if __name__=='__main__':main()
