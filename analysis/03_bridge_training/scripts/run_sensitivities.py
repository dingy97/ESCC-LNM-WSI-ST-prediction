"""Prespecified QC grid and RNA-stratum refits; unchanged main run is retained."""
from pathlib import Path
from datetime import datetime
import itertools
import math
import json
import subprocess
import sys
R=Path(__file__).resolve().parents[2];T=R/'03_bridge_training'
card=json.loads((R/'07_reports/stage2_primary_results.json').read_text(encoding='utf-8'))
main=f'{card["encoder"]}:{card["group"]}:{card["family"]}'
jobs=[]
for q,f in itertools.product([.90,.95,.99],[.05,.10,.20]):
    if math.isclose(q,.95) and math.isclose(f,.1):continue
    jobs.append((f'QC_q{q}_f{f}',['--only',main,'--quantile',str(q),'--support',str(f)]))
for rna in ['annotated','unannotated']:
    jobs.append((rna+'_main',['--RNA',rna,'--only',main]))
    jobs.append((rna+'_numerical_MLP',['--RNA',rna,'--only','ResNet50:B:MLP']))
note={'at':datetime.now().astimezone().isoformat(),
 'change':'Before any RNA-stratum sensitivity was fitted, add exact stratum-specific negative probe counts to the sensitivity loader. Main all-cell run is unchanged and preserved with original code snapshot.',
 'reason':'Using all-cell controls for a smaller RNA subset would confound annotation sensitivity with an inflated background screen',
 'original_main_run':card['run'],'main_model_was_not_retuned':True}
(T/'config/annotation_control_input_note.json').write_text(json.dumps(note,indent=2),encoding='utf-8')
records=[]
for name,args in jobs:
    out=T/'logs'/f'sensitivity_{name}.stdout.log';err=T/'logs'/f'sensitivity_{name}.stderr.log'
    argv=[sys.executable,'-X','utf8',str(Path(__file__).with_name('train_bridge.py')),'--phase','full']+args
    start=datetime.now().astimezone().isoformat()
    with out.open('w',encoding='utf-8') as stdout,err.open('w',encoding='utf-8') as stderr:
        result=subprocess.run(argv,stdout=stdout,stderr=stderr)
    records.append({'name':name,'argv':argv,'started_at':start,'ended_at':datetime.now().astimezone().isoformat(),'exit_code':result.returncode})
    (T/'sensitivity_execution_log.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(f'{name}: exit={result.returncode}',flush=True)
    if result.returncode:raise RuntimeError(f'Sensitivity failed: {name}; retained logs')
print('All 12 prespecified QC/RNA sensitivity jobs completed',flush=True)
