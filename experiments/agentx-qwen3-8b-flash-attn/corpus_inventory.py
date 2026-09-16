import json,pathlib
from datasets import load_dataset
from aiperf.dataset.loader.weka_trace_models import WekaTrace
from aiperf.dataset.loader.weka_trace import _trace_peak_context_length
p=pathlib.Path.cwd(); ds=load_dataset('semianalysisai/cc-traces-weka-062126',split='train')
rows=[]
for row in ds:
 trace=WekaTrace.model_validate(row)
 peak=_trace_peak_context_length(trace)
 rows.append({'id':trace.id,'peak_context':peak,'eligible_131072':peak<=131072,'eligible_40960':peak<=40960})
result={'dataset':'semianalysisai/cc-traces-weka-062126','revision':'23f152f6f0f9399a85901b89a6458def0ef16729','total_traces':len(rows),'eligible_131072':sum(x['eligible_131072'] for x in rows),'eligible_40960':sum(x['eligible_40960'] for x in rows),'traces':rows}
(p/'corpus-inventory.json').write_text(json.dumps(result,indent=2))
assert result['eligible_131072']>0,result
print({k:v for k,v in result.items() if k!='traces'},flush=True)
