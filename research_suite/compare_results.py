import argparse
from pathlib import Path
import numpy as np
from common import read,dump,sha
from metrics import interval

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Paired case comparisons on the same evaluation cohort, without retraining.');p.add_argument('--baseline',required=True);p.add_argument('--candidate',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();a=read(args.baseline);b=read(args.candidate)
    if Path(args.output).exists():raise ValueError('Output exists')
    ar={c['case_id']:c for c in a['cases']};br={c['case_id']:c for c in b['cases']}
    if set(ar)!=set(br):raise ValueError('Paired comparison requires exactly the same cases')
    result={'baseline_sha256':sha(args.baseline),'candidate_sha256':sha(args.candidate),
        'note':'Conditional paired case bootstrap. Does not adjust for model selection or repeated experiments. Verify participant independence.'}
    for name in ['whole_tumor','tumor_core','enhancing_tumor']:
        deltas=[]
        for cid in sorted(ar):
            x=ar[cid]['regions'][name]['dice'];y=br[cid]['regions'][name]['dice']
            if (x is None)!=(y is None):raise ValueError('Defined-case mismatch; report empty-region conventions separately')
            if x is not None:deltas.append(100*(y-x))
        result[name]={'paired_cases':len(deltas),'mean_difference_percentage_points':float(np.mean(deltas)) if deltas else None,
            'conditional_95ci_pp':interval(deltas),'improved':sum(v>0 for v in deltas),'worse':sum(v<0 for v in deltas)}
    dump(args.output,result);print(result)
