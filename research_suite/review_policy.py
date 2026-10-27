import argparse
from pathlib import Path
import numpy as np
from common import read,dump,sha,csv_out

def extract(report,key):
    items=[c for c in report['cases'] if c.get(key) is not None and c['regions']['whole_tumor']['dice'] is not None]
    return items,np.array([c[key] for c in items]),np.array([c['regions']['whole_tumor']['dice'] for c in items])

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Fit a development review threshold, then evaluate the frozen rule on another cohort.');sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('fit');a.add_argument('--report',required=True);a.add_argument('--plan',required=True);a.add_argument('--output',required=True)
    a.add_argument('--uncertainty',default='mean_brain_entropy');a.add_argument('--failure-dice',type=float,default=.7);a.add_argument('--target-sensitivity',type=float,default=.8)
    b=sub.add_parser('evaluate');b.add_argument('--report',required=True);b.add_argument('--plan',required=True);b.add_argument('--policy',required=True);b.add_argument('--output',required=True)
    args=p.parse_args();report=read(args.report);plan=read(args.plan)
    if Path(args.output).exists():raise ValueError('Output exists')
    if args.command=='fit':
        if report['evaluation']!='development_validation':raise ValueError('Fit only on development validation')
        if not 0<args.target_sensitivity<=1 or not 0<args.failure_dice<1:raise ValueError('Invalid policy targets')
        items,u,dice=extract(report,args.uncertainty);fail=dice<args.failure_dice
        if not len(items) or not fail.any():raise ValueError('No defined development failures; cannot fit a failure-detection rule')
        candidates=[float(t) for t in np.unique(u) if np.mean(u[fail]>=t)>=args.target_sensitivity]
        threshold=max(candidates)
        dump(args.output,{'uncertainty':args.uncertainty,'threshold':threshold,'failure_dice':args.failure_dice,
            'target_development_sensitivity':args.target_sensitivity,'checkpoint_sha256':plan['checkpoint_sha256'],
            'development_report_sha256':sha(args.report),'note':'Development-fitted rule. Not independently validated. Undefined uncertainty always requires review.'})
    else:
        policy=read(args.policy)
        if plan['checkpoint_sha256']!=policy['checkpoint_sha256']:raise ValueError('Policy was fitted for a different model')
        if sha(args.report)==policy['development_report_sha256']:raise ValueError('Use a different cohort for policy evaluation')
        items,u,dice=extract(report,policy['uncertainty']);fail=dice<policy['failure_dice'];flags=u>=policy['threshold']
        result={'policy_sha256':sha(args.policy),'evaluation':report['evaluation'],'defined_cases':len(items),
            'undefined_cases_require_review':len(report['cases'])-len(items),'flagged_defined_cases':int(flags.sum()),
            'failure_sensitivity':float(flags[fail].mean()) if fail.any() else None,
            'flag_precision':float(fail[flags].mean()) if flags.any() else None,
            'retained_fraction_all_cases':float((~flags).sum()/len(report['cases'])) if report['cases'] else None,
            'retained_mean_dice':float(dice[~flags].mean()) if (~flags).any() else None,
            'note':'External validation status depends on verified cohort independence, not on the filename.'}
        dump(args.output,result);print(result)
