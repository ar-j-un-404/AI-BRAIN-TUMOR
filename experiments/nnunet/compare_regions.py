import argparse,json,sys,hashlib
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(a.project/'research_suite'));from metrics import interval
    aa=json.loads(a.baseline.read_text());bb=json.loads(a.candidate.read_text())
    ar={c['case_id']:c for c in aa['cases']};br={c['case_id']:c for c in bb['cases']}
    if len(ar)!=73 or set(ar)!=set(br):raise ValueError('Cohort mismatch')
    result={'note':'Candidate minus baseline. Conditional paired case intervals exclude model-selection/training uncertainty. Differences use mutually defined Dice only; undefined cases are listed, not filled with perfect scores. Reduced-budget development benchmark; not equal compute.',
        'baseline_sha256':hashlib.sha256(a.baseline.read_bytes()).hexdigest(),'candidate_sha256':hashlib.sha256(a.candidate.read_bytes()).hexdigest()}
    for region in ('whole_tumor','tumor_core','enhancing_tumor'):
        differences=[];excluded=[]
        for cid in sorted(ar):
            x=ar[cid]['regions'][region];y=br[cid]['regions'][region]
            if x['reference_voxels']!=y['reference_voxels']:raise ValueError('Reference voxel counts differ')
            if x['dice'] is None or y['dice'] is None:excluded.append({'case_id':cid,'baseline_dice':x['dice'],'candidate_dice':y['dice']})
            else:differences.append(100*(y['dice']-x['dice']))
        result[region]={'paired_defined_cases':len(differences),'excluded':excluded,'mean_difference_pp':sum(differences)/len(differences) if differences else None,'conditional_95ci_pp':interval(differences),'improved':sum(v>0 for v in differences),'worse':sum(v<0 for v in differences)}
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
