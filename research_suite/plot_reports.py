import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import read,rows

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Create publication-style plots from real saved evaluation outputs only.');p.add_argument('--report',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();report=read(args.report);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if 'confusion_matrix' in report:
        cm=np.array(report['confusion_matrix']);fig,ax=plt.subplots(figsize=(5,4));ax.imshow(cm,cmap='Blues')
        for (i,j),v in np.ndenumerate(cm):ax.text(j,i,str(v),ha='center',va='center')
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Healthy','Tumor'],yticklabels=['Healthy','Tumor'],xlabel='Predicted',ylabel='Reference')
        fig.tight_layout();fig.savefig(out/'confusion_matrix.png',dpi=200);plt.close(fig)
        bins=report['reliability_bins'];fig,ax=plt.subplots(figsize=(5,4));ax.plot([0,1],[0,1],'--',color='gray')
        ax.plot([b['probability'] for b in bins],[b['observed_positive_rate'] for b in bins],'o-')
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean model score',ylabel='Observed tumor proportion',title='Reliability diagram')
        fig.tight_layout();fig.savefig(out/'calibration.png',dpi=200);plt.close(fig)
        curves={'classification':report['risk_coverage']}
    else:
        cases=report['cases'];fig,ax=plt.subplots(figsize=(7,4));names=['whole_tumor','tumor_core','enhancing_tumor']
        values=[[c['regions'][n]['dice'] for c in cases if c['regions'][n]['dice'] is not None] for n in names]
        ax.boxplot(values,tick_labels=['WT','TC','ET']);ax.set(ylim=(0,1),ylabel='Case Dice',title='Defined cases only; see report for empty-region counts')
        fig.tight_layout();fig.savefig(out/'segmentation_dice.png',dpi=200);plt.close(fig)
        curves={k:v for k,v in report.items() if k.endswith('_risk_coverage')}
        worst=sorted(cases,key=lambda c:c['regions']['whole_tumor']['dice'] if c['regions']['whole_tumor']['dice'] is not None else 2)[:10]
        from common import dump
        dump(out/'failure_review.json',worst)
    fig,ax=plt.subplots(figsize=(6,4))
    for name,value in curves.items():
        curve=value['curve']
        if curve:ax.plot([c['coverage'] for c in curve],[c['risk'] for c in curve],label=name.replace('_risk_coverage',''))
    ax.set(xlabel='Retained fraction',ylabel='Risk',xlim=(0,1));ax.legend(fontsize=7);fig.tight_layout();fig.savefig(out/'risk_coverage.png',dpi=200);plt.close(fig)
    print('Saved figures:',out)
