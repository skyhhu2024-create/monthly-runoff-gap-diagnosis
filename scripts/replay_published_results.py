"""Replay all published M diagnoses from the accompanying Zenodo evidence tables.

No reconstruction fitting, hidden-flow reading or figure generation is needed.
Use --refit to additionally repeat TRAIN isotonic fitting and CAL selection.
"""
from pathlib import Path
import argparse, sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from gsim_gap_diagnostic._io import read_frame,new_directory,write_json
from gsim_gap_diagnostic.diagnosis import diagnose,load_model
from gsim_gap_diagnostic.evaluation import evaluate,cross_validate

ALGORITHMS=['idw','ridge','rf','extratrees','xgboost']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True,help='Unpacked Zenodo data root containing cohort/ and primary_M/')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--refit',action='store_true',help='Also refit original-cohort maps and CAL policies, never on expansion labels')
    p.add_argument('--bootstrap',type=int,default=1000)
    args=p.parse_args();root=args.data_root;out=new_directory(args.output)
    reg=read_frame(root/'cohort/cv_case_registry.csv')
    reference=read_frame(root/'primary_M/train_isotonic_predictions_before_evaluation.csv.gz')
    model_root=Path(__file__).resolve().parents[1]/'configs/frozen_models'
    inputs={
        'original_event':('cohort/visible_features_full.csv.gz','historical_H/frozen_A/labels_for_oof_evaluation.csv.gz'),
        'seasonal_event':('supplementary/seasonal/visible_features_seasonal_event.csv.gz','supplementary/seasonal/labels_for_evaluation.csv.gz')}
    audit=[];metrics=[]
    for event,(feature_file,label_file) in inputs.items():
        f=read_frame(root/feature_file);labels=read_frame(root/label_file)
        f=f.loc[f.algorithm.isin(ALGORITHMS)&f.seed.eq(20261014)].copy()
        labels=labels.loc[labels.algorithm.isin(ALGORITHMS)&labels.seed.eq(20261014)].copy()
        predictions=[]
        for algorithm in ALGORITHMS:
            for fold in range(5):
                model=load_model(model_root/f'M_{event}_{algorithm}_fold{fold}.json')
                te=f.loc[f.algorithm.eq(algorithm)&f.cv_test_fold.eq(fold)]
                result=diagnose(te,model);result['outer_fold']=fold
                predictions.append(result)
        predictions=pd.concat(predictions,ignore_index=True)
        # Persist predictions before accessing the reference or TEST labels.
        predictions.to_csv(out/f'{event}_predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        for algorithm in ALGORITHMS:
            current=predictions.loc[predictions.algorithm.eq(algorithm)].sort_values('case_id').reset_index(drop=True)
            expected=reference.loc[reference.event.eq(event)&reference.algorithm.eq(algorithm)].sort_values('case_id').reset_index(drop=True)
            assert np.array_equal(current.case_id,expected.case_id),'Different registered cases'
            probability_equal=np.allclose(current.probability,expected.probability,rtol=0,atol=1e-12,equal_nan=True)
            exact=np.array_equal(current.adopted,expected.adopted) and np.array_equal(current.eligible,expected.eligible)
            if not probability_equal or not exact:raise AssertionError(f'Published result replay failed: {event}/{algorithm}')
            error=float(np.nanmax(abs(current.probability.to_numpy()-expected.probability.to_numpy())))
            audit.append({'event':event,'algorithm':algorithm,'n_registered':len(current),'probability_max_absolute_error':error,'decisions_exact':exact})
            lab=labels.loc[labels.algorithm.eq(algorithm)]
            for weight in ['station_equal','block_equal']:
                metrics.append({'event':event,'algorithm':algorithm,**evaluate(current,lab,weighting=weight,n_bootstrap=args.bootstrap)})
        if args.refit:
            cross_validate(f,labels,reg,out/f'refit_{event}',method='M',event=event)
    write_json(out/'replay_audit.json',{'status':'PASS','rows_checked':sum(x['n_registered'] for x in audit),'checks':audit})
    write_json(out/'metrics.json',metrics)
    print('PASS: replayed',sum(x['n_registered'] for x in audit),'gap diagnoses; all acceptance decisions match.')


if __name__=='__main__':main()

