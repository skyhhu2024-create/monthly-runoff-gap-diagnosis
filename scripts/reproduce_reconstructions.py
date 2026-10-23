"""Rebuild registered current/historical reconstructions from the Zenodo calendar.

Only monthly GSIM MEAN indices are used. The target query is masked before donor
selection and historical replay. Each decision input is saved before scoring
the current evaluation truth. This script generates no figures.
"""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from gsim_gap_diagnostic.pipeline import run_case
from gsim_gap_diagnostic.scores import event_score,climatology_prediction
from gsim_gap_diagnostic._io import read_frame,new_directory,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--algorithm',nargs='+',choices=['idw','ridge','rf','extratrees','xgboost'],default=['rf'])
    p.add_argument('--seed',nargs='+',type=int,default=[20261014])
    p.add_argument('--case-id',nargs='+',help='Optional registered case IDs; omit to process all8152')
    p.add_argument('--event',choices=['original_event','seasonal_event'],default='original_event')
    args=p.parse_args();root=args.data_root;out=new_directory(args.output)
    reg=read_frame(root/'cohort/cv_case_registry.csv')
    if args.case_id:
        if set(args.case_id)-set(reg.case_id):raise ValueError('Unregistered case ID.')
        reg=reg.loc[reg.case_id.isin(args.case_id)]
    meta=read_frame(root/'inputs/monthly_GSIM/all_station_metadata.csv')
    archive=np.load(root/'inputs/monthly_GSIM/global_observed_calendar.npz',allow_pickle=False)
    q_all=np.where(archive['observed_mask'],archive['q'],np.nan)
    months=np.asarray([int(str(v).split('-')[1])-1 for v in archive['dates']],int)
    anchors=meta.loc[meta.coverage.ge(.9)&meta.category.ne('target')&np.isfinite(meta.lat)&np.isfinite(meta.lon)].copy()
    assert set(reg.station_idx).isdisjoint(anchors.station_idx),'Target/donor pools must be separate.'
    lat=np.deg2rad(anchors.lat.to_numpy(float));lon=np.deg2rad(anchors.lon.to_numpy(float))
    features=[];labels=[];histories=[];current=[]
    for number,row in enumerate(reg.itertuples(index=False),start=1):
        target=q_all[int(row.station_idx)].copy();query=np.arange(row.gap_start,row.gap_end_exclusive)
        target[query]=np.nan
        a=np.sin((lat-np.deg2rad(row.lat))/2)**2+np.cos(lat)*np.cos(np.deg2rad(row.lat))*np.sin((lon-np.deg2rad(row.lon))/2)**2
        order=np.argsort(a,kind='stable')[:100]
        pool=anchors.station_idx.to_numpy(int)[order]
        distance=6371*2*np.arcsin(np.sqrt(np.clip(a[order],0,1)))
        donors=q_all[pool].T.copy()
        for algorithm in args.algorithm:
            for seed in args.seed:
                result=run_case(target,donors,query,months,distance,case_id=row.case_id,station_id=row.station_id,
                    block=row.block,algorithm=algorithm,seed=seed,position=row.position,event=args.event)
                keys={'case_id':row.case_id,'station_id':row.station_id,'block':row.block,'algorithm':algorithm,'seed':seed}
                feature={**result['features'],'cv_test_fold':int(row.cv_test_fold),'source_population':row.source_population,'split':row.split}
                features.append(feature)
                histories.extend([{**keys,**h} for h in result['history_records']])
                current.append({**keys,'query_indices':query.tolist(),'prediction_query':result['prediction_query'].tolist()})
                # The per-case proof precedes all current-target scoring.
                write_json(out/f'case_{number:05d}_{algorithm}_{seed}_before_scoring.json',
                    {'features':feature,'history':result['history_records'],'current_prediction':current[-1],'current_truth_read':False})
                truth=q_all[int(row.station_idx),query].copy()
                baseline=climatology_prediction(target,months,query)
                labels.append({**keys,**event_score(truth,result['prediction_query'],baseline,args.event)})
        print(json.dumps({'completed_stations':number,'registered_stations':len(reg),'case_id':row.case_id}),flush=True)
    pd.DataFrame(features).to_csv(out/'visible_features.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame(histories).to_csv(out/'history_records.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    write_json(out/'current_predictions.json',current)
    pd.DataFrame(labels).to_csv(out/'labels_after_prediction.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    write_json(out/'receipt.json',{'status':'complete','registered_stations':len(reg),'algorithm_rows':len(features),
        'current_target_masked_before_fit':True,'current_predictions_persisted_before_scoring':True,'event':args.event})


if __name__=='__main__':main()

