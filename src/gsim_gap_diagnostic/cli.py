"""Command-line entry points for reconstruction, fitting and published-result replay."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from ._io import read_frame, write_json, sha256, new_directory


def load_monthly_inputs(flow_path, metadata_path, gap_start, gap_end):
    frame = pd.read_csv(flow_path,float_precision='round_trip')
    if not {'date','target'}.issubset(frame): raise ValueError('Flow CSV needs date and target columns.')
    dates=pd.to_datetime(frame.date,errors='raise').dt.to_period('M')
    if dates.duplicated().any() or not np.all(np.diff(dates.astype('int64'))==1):
        raise ValueError('Keep every calendar month exactly once, in increasing order; use empty cells for missing flows.')
    start=pd.Period(gap_start,'M');end=pd.Period(gap_end,'M')
    q=np.flatnonzero(((dates>=start)&(dates<=end)).to_numpy())
    if end<start or len(q)!=end.ordinal-start.ordinal+1: raise ValueError('Complete inclusive gap bounds must be inside the input calendar.')
    metadata=pd.read_csv(metadata_path,dtype={'station_id':str},float_precision='round_trip')
    if not {'station_id','latitude','longitude'}.issubset(metadata): raise ValueError('Metadata needs station_id, latitude, longitude.')
    if metadata.station_id.duplicated().any(): raise ValueError('Duplicate station metadata.')
    meta=metadata.set_index('station_id')
    donor_names=[c for c in frame if c not in ('date','target')]
    if not set(['target']+donor_names).issubset(meta.index): raise ValueError('Every target/donor column must have matching metadata station_id.')
    for column in ['target']+donor_names: frame[column]=pd.to_numeric(frame[column],errors='raise')
    y=frame.target.to_numpy(float)
    if not np.isnan(y[q]).all(): raise ValueError('Remove every current-gap target value before reconstruction; scoring truth must be stored separately.')
    target=meta.loc['target'];dl=meta.loc[donor_names]
    lat=np.deg2rad(dl.latitude.to_numpy(float));lon=np.deg2rad(dl.longitude.to_numpy(float))
    tlat=np.deg2rad(float(target.latitude));tlon=np.deg2rad(float(target.longitude))
    h=np.sin((lat-tlat)/2)**2+np.cos(lat)*np.cos(tlat)*np.sin((lon-tlon)/2)**2
    distances=6371*2*np.arcsin(np.sqrt(np.clip(h,0,1)))
    if 'distance_km' in dl:
        exact=pd.to_numeric(dl.distance_km,errors='raise').to_numpy(float)
        if not np.isfinite(exact).all() or np.any(exact<0): raise ValueError('distance_km must be finite and nonnegative for every candidate.')
        if not np.allclose(distances,exact,rtol=1e-7,atol=1e-5): raise ValueError('Provided geographic distances disagree with station coordinates.')
        distances=exact
    # Candidate CSV is already donor-pool screened; preserve original nearest100 rule.
    order=np.argsort(distances,kind='stable')[:100]
    donors=frame[donor_names].to_numpy(float)[:,order]
    return frame, y, donors, q, dates.dt.month.to_numpy()-1, distances[order], target, [donor_names[i] for i in order]


def reconstruct_command(args):
    from .pipeline import run_case
    from .spatial import block_id
    from .diagnosis import diagnose, load_model
    out=new_directory(args.output)
    frame,y,d,q,months,dist,target,names=load_monthly_inputs(args.flow,args.metadata,args.gap_start,args.gap_end)
    position=args.position
    if position is None:
        before=np.isfinite(y[:q[0]]).any();after=np.isfinite(y[q[-1]+1:]).any()
        position='interior' if before and after else 'leading' if after else 'trailing'
    # Support-direction restrictions are explicit; leading/trailing cannot use forbidden target months.
    result=run_case(y,d,q,months,dist,case_id=args.case_id,station_id=args.station_id,
        block=block_id(float(target.latitude),float(target.longitude)),algorithm=args.algorithm,
        seed=args.seed,position=position,event=args.event,input_preflight=args.input_preflight)
    features=pd.DataFrame([result['features']]);features.to_csv(out/'visible_features.csv',index=False)
    pd.DataFrame(result['history_records']).to_csv(out/'history_records.csv',index=False)
    pd.DataFrame({'date':frame.date.iloc[q].to_numpy(),'prediction':result['prediction_query']}).to_csv(out/'current_reconstruction.csv',index=False)
    write_json(out/'history_windows.json',result['history_windows'])
    if args.model:
        model=load_model(args.model,trusted_pickle=args.trusted_local_pickle)
        if model.get('event','original_event')!=args.event: raise ValueError('Requested event differs from frozen model event.')
        diagnose(features,model).to_csv(out/'decisions.csv',index=False)
    write_json(out/'run_receipt.json',{'status':'complete','current_truth_read':False,
        'case_id':args.case_id,'position':position,'algorithm':args.algorithm,'seed':args.seed,
        'event':args.event,'candidate_station_columns':names,'input_preflight':result['input_preflight'],
        'model_summary':result['model_summary'],
        'input_sha256':{'flow':sha256(args.flow),'metadata':sha256(args.metadata)},
        'diagnostic_model_sha256':sha256(args.model) if args.model else None})
    print(json.dumps({'status':'complete','output':str(out),'historical_windows':result['features']['hist_window_count'],
        'valid_historical_windows':result['features']['hist_valid_count']}))


def main(argv=None):
    parser=argparse.ArgumentParser(description='Monthly streamflow gap recoverability diagnosis')
    commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('reconstruct',help='Run a visible-only current gap and its historical pseudo-gaps')
    for name in ['flow','metadata','gap-start','gap-end','case-id','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--station-id',default='target');p.add_argument('--algorithm',choices=['idw','ridge','rf','extratrees','xgboost'],default='rf')
    p.add_argument('--seed',type=int,default=20261014);p.add_argument('--position',choices=['interior','leading','trailing'])
    p.add_argument('--event',choices=['original_event','seasonal_event'],default='original_event')
    p.add_argument('--input-preflight',action='store_true');p.add_argument('--model');p.add_argument('--trusted-local-pickle',action='store_true')
    p=commands.add_parser('fit',help='Fit a diagnostic map on TRAIN and choose policy on independent CAL')
    for name in ['features','labels','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--method',choices=['M','H'],default='M');p.add_argument('--event',choices=['original_event','seasonal_event'],default='original_event')
    p=commands.add_parser('diagnose',help='Apply a frozen model using visible historical features only')
    for name in ['features','model','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--trusted-local-pickle',action='store_true')
    p=commands.add_parser('cross-validate',help='Refit original-cohort models and score full registered spatial folds')
    for name in ['features','labels','registry','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--method',choices=['M','H'],default='M');p.add_argument('--event',choices=['original_event','seasonal_event'],default='original_event')
    p=commands.add_parser('evaluate',help='Evaluate saved decisions only after their persistence')
    for name in ['decisions','labels','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--bootstrap',type=int,default=1000)
    args=parser.parse_args(argv)
    if args.command=='reconstruct':return reconstruct_command(args)
    from .diagnosis import fit_diagnostic, diagnose, load_model, save_model
    if args.command=='fit':
        out=new_directory(args.output);model=fit_diagnostic(read_frame(args.features),read_frame(args.labels),args.method,args.event)
        save_model(out/('model.json' if args.method=='M' else 'model.pkl'),model)
        write_json(out/'fit_receipt.json',{'method':args.method,'event':args.event,**model['fit_audit'],'policy':model['policy']})
    elif args.command=='diagnose':
        model=load_model(args.model,trusted_pickle=args.trusted_local_pickle)
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        diagnose(read_frame(args.features),model).to_csv(args.output,index=False)
    elif args.command=='cross-validate':
        from .evaluation import cross_validate
        cross_validate(read_frame(args.features),read_frame(args.labels),read_frame(args.registry),args.output,method=args.method,event=args.event)
    elif args.command=='evaluate':
        from .evaluation import evaluate
        d=read_frame(args.decisions);labels=read_frame(args.labels);results=[]
        for (algorithm,seed),part in d.groupby(['algorithm','seed'],sort=True):
            lk=part[['case_id','algorithm','seed']];lab=lk.merge(labels,on=['case_id','algorithm','seed'],validate='one_to_one')
            for weight in ['station_equal','block_equal']:
                results.append({'algorithm':algorithm,'seed':int(seed),**evaluate(part,lab,weighting=weight,n_bootstrap=args.bootstrap)})
        write_json(args.output,results)

