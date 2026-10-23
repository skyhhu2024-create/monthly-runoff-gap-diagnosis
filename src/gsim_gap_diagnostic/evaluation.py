"""Post-prediction evaluation retaining every registered gap in denominators."""
import numpy as np
import pandas as pd
from .diagnosis import KEYS
from ._history_model import valid_labels, as_bool


def evaluate(decisions, labels, *, weighting='station_equal', n_bootstrap=1000, seed=20261014):
    if decisions.duplicated(KEYS).any() or labels.duplicated(KEYS).any(): raise ValueError('Duplicate prediction/label identities.')
    if set(map(tuple,decisions[KEYS].to_numpy())) != set(map(tuple,labels[KEYS].to_numpy())):
        raise ValueError('Evaluation must retain exactly the same registered case identities.')
    wanted=KEYS+['R','NSE','pass_joint','evaluable','model_available']
    f=decisions.merge(labels[wanted],on=KEYS,validate='one_to_one')
    if f.algorithm.nunique()!=1 or f.seed.nunique()!=1:
        raise ValueError('Evaluate one algorithm and seed at a time.')
    known=valid_labels(f);success=known&f.pass_joint.eq(1).to_numpy();a=as_bool(f.adopted)
    blocks,codes=np.unique(f.block.to_numpy(str),return_inverse=True)
    if weighting=='station_equal':w=np.ones(len(f))
    elif weighting=='block_equal':w=1/np.bincount(codes)[codes]
    else:raise ValueError('Unknown weighting.')
    mat=np.column_stack([np.ones(len(f)),known,success,a,a&success,a&known])*w[:,None]
    agg=np.zeros((len(blocks),mat.shape[1]));np.add.at(agg,codes,mat)
    def div(x,y):return np.divide(x,y,out=np.full(np.shape(x),np.nan,dtype=float),where=y>0)
    def rates(t):return {'precision':div(t[...,4],t[...,3]),'recall':div(t[...,4],t[...,2]),
        'coverage':div(t[...,3],t[...,0]),'baseline_success':div(t[...,2],t[...,0])}
    result={k:float(v) for k,v in rates(agg.sum(0)).items()}
    result.update(weighting=weighting,n_registered=len(f),n_known=int(known.sum()),n_success=int(success.sum()),
        n_adopted=int(a.sum()),n_adopted_success=int((a&success).sum()),n_adopted_unknown=int((a&~known).sum()),n_blocks=len(blocks))
    if n_bootstrap and len(blocks)>1:
        draws=np.random.default_rng(seed).multinomial(len(blocks),np.full(len(blocks),1/len(blocks)),size=n_bootstrap)
        for k,values in rates(draws@agg).items():
            finite=values[np.isfinite(values)]
            result[k+'_ci95']=np.quantile(finite,[.025,.975]).tolist() if len(finite) else [None,None]
        result['bootstrap_draws']=n_bootstrap
    return result


def cross_validate(features, labels, registry, output, *, method='M', event='original_event'):
    """Original five-fold rotation; fit original cohort only, predict full registry.

    Persist all out-of-fold predictions before performing any TEST-label join.
    Registered original block IDs and test folds are used unchanged.
    """
    from .diagnosis import fit_diagnostic, diagnose, save_model
    from .spatial import roles_for_fold
    from ._io import new_directory, write_json
    out=new_directory(output);registry=registry[['case_id','station_id','block','cv_test_fold','source_population']].copy()
    if registry.case_id.duplicated().any() or registry.station_id.duplicated().any(): raise ValueError('Registry must have one case per station.')
    if not set(registry.cv_test_fold).issubset(set(range(5))): raise ValueError('Expected frozen fold IDs 0..4.')
    if registry.groupby('block').cv_test_fold.nunique().gt(1).any(): raise ValueError('A spatial block cannot span folds.')
    identity=features[['case_id','station_id','block']].drop_duplicates()
    check=identity.merge(registry,on='case_id',suffixes=('_features',''),validate='one_to_one')
    if len(check)!=len(registry) or not check.station_id_features.eq(check.station_id).all() or not check.block_features.eq(check.block).all():
        raise ValueError('Feature identity differs from the complete registry.')
    extra=['cv_test_fold','source_population'];f=features.drop(columns=[c for c in extra if c in features]).merge(registry[['case_id']+extra],on='case_id',validate='many_to_one')
    predictions=[];audits=[]
    for (algorithm,seed),part in f.groupby(['algorithm','seed'],sort=True):
        if set(part.case_id)!=set(registry.case_id): raise ValueError('Every algorithm must retain every registered gap.')
        for fold in range(5):
            roles=roles_for_fold(part.cv_test_fold,fold)
            development=part.loc[(roles!='test')&part.source_population.eq('old3262')].copy()
            development['split']=roles_for_fold(development.cv_test_fold,fold)
            te=part.loc[roles=='test'].copy()
            if set(development.block)&set(te.block): raise ValueError('Development/TEST block overlap.')
            lk=development[KEYS];lab=lk.merge(labels,on=KEYS,validate='one_to_one')
            model=fit_diagnostic(development,lab,method,event)
            model['outer_fold']=fold
            filename=f'model_{method}_{event}_{algorithm}_seed{seed}_fold{fold}'+('.json' if method=='M' else '.pkl')
            save_model(out/filename,model)
            pred=diagnose(te,model);pred['outer_fold']=fold;predictions.append(pred)
            audits.append({'algorithm':algorithm,'seed':int(seed),'outer_fold':fold,**model['fit_audit']})
    predictions=pd.concat(predictions,ignore_index=True)
    predictions.to_csv(out/'predictions_before_evaluation.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    write_json(out/'prediction_receipt.json',{'status':'complete','rows':len(predictions),'test_labels_joined_to_predictions':False,'fit_audits':audits})
    metrics=[]
    for (algorithm,seed),part in predictions.groupby(['algorithm','seed'],sort=True):
        lk=part[KEYS];lab=lk.merge(labels,on=KEYS,validate='one_to_one')
        for weight in ['station_equal','block_equal']:
            metrics.append({'algorithm':algorithm,'seed':int(seed),**evaluate(part,lab,weighting=weight)})
    write_json(out/'metrics.json',metrics)
    return predictions

