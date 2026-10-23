"""Published input preflight and supervision-support controls.

These are experimental contrasts. Baseline reconstruction and diagnosis retain
their original inputs unless the caller explicitly selects a control.
"""
import numpy as np
from . import windows as sel
from .replay import prepare_history
from .inputs import finish_prediction, ridge_fit
from .reconstruction import PARAMETERS, RandomForestRegressor, ExtraTreesRegressor


def candidates(y,d,q,months,position):
    length=len(q);n=int(np.isfinite(y).sum());pair=sel._mean_paired(y,d);cov=sel._coverage(d,q)
    complete=np.convolve(np.isfinite(y).astype(int),np.ones(length,dtype=int),mode='valid')==length
    out=[]
    for start in np.flatnonzero(complete):
        start=int(start);iy,idd,hq=sel.make_replay(y,d,start,q,position);inner_n=int(np.isfinite(iy).sum())
        if inner_n<12:continue
        if position=='interior' and not(np.isfinite(iy[:start]).any() and np.isfinite(iy[start+length:]).any()):continue
        ip=sel._mean_paired(iy,idd);ic=sel._coverage(idd,hq);md=abs(int(months[start])-int(months[q[0]]))%12
        components={'season':float(min(md,12-md)/6.),'target_support':sel._relative_difference(inner_n,n),'paired_support':sel._relative_difference(ip,pair),'query_donor_coverage':abs(ic-cov)}
        out.append({'start':start,'end_exclusive':start+length,'distance':float(np.mean(list(components.values()))),'distance_components':components,'inner_n_target':inner_n,'inner_mean_paired_months':ip,'replay_gap_donor_coverage':ic})
    return out


def input_preflight_windows(y,d,q,months,distances,position,case_id):
    """Original candidate order/budget; require donor inputs for every gap month."""
    cand = candidates(y,d,q,months,position)
    ranked = sorted(cand,key=lambda z:(z['distance'],sel._hash(case_id,'matched-tie',z['start'])))
    orig = sel._greedy(ranked,4)
    random = sel._greedy(sorted(cand,key=lambda z:sel._hash(case_id,'unmatched',z['start'])),4)
    orig = orig[:min(len(orig),len(random))]
    chosen=[]; prepared={}; pool_rejected=0; active_rejected=0
    for z in ranked:
        if len(chosen)>=len(orig):break
        if any(not(z['end_exclusive']<=v['start'] or z['start']>=v['end_exclusive']) for v in chosen):continue
        hq=np.arange(z['start'],z['end_exclusive'])
        if not np.all(np.any(np.isfinite(d[hq])&np.isfinite(d[q]),axis=1)):
            pool_rejected+=1;continue
        pp,_=prepare_history(y,d,q,z['start'],months,distances,position,case_id+f":history:{z['start']}")
        if not np.isfinite(pp['donor_values'][hq]).any(axis=1).all():active_rejected+=1;continue
        chosen.append(z);prepared[z['start']]=pp
    return chosen, prepared, {'enabled':True,'budget':len(orig),'candidate_count':len(cand),
        'pool_rejected':pool_rejected,'active_rejected':active_rejected,
        'original_selected_starts':[z['start'] for z in orig],
        'new_selected_starts':[z['start'] for z in chosen]}


def fit_supported(p,alg,seed=20261014):
    """Original learner supervision control; fitted scaling and memory unchanged."""
    support=np.isfinite(p['donor_values']).any(axis=1);training=p['observed_mask']&support
    count=int(training.sum());n0=int(p['observed_mask'].sum())
    if p['fallback'] is not None:
        z=finish_prediction(None,p)
    elif count<12:
        pp=dict(p);pp['fallback']='too_few_supported_supervision_months';z=finish_prediction(None,pp)
    else:
        x=p['X_flat'];y=p['y_standardized'][training]
        if alg=='ridge':
            beta=ridge_fit(np.asarray(x[training],float),y,alpha=PARAMETERS[alg]['alpha']);prediction=beta[0]+np.asarray(x,float)@beta[1:]
        else:
            if alg=='rf':model=RandomForestRegressor(**PARAMETERS[alg],random_state=seed)
            elif alg=='extratrees':model=ExtraTreesRegressor(**PARAMETERS[alg],random_state=seed)
            elif alg=='xgboost':
                from xgboost import XGBRegressor
                model=XGBRegressor(**PARAMETERS[alg],random_state=seed,objective='reg:squarederror',verbosity=0)
            else:raise ValueError('Supported learners: ridge, rf, extratrees, xgboost.')
            model.fit(x[training],y);prediction=model.predict(x)
        z=finish_prediction(prediction,p)
    z['summary'].update(supervision_original_months=n0,supervision_supported_months=count,
        supervision_removed_months=n0-count,algorithm=alg,seed=seed,input_recipe_unchanged=True,
        residual_memory_mask_unchanged=True)
    return z

