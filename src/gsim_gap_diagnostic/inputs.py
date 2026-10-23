"""Pure NumPy visible preprocessing shared by all seven reconstruction families.

No archive, outcome table, fitted checkpoint, or external attributes are read.
The 12-month windows preserve calendar positions, including missing months.
"""
from __future__ import annotations
import numpy as np

VERSION = 'shared-visible-calendar12-28channels-20261014-v1'
SEEDS = (20261014, 20261015, 20261016)
WINDOW = 12
SLOTS = 8
CHANNELS = 28
ALGORITHMS = ('idw', 'ridge', 'rf', 'extratrees', 'xgboost', 'lstm', 'tcn')


def harmonics(months):
    angle = 2*np.pi*np.asarray(months, float)/12
    return np.column_stack([np.sin(angle), np.cos(angle), np.sin(2*angle), np.cos(2*angle)])


def ridge_fit(x, y, alpha=10.):
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(design.shape[1])*alpha; penalty[0,0] = 0.
    return np.linalg.solve(design.T@design+penalty, design.T@y)


def seasonal_prediction(values, observed, month_features):
    if not observed.any(): return np.full(len(values), np.nan)
    if observed.sum() < 5: return np.full(len(values), float(np.mean(values[observed])))
    beta = ridge_fit(month_features[observed], values[observed])
    return beta[0] + month_features@beta[1:]


def select_donors(y, donors):
    """Exact legacy positive-correlation, >=6 paired months, top8 rule."""
    common = np.isfinite(y)[:,None] & np.isfinite(donors)
    count = common.sum(axis=0)
    xx = np.where(common, donors, 0.); yy = np.where(common, y[:,None], 0.)
    xc = np.where(common, xx-xx.sum(axis=0)/np.maximum(count,1), 0.)
    yc = np.where(common, yy-yy.sum(axis=0)/np.maximum(count,1), 0.)
    denominator = np.sqrt((xc*xc).sum(axis=0)*(yc*yc).sum(axis=0))
    corr = np.divide((xc*yc).sum(axis=0), denominator, out=np.zeros_like(denominator), where=denominator>1e-12)
    eligible = np.flatnonzero((count>=6)&(corr>0))
    return eligible[np.argsort(-corr[eligible], kind='stable')[:SLOTS]]


def prepare_visible(y_visible, donors_visible, query_indices, months, distances_km, case_id):
    y = np.asarray(y_visible, float).copy(); d = np.asarray(donors_visible, float).copy()
    raw_query = np.asarray(query_indices); query = np.asarray(query_indices, int)
    months = np.asarray(months); distances = np.asarray(distances_km, float)
    if y.ndim != 1 or d.ndim != 2 or d.shape[0] != len(y) or d.shape[1]>100 or months.shape != y.shape:
        raise ValueError('Expected target/months T and donor array T by at most100.')
    if len(y)<WINDOW or query.ndim!=1 or not len(query) or not np.array_equal(query,raw_query) or query[0]<0 or query[-1]>=len(y) or np.any(np.diff(query)!=1):
        raise ValueError('Query must be contiguous integer indices within a >=12-month calendar.')
    if not np.isnan(y[query]).all(): raise ValueError('Every query target must already be NaN.')
    if np.isinf(y).any() or np.isinf(d).any(): raise ValueError('Only finite values or NaN are supported.')
    if not np.isfinite(months).all() or np.any(months!=months.astype(int)) or np.any((months<0)|(months>11)):
        raise ValueError('months must be calendar integers0..11.')
    if np.any((np.diff(months.astype(int)) % 12) != 1): raise ValueError('Do not compress missing calendar months.')
    if distances.shape!=(d.shape[1],) or not np.isfinite(distances).all() or np.any(distances<0):
        raise ValueError('Each candidate needs its nonnegative geographic distance in km.')
    if not isinstance(case_id,str) or not case_id: raise ValueError('Stable case_id required.')
    observed = np.isfinite(y); n = int(observed.sum()); h = harmonics(months)
    y_mean = float(y[observed].mean()) if n else 0.
    raw_y_scale = float(y[observed].std()) if n else 0.
    y_scale = raw_y_scale if raw_y_scale>1e-12 else 1.
    standardized = np.full(len(y), np.nan); standardized[observed]=(y[observed]-y_mean)/y_scale
    seasonal = seasonal_prediction(y, observed, h)
    fallback = 'empty_target' if n==0 else 'too_few_target_months' if n<12 else 'constant_target' if raw_y_scale**2<=1e-12 else None
    selected = select_donors(y,d)
    monthly = np.zeros((len(y),CHANNELS),float); monthly[:,:4]=h
    monthly[:,np.arange(SLOTS)*3+6]=1.
    donor_values = np.full((len(y),SLOTS),np.nan)
    active, calibrations = [], []
    for candidate in selected:
        paired = observed & np.isfinite(d[:,candidate]); values = d[:,candidate]
        mean = float(values[paired].mean()); scale = float(values[paired].std())
        if paired.sum()<6 or scale<=max(1e-10,abs(mean)*1e-8): continue
        slot = len(active); season = seasonal_prediction(values,paired,h)
        anomaly = values-season
        anomaly_scale=max(float(np.sqrt(np.mean(anomaly[paired]**2))),scale*1e-6,1e-10)
        missing=~np.isfinite(values)
        monthly[:,4+slot*3]=(np.where(missing,season,values)-mean)/scale
        monthly[:,5+slot*3]=np.where(missing,0.,anomaly/anomaly_scale)
        monthly[:,6+slot*3]=missing.astype(float)
        donor_values[:,slot]=values
        active.append(int(candidate))
        calibrations.append({'slot':slot,'candidate_index':int(candidate),'paired_months':int(paired.sum()),
            'donor_mean':mean,'donor_scale':scale,'donor_anomaly_scale':anomaly_scale,
            'target_paired_mean':float(y[paired].mean()),'target_paired_scale':float(y[paired].std()),
            'distance_km':float(distances[candidate])})
    padded = np.zeros((WINDOW-1,CHANNELS),float)
    padded[:,:4]=harmonics((int(months[0])+np.arange(-WINDOW+1,0))%12)
    padded[:,np.arange(SLOTS)*3+6]=1.
    continuous=np.concatenate([padded,monthly])
    sequence=np.stack([continuous[t:t+WINDOW] for t in range(len(y))]).astype(np.float32)
    assert sequence.shape==(len(y),WINDOW,CHANNELS) and np.isfinite(sequence).all()
    active_n=len(active)
    coverage=np.isfinite(donor_values[:,:active_n]).mean(axis=1) if active_n else np.zeros(len(y))
    summary={'n_target':n,'n_paired_months':int(np.sum(observed&(coverage>0))),
        'n_donors':active_n,'donor_gap_coverage':float(coverage[query].mean()),
        'diagnostic_available':bool(n>=6 and raw_y_scale**2>1e-12),
        'target_scale':raw_y_scale,'target_scaling_fit_months':n,'selected_candidate_count':len(selected),
        'feature_count_per_month':CHANNELS,'sequence_length':WINDOW,'flattened_feature_count':WINDOW*CHANNELS,
        'fallback':fallback}
    return {'X_sequence':sequence,'X_flat':sequence.reshape(len(y),-1), 'X_monthly':monthly.astype(np.float32),
        'y_standardized':standardized,'observed_mask':observed,'query_indices':query.copy(),
        'y_visible':y,'y_mean':y_mean,'y_scale':y_scale,'months':months.astype(int),
        'selected_indices':selected,'active_indices':np.asarray(active,int),'donor_values':donor_values,
        'donor_calibrations':calibrations,'summary':summary,'fallback':fallback,
        'fallback_prediction_all':np.maximum(seasonal,0.),'case_id':case_id,
        'provenance':{'version':VERSION,'calendar_compacted':False,'current_target_hidden':True,
            'donor_selection_recomputed_under_current_mask':True,'preprocessing_uses_visible_target_paired_months_only':True,
            'target_values_in_predictors':False,'calendar_window':'previous11 and current month',
            'left_padding':'raw/anomaly zeros, missing flags one, calendar harmonics continued',
            'donor_slots':'descending visible target correlation; unused slots raw/anomaly0 and missing1'}}


def residual_memory(residual, observed):
    adjacent=observed[:-1]&observed[1:]; pairs=int(adjacent.sum())
    if pairs>=3:
        previous=residual[:-1][adjacent];following=residual[1:][adjacent]
        denominator=float(previous@previous)
        coefficient=float(previous@following)/denominator if denominator>1e-12 else 0.
        phi=float(np.clip(coefficient*pairs/(pairs+5),-.95,.95))
    else:phi=0.
    correction=np.zeros(len(residual)); known=np.flatnonzero(observed)
    for t in np.flatnonzero(~observed):
        insertion=np.searchsorted(known,t)
        left=int(known[insertion-1]) if insertion else None
        right=int(known[insertion]) if insertion<len(known) else None
        if left is not None and right is not None:
            a,b=phi**(t-left),phi**(right-t);link=a*b
            correction[t]=((a-b*link)*residual[left]+(b-a*link)*residual[right])/max(1-link**2,1e-12)
        elif left is not None:correction[t]=phi**(t-left)*residual[left]
        elif right is not None:correction[t]=phi**(right-t)*residual[right]
    return correction,{'phi':phi,'adjacent_visible_pairs':pairs,'residuals_from_this_models_visible_training_fit':True}


def finish_prediction(pred_all_std, prepared):
    """All six learners receive the same rule, using their own fit residuals."""
    query=prepared['query_indices']
    if prepared['fallback'] is not None:
        prediction=prepared['fallback_prediction_all'].copy()
        return {'prediction_all':prediction,'prediction_query':prediction[query],
                'summary':{'model_fitted':False,'fallback':prepared['fallback'],'memory_applied':False}}
    pred=np.asarray(pred_all_std,float).reshape(-1)
    if pred.shape!=prepared['y_visible'].shape:raise ValueError('Model must predict all T calendar rows.')
    if not np.isfinite(pred).all():raise ValueError('Nonfinite learner predictions must be reported as a model failure.')
    raw=pred*prepared['y_scale']+prepared['y_mean']
    correction,memory=residual_memory(prepared['y_visible']-raw,prepared['observed_mask'])
    corrected=raw+correction
    return {'prediction_all':np.maximum(corrected,0.),'prediction_query':np.maximum(corrected[query],0.),
            'summary':{'model_fitted':True,'fallback':None,'memory_applied':True,'memory':memory,
                       'negative_query_predictions_before_clip':int((corrected[query]<0).sum())}}


def predict_idw(prepared):
    """Geographic IDW over shared active donors, calibrated and raw auxiliaries.

    Exact colocated candidates share all weight. Missing donor months are never
    filled for IDW. A month with no eligible observation remains unavailable.
    """
    values=prepared['donor_values'];cal=prepared['donor_calibrations'];query=prepared['query_indices']
    calibrated=np.full(len(values),np.nan);raw=calibrated.copy()
    for t in range(len(values)):
        usable=[i for i in range(len(cal)) if np.isfinite(values[t,i])]
        if not usable:continue
        zero=[i for i in usable if cal[i]['distance_km']==0.]
        if zero:usable=zero;weights=np.ones(len(usable))
        else:
            distance=np.array([cal[i]['distance_km'] for i in usable]);weights=(distance.min()/distance)**2
        v=np.array([values[t,i] for i in usable])
        mapped=np.array([cal[i]['target_paired_mean']+cal[i]['target_paired_scale']*
                         (values[t,i]-cal[i]['donor_mean'])/cal[i]['donor_scale'] for i in usable])
        raw[t]=np.average(v,weights=weights);calibrated[t]=np.average(mapped,weights=weights)
    return {'idw':np.maximum(calibrated[query],0.),'raw_idw':np.maximum(raw[query],0.),
        'season':prepared['fallback_prediction_all'][query].copy(),
        'summary':{'deterministic':True,'memory_applied':False,'geographic_inverse_distance_power':2,
            'shared_mask_selected_donors':True,'calibration':'pair-specific visible common-period target/donor means and standard deviations',
            'no_donor_query_months':int((~np.isfinite(calibrated[query])).sum()),'zero_distance':'equal weight within observed colocated group',
            'missing_observations_imputed':False}}
