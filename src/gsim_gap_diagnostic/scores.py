"""Original whole-gap scoring and historical summary semantics."""
import numpy as np

def score_truth(truth,prediction):
    truth=np.asarray(truth,float);prediction=np.asarray(prediction,float)
    target_finite=bool(np.isfinite(truth).all());prediction_finite=bool(np.isfinite(prediction).all())
    finite=target_finite and prediction_finite
    variance=float(np.var(truth)) if target_finite else np.nan
    pred_variance=float(np.var(prediction)) if prediction_finite else np.nan
    if not target_finite:status='target_truth_incomplete'
    elif not prediction_finite:status='nonfinite_prediction'
    elif not np.isfinite(variance) or variance<=1e-12:status='target_variance_insufficient'
    elif not np.isfinite(pred_variance) or pred_variance<=1e-12:status='constant_prediction'
    else:status='evaluable'
    evaluable=status=='evaluable'
    r=float(np.corrcoef(truth,prediction)[0,1]) if evaluable else np.nan
    nse=1-float(np.mean((truth-prediction)**2))/variance if finite and variance>1e-12 else np.nan
    return {'R':r,'NSE':nse,'evaluable':evaluable,'model_available':prediction_finite,
        'pass_R':int(r>=.7) if evaluable else np.nan,'pass_joint':int(r>=.7 and nse>=.5) if evaluable else np.nan,
        'evaluation_status':status,'target_variance':variance,'prediction_variance':pred_variance,
        'R_defined':bool(np.isfinite(r)),'NSE_defined':bool(np.isfinite(nse))}



def history_summary(records):
    valid=[r for r in records if r['evaluable']]
    result={'hist_window_count':len(records),'hist_valid_count':len(valid),
        'hist_distance_mean':float(np.mean([r['distance'] for r in records])) if records else np.nan}
    for metric in ('R','NSE'):
        result[f'hist_{metric}_median']=float(np.median([r[metric] for r in valid])) if valid else np.nan
        result[f'hist_{metric}_iqr']=float(np.diff(np.quantile([r[metric] for r in valid],[.25,.75]))[0]) if len(valid)>1 else np.nan
    result['hist_joint_rate']=float(np.mean([r['pass_joint'] for r in valid])) if valid else np.nan
    return result



def climatology_prediction(y_visible, months, query):
    """Calendar-month means fitted only to visible target observations."""
    y = np.asarray(y_visible, float); months = np.asarray(months, int)
    climatology = {m: np.mean(y[(months == m) & np.isfinite(y)])
                   for m in np.unique(months) if np.any((months == m) & np.isfinite(y))}
    return np.asarray([climatology.get(m, np.nan) for m in months[query]])


def event_score(truth, prediction, baseline=None, event='original_event'):
    result = score_truth(truth, prediction)
    if event not in ('original_event', 'seasonal_event'):
        raise ValueError('Unknown registered event.')
    if event == 'seasonal_event':
        truth = np.asarray(truth, float); baseline = np.asarray(baseline, float)
        baseline_sse = float(np.sum((truth-baseline)**2)) if (
            baseline.shape == truth.shape and np.isfinite(truth).all() and np.isfinite(baseline).all()) else np.nan
        reference_known = np.isfinite(baseline_sse) and baseline_sse > 1e-12
        known = result['evaluable'] and reference_known
        skill = 1-float(np.sum((truth-np.asarray(prediction, float))**2))/baseline_sse if known else np.nan
        result.update(evaluable=bool(known), baseline_sse=baseline_sse, climatology_skill=skill,
            pass_joint=int(result['pass_joint'] == 1 and skill > 0) if known else np.nan)
    return result
