"""Visible historical-score algorithm selection, before current labels are joined."""
import numpy as np
import pandas as pd
from .diagnosis import validate_visible, eligible

ALGORITHM_ORDER=['idw','ridge','rf','extratrees','xgboost']


def select_by_history(features, algorithms=ALGORITHM_ORDER, name='select5_histNSE'):
    """Exact fixed priority for ties, RF fallback with diagnostic abstention.

    The selected system needs its own TRAIN map and CAL policy; taking the
    maximum of separately calibrated probabilities is not equivalent.
    """
    validate_visible(features)
    if features.seed.nunique()!=1:raise ValueError('Select within one seed.')
    p=features.loc[features.algorithm.isin(algorithms)].copy()
    if p.groupby('case_id').algorithm.nunique().ne(len(algorithms)).any():
        raise ValueError('Every case must retain every candidate algorithm.')
    p['_eligible']=eligible(p,p.hist_NSE_median.to_numpy(float))
    p['_rank']=p.algorithm.map({a:i for i,a in enumerate(algorithms)})
    chosen=p.loc[p._eligible].sort_values(['case_id','hist_NSE_median','_rank'],ascending=[True,False,True]).drop_duplicates('case_id')
    no=set(p.case_id)-set(chosen.case_id)
    fallback=p.loc[p.case_id.isin(no)&p.algorithm.eq('rf')].copy()
    fallback['diagnostic_available']=False;fallback['hist_valid_count']=0
    for c in ['hist_R_median','hist_NSE_median','hist_joint_rate','hist_R_iqr','hist_NSE_iqr']:
        if c in fallback:fallback[c]=np.nan
    result=pd.concat([chosen,fallback],ignore_index=True).drop(columns=['_eligible','_rank'])
    result['selected_algorithm']=result.algorithm;result['algorithm']=name
    return result

