"""Frozen conventional learners; numerical fit bodies retained from the experiment."""
import time
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from .inputs import finish_prediction, predict_idw, ridge_fit

DEFAULT_ALGORITHMS=('idw','ridge','rf','extratrees','xgboost')



PARAMETERS={'ridge':{'alpha':10.},
    'rf':{'n_estimators':256,'min_samples_leaf':3,'max_features':1.,'n_jobs':1},
    'extratrees':{'n_estimators':256,'min_samples_leaf':3,'max_features':1.,'n_jobs':1},
    'xgboost':{'n_estimators':300,'max_depth':3,'learning_rate':.03,'subsample':.8,
        'colsample_bytree':.8,'reg_lambda':1.,'tree_method':'hist','n_jobs':1}}



def fit_classical(prepared,algorithm,seed):
    if algorithm not in PARAMETERS:raise ValueError('Unknown learner: '+algorithm)
    if prepared['fallback'] is not None:
        result=finish_prediction(None,prepared)
        result['summary'].update(algorithm=algorithm,seed=int(seed),deterministic_seed_reuse=algorithm=='ridge')
        return result
    x=prepared['X_flat'];observed=prepared['observed_mask'];y=prepared['y_standardized'][observed]
    started=time.perf_counter()
    try:
        if algorithm=='ridge':
            # Double precision solve; exactly the common input values, no new predictors.
            beta=ridge_fit(np.asarray(x[observed],float),y,alpha=PARAMETERS['ridge']['alpha'])
            predictions=beta[0]+np.asarray(x,float)@beta[1:]
        else:
            if algorithm=='rf':model=RandomForestRegressor(**PARAMETERS[algorithm],random_state=int(seed))
            elif algorithm=='extratrees':model=ExtraTreesRegressor(**PARAMETERS[algorithm],random_state=int(seed))
            else:
                from xgboost import XGBRegressor
                model=XGBRegressor(**PARAMETERS[algorithm],random_state=int(seed),objective='reg:squarederror',verbosity=0)
            model.fit(x[observed],y);predictions=model.predict(x)
        result=finish_prediction(predictions,prepared)
        result['summary'].update(algorithm=algorithm,seed=int(seed),parameters=PARAMETERS[algorithm],
            fit_seconds=time.perf_counter()-started,fit_failed=False,deterministic_seed_reuse=algorithm=='ridge')
    except Exception as exc:
        result={'prediction_query':np.full(len(prepared['query_indices']),np.nan),
            'prediction_all':np.full(len(observed),np.nan),'summary':{'model_fitted':False,
                'fit_failed':True,'error_type':type(exc).__name__,'error':str(exc),
                'algorithm':algorithm,'seed':int(seed),'parameters':PARAMETERS[algorithm],
                'fit_seconds':time.perf_counter()-started}}
    return result



def reconstruct(prepared, algorithm='rf', seed=20261014):
    if algorithm == 'idw':
        result = predict_idw(prepared)
        return {'prediction_query': result['idw'], 'summary': {
            **result['summary'], 'algorithm': algorithm, 'seed': int(seed),
            'model_fitted': False, 'fit_failed': False}}
    return fit_classical(prepared, algorithm, int(seed))


def reconstruct_optional_neural(prepared, algorithm, seed=20261014, device='cpu'):
    """Supplementary local neural learners with the frozen TRAIN-pilot settings.

    Imported lazily; neither torch nor a GPU is needed for the main methods.
    The original experiment used CUDA. CPU runs implement the same architecture
    and optimizer but need not be bitwise equal across numerical backends.
    """
    if algorithm not in ('lstm','tcn'):raise ValueError('Optional algorithms: lstm or tcn.')
    if prepared['fallback'] is not None:return finish_prediction(None,prepared)
    from .optional_neural import fit_prepared_batch
    config={'hidden_size':32 if algorithm=='lstm' else 16,'epochs':50,
        'learning_rate':.01,'weight_decay':1e-4,'gradient_clip':5.,'sequence_length':12,'input_features':28}
    fit=fit_prepared_batch([prepared],algorithm,int(seed),config=config,device=device)[0]
    result=finish_prediction(fit['prediction_all_std'],prepared)
    result['summary'].update(algorithm=algorithm,seed=int(seed),neural_training=fit['training_summary'])
    return result
