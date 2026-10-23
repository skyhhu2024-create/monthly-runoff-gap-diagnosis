"""Paired information-reduction experiment with fixed identity-only mask ordering.

The target_conditional arm removes target values on exactly the same dates as
the overlap arm masks donors. Its conditional comparison is against overlap.
"""
import hashlib
import numpy as np

MASK_SALT='R7-nested-visible-masks-20261014-v1'


def token(*parts):
    return hashlib.sha256('|'.join(map(str,parts)).encode('utf8')).hexdigest()


def scenarios():
    result=[dict(scenario='reference',condition='reference',retained_fraction=1.,mask_replicate=-1)]
    for condition in ['overlap','target_conditional','gap_donors']:
        for fraction in [.5,.25]:
            for replicate in [0,1]:
                result.append(dict(scenario=f'{condition}_r{int(fraction*100)}_m{replicate}',condition=condition,
                                   retained_fraction=fraction,mask_replicate=replicate))
    return result


def date_order(case_id,y,q,replicate):
    protected=np.zeros(len(y),bool);protected[max(0,int(q[0])-11):int(q[-1])+1]=True
    before=np.flatnonzero(np.isfinite(y[:q[0]]))
    after=np.flatnonzero(np.isfinite(y[q[-1]+1:]))+q[-1]+1
    if len(before):protected[before[-1]]=True
    if len(after):protected[after[0]]=True
    available=np.flatnonzero(np.isfinite(y)&~protected)
    length=len(q);offset=int(token(MASK_SALT,case_id,replicate,'offset'),16)%length
    block=(available+offset)//length
    block_order=sorted(np.unique(block),key=lambda b:token(MASK_SALT,case_id,replicate,'block',int(b)))
    order=np.concatenate([available[block==b] for b in block_order]) if len(block_order) else available
    if len(order)<int(np.floor(.75*np.isfinite(y).sum())):
        raise ValueError('Target support is insufficient for the published protected-date reduction rule.')
    return order,{'protected_dates':np.flatnonzero(protected),'calendar_block_offset':offset,
                  'block_length':length,'eligible_removal_dates':available,'nested_date_order':order}


def apply_reduction(y_visible,donors,query,*,case_id,donor_ids,baseline_active_indices,
                    condition='reference',retained_fraction=1.,mask_replicate=-1):
    y=np.asarray(y_visible,float);d=np.asarray(donors,float);q=np.asarray(query,int)
    if not np.isnan(y[q]).all():raise ValueError('Current target query must already be hidden.')
    if len(donor_ids)!=d.shape[1]:raise ValueError('Physical donor IDs must match candidate columns.')
    yy=y.copy();dd=d.copy();details={'condition':condition,'retained_fraction':retained_fraction,'mask_replicate':mask_replicate}
    fraction=float(retained_fraction);replicate=int(mask_replicate)
    if condition!='reference' and (fraction not in (.5,.25) or replicate not in (0,1)):
        raise ValueError('Published reductions use fractions .5/.25 and mask replicates0/1.')
    if condition in ('overlap','target_conditional'):
        order,detail=date_order(case_id,y,q,replicate);n=int(np.floor((1-fraction)*np.isfinite(y).sum()))
        removed=order[:n];dd[removed,:]=np.nan
        if condition=='target_conditional':yy[removed]=np.nan
        details.update(detail,removed_dates=removed,requested_removed_target_dates=n,
                       actual_removed_target_dates=int(np.isfinite(y).sum()-np.isfinite(yy).sum()))
    elif condition=='gap_donors':
        active=np.asarray(baseline_active_indices,int)
        ordered=sorted(active.tolist(),key=lambda candidate:token(MASK_SALT,case_id,replicate,'donor',str(donor_ids[candidate])))
        n=max(1,int(np.ceil(len(active)*fraction))) if len(active) else 0
        retained=np.asarray(ordered[:n],int);drop=np.ones(d.shape[1],bool);drop[retained]=False
        dd[np.ix_(q,np.flatnonzero(drop))]=np.nan
        details.update(baseline_active_candidate_indices=active,retained_candidate_indices=retained,
            retained_physical_donor_ids=[str(donor_ids[c]) for c in retained],baseline_active_count=len(active),retained_active_count=n)
    elif condition!='reference':raise ValueError(condition)
    assert np.isnan(yy[q]).all()
    assert np.all(~np.isfinite(yy)|np.isfinite(y)) and np.all(~np.isfinite(dd)|np.isfinite(d))
    return yy,dd,details

