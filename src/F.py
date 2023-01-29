import pandas as pd
from scipy import stats

def f_two_way(df: pd.DataFrame, factor1: str, factor2: str, target: str):
    """
    両側検定を仮定 
    
    -----------------
        df: DataFrame
            因子２つとそれに対応するt_columnが入ったデータフレーム
            下記構造であることが期待される。

            *****************************
            |factor1|factor2|target|
            -----------------
            |a1|b2|t1|
            |a2|b2|t2|

            |an|b2|t3|
            *****************************
            
        factor1: str
            因子１とするカラム名

        factor2: str
            因子2とするカラム名

        target: str
            
            
    exception
    ---------
        [factor1, factor2]ごとの試行回数が２以上であり、かつ均一

    return
    ------
        {
            "interraction": {"F:", "df","Prob": }
            target1: {"F:", "df","Prob": }
            target2: {"F:", "df","Prob": }
        }
    """
    
    
    assert target in df, f'{target} is not found'
    assert factor1 in df, f'{factor1} is not found'
    assert factor2 in df, f'{factor2} is not found'
    assert df.groupby(by = [factor1, factor2]).count()[target].nunique() == 1, f"This function assumes that samples are the same among each parameter specification"

    a = df[factor1].nunique()
    b = df[factor2].nunique()
    r = df.groupby(by = [factor1, factor2]).count()[target][0]
    n = a*b*r

    assert r >= 2, f"number of samples per parameter must be more than 2"
    assert n == df.shape[0], f"something is wrong, n = {n} and number of samples is {df.shape[0]}"

    ct = (df[target].sum()**2)/n
    
    ## 平方和
    S_T = (df[target]**2).sum() -ct
    S_A_OR_B = (df.groupby(by = [factor1, factor2])[target].sum()**2).sum()/r - ct
    SE = S_T - S_A_OR_B
    SA = (df.groupby(by = factor1)[target].sum()**2).sum()/(b*r) - ct
    SB = (df.groupby(by = factor2)[target].sum()**2).sum()/(a*r) - ct
    S_A_and_B = S_A_OR_B - SA - SB
    
    ## 平均平方 (不偏分散)
    VA = SA / (a-1)
    VB = SB / (b-1)
    V_A_and_B = S_A_and_B/((a-1)*(b-1))
    VE = SE/(a*b*(r-1))

    ## F値 (上から、交互作用がないと仮定したときのF値, Aが有効でないと仮定したときのの)
    fval_interaction = V_A_and_B / VE
    fval_1 = VA / VE
    fval_2 = VB / VE

    fs = [fval_interaction, fval_1, fval_2]
    labels = ["interfaction", factor1, factor2]
    degrees = [(a-1)*(b-1), a-1, b-1]
    ps = [ min(stats.f.cdf(f, degree, a*b*(r-1)), stats.f.sf(f, degree, a*b*(r-1))) * 2 for f, degree in zip(fs, degrees)]
    
    
    
    ret = {}
    
    for l,f,d,p in zip(labels,fs,degrees,ps):
        ret[l] = {}
        ret[l]["prob"] = p
        ret[l]["f"] = f
        ret[l]["degree"] = (d,a*b*(r-1))

    return ret

