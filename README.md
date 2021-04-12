# Network
 Neural Network based product recommendation


def descriptive_summary(df, path=None):
    """Generate descriptive statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe for which statistics are calculated.
    path: str, default=None
        Path in which descriptive_summary excel is exported with formatting.

    Returns
    -------
    dict
    df : pd.DataFrame
        Dataframe for which statistics are calculated.
    path: str, default=None
        
    Note
    ----
    In the desc output following are the suggested values.
    
    dtype
    nan count
    nan rate
    fill rate
    closed range
    range length
    IQR
    mean_skew
    mode skew
    median skew
    quantile skew
    SNR dB
    kurtosis

    """

    # describe_pd
    desc = df.describe(include='all', percentiles=[.01, .25, .5, .75, .99]).T
    desc['mode'] = df.mode(axis=0).T
    desc['dtype'] = df.dtypes

    # describe_counts
    desc['nan count'] = df.isna().sum()
    desc['nan rate'] = desc['nan count']/df.shape[0]
    desc['fill rate'] = desc['count']/df.shape[0]

    # 1D statistics: Skew and Kurtosis
    desc['closed range'] = list(zip(desc['min'], desc['max']))
    desc['range length'] = desc['max']-desc['min']
    desc['IQR'] = desc['75%'] - desc['25%']
    desc = desc.join(pd.Series(df.skew(axis=0), name='mean_skew'))
    desc['mode skew'] = desc.apply(lambda x: (x['mean'] - x['mode'])/x['std'] if (
        isinstance(x['mode'], (int, float, bool)) & (x['std'] != 0)) else np.nan, axis=1)
    desc['median skew'] = desc.apply(lambda x: (
        x['mean'] - x['50%'])/x['std'] if x['std'] != 0 else np.nan, axis=1)
    desc['quantile skew'] = desc.apply(lambda x: (x['75%'] - 2*x['50%'] + x['25%'])/(
        x['75%'] - x['25%']) if (x['75%'] - x['25%']) != 0 else np.nan, axis=1)
    desc['SNR dB'] = desc.apply(lambda x: (x['mean']/x['std'])**2 if (
        isinstance(x['mode'], (int, float, bool)) & (x['std'] != 0)) else np.nan, axis=1)
    desc = desc.join(pd.Series(df.kurtosis(axis=0), name='kurtosis'))

    desc = desc.reset_index().rename(columns={'index': 'feature_name'})
    corr = df.corr().reset_index().rename(columns={'index': 'corr'})

    if path is None:
        pass
    else:
        try:
            writer = pd.ExcelWriter(path, engine='xlsxwriter')
            desc.to_excel(writer, sheet_name='summary', index=False)
            corr.to_excel(writer, sheet_name='corr', index=False)
            workbook = writer.book
            worksheet = writer.sheets['summary']
            percentage_fmt = workbook.add_format({'num_format': '0.00%'})
            worksheet.set_column('R:S', None, percentage_fmt)
            writer.save()
        except:
            pass
    return {'desc': desc, 'corr': corr}
