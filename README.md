# Network
 Neural Network based product recommendation
 
 https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/8655105502604976/2769974586555098/7914230718189587/latest.html

 ```python
 from scipy.spatial.distance import cdist
 sub_arr = cdist(sira, sira, lambda u, v: u>=v)
 concern = (sub_arr*concern).sum(axis=0)
 fraud = (sub_arr*fraud).sum(axis=0)
 ```
%pip install optbinning pandas-profiling -U -q

import pandas as pd
import numpy as np

sdf = pd.concat([transformed_data, holding_matrix], axis=1)

from pandas_profiling import ProfileReport
profile = ProfileReport(sdf, title="Univariate Minimal Report", minimal=True)

profile.to_file("minimal_report.html")

import json
json_data = json.loads(profile.to_json())

keys = ['variable_name', 'n_distinct', 'p_distinct', 'is_unique', 'n_unique', 'p_unique', 'type', 'hashable', 'ordering', 'n_missing', 'n', 'p_missing', 'count', 
 'memory_size', 'n_negative', 'p_negative', 'n_infinite', 'n_zeros', 'mean', 'std', 'variance', 'min', 'max', 'kurtosis', 'skewness', 'sum', 
 'mad', 'range', '5%', '25%', '50%', '75%', '95%', 'iqr', 'cv', 'p_zeros', 'p_infinite', 'monotonic_increase', 'monotonic_decrease', 
 'monotonic_increase_strict', 'monotonic_decrease_strict', 'monotonic', 'top_ten_value_counts_without_nan', 'bottom_ten_value_counts_without_nan']

profile_dict = {k:[] for k in keys}
for i in json_data['variables'].keys():
    for j in keys:
        try:
            if j=='variable_name':
                profile_dict['variable_name'].append(i)
            elif j=='top_ten_value_counts_without_nan':
                top_ten_value_counts_without_nan = list(sdf[i].value_counts().iloc[:10].index.astype(str))
                profile_dict[j].append(top_ten_value_counts_without_nan)
            elif j=='bottom_ten_value_counts_without_nan':
                bottom_ten_value_counts_without_nan = list(sdf[i].value_counts().iloc[-10:].index.astype(str))
                profile_dict[j].append(bottom_ten_value_counts_without_nan)
            else:
                profile_dict[j].append(json_data['variables'][i][j])
        except:
            profile_dict[j].append(np.nan)

pd.DataFrame(profile_dict).to_excel('univariate summary.xlsx', index=False)
