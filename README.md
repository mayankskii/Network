# Network
 Neural Network based product recommendation
 
 https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/8655105502604976/2769974586555098/7914230718189587/latest.html


 ```python
 from scipy.spatial.distance import cdist
 sub_arr = cdist(sira, sira, lambda u, v: u>=v)
 concern = (sub_arr*concern).sum(axis=0)
 fraud = (sub_arr*fraud).sum(axis=0)
 ```
