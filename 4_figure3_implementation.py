
# Table Implementation

import numpy as np
import pandas as pd
from pandas import read_csv

file2open = 'sigma_03.csv'
data = read_csv(file2open)
data_01 = data.values
# print(data_01.shape,type(data_01))
data_02 = pd.DataFrame(data_01)
#修改列名的方法！！！！
data_02.columns = ['sigma1', 'sigma2']
data_02.to_csv('sigma_05.csv', index=True)

data_03 = data_01[-10::]
data_04 = pd.DataFrame(data_03)
data_04.columns = ['sigma1', 'sigma2']
data_04.to_csv('sigma_06.csv', index=True)
