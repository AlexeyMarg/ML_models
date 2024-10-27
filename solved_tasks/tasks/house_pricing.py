import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO
import os
import tarfile
from  six.moves import urllib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DOWNLOAD_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml/master/'
HOUSING_URL = DOWNLOAD_ROOT + 'datasets/housing/housing.tgz'
HOUSING_PATH = os.path.join('datasets', 'housing')
OUTPUT_PATH = os.path.join('output')

# download and unpack dataset
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, 'housing_tgz')
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def make_output_log(output_path=OUTPUT_PATH):
    if not os.path.isdir(output_path):
        os.makedirs(os.path.join(output_path, 'report'))
        os.makedirs(os.path.join(output_path, 'figs'))
    report_path = os.path.join(output_path, 'report', 'report.txt')
    figs_path = os.path.join(output_path, 'figs')
    return open(report_path, 'w'), figs_path

fetch_housing_data()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)

def show_all_dataframe():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
show_all_dataframe()

# write to report some statistics
housing = load_housing_data()
output_report, figs_path = make_output_log()
output_report.write('Imported dataframe head:\n\n')
output_report.write(str(housing.head()))
output_report.write('\n\nInfo about dataset:\n\n')
buffer = StringIO()
housing.info(buf=buffer)
s = buffer.getvalue()
output_report.write(s)
output_report.write('\n\nStatistics:\n\n')
output_report.write(str(housing.describe()))


housing.hist(bins=50, figsize=(20, 15))
plt.savefig(os.path.join(figs_path, 'Histogram'))
plt.close()
output_report.write('\n\nFeautures histograms in ' + os.path.join(figs_path, 'Histogram.png\n\n'))

# split for train and test
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# stratification of income
housing['income_cat'] = np.ceil(housing['median_income'] / 1.5)
housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)


housing['income_cat'].hist()
plt.savefig(os.path.join(figs_path, 'Income categories histogram'))
plt.close()
output_report.write('\n\nIncome categorization in ' + os.path.join(figs_path, 'Income categories histogram.png\n\n'))

# Stratified split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_traint_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

for set_ in (strat_traint_set, strat_test_set):
    set_.drop('income_cat', axis=1, inplace=True)

#geographic data
housing = strat_traint_set.copy()
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
             s=housing['population']/100, label='population',
             figsize=(10, 7), c='median_house_value',
             cmap=plt.get_cmap('jet'), colorbar=True)
plt.savefig(os.path.join(figs_path, 'Geographic density'))
plt.close()
output_report.write('\n\nGeographic density in ' + os.path.join(figs_path, 'Geographic density.png\n\n'))

# search of dependencies
corr_matrix = housing.corr()
output_report.write('Data \'median_house_value\' correlation:\n\n')
output_report.write(str(corr_matrix['median_house_value'].sort_values(ascending=False))+'\n\n')


attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
pd.plotting.scatter_matrix(housing[attributes], figsize=(12, 8))
plt.savefig(os.path.join(figs_path, 'Scatter matrix'))
output_report.write('Scatter matrix in ' + os.path.join(figs_path, 'Scatter matrix.png\n\n'))
plt.close()

# add some feautures
housing['rooms_per_household'] = housing['total_rooms'] / housing['households']
housing['bedrooms_per_room'] = housing['total_bedrooms'] / housing['total_rooms']
housing['population_per_household'] = housing['population'] / housing['households']

corr_matrix = housing.corr()
output_report.write('Data \'median_house_value\' correlation with added feutures:\n\n')
output_report.write(str(corr_matrix['median_house_value'].sort_values(ascending=False))+'\n\n')

# clear unknown data data
#housing = housing.dropna(subset=['total_bedrooms'])
#housing = housing.drop('total_bedrooms', axis=1)
#housing = housing.fillna(housing['total_bedrooms'].median(), inplace=True)
imputer = SimpleImputer(strategy='median')
housing_num = housing.drop('ocean_proximity', axis=1)
imputer.fit(housing_num)
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns = housing_num.columns)

# clear catigorial attribute
housing_cat = housing['ocean_proximity']
housing_cat_encoded, housing_categories = housing_cat.factorize()
encoder = OneHotEncoder(categories='auto')
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1)).toarray()
housing_cat_df = pd.DataFrame(housing_cat_1hot, columns=housing_categories)

# scaling
scaler = StandardScaler()
X = scaler.fit_transform(housing_tr)
housing_tr_scaled = pd.DataFrame(X, columns = housing_num.columns)

housing_prepared = pd.concat([housing_tr_scaled, housing_cat_df], axis=1)
output_report.write('Preprocessed dataset head:\n')
output_report.write(str(housing_prepared.head())+'\n')

#plt.show()
output_report.close()