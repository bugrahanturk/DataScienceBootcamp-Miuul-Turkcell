import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


def load(dataset):
    if dataset == "train":
        data = pd.read_csv("Week8/datasets/HousePrice/train.csv")
    elif dataset == "test":
        data = pd.read_csv("Week8/datasets/HousePrice/test.csv")
    else:
        return print("Enter valid dataset")
    return data


def check_df(dataframe, head=5, tail=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head ######################")
    print(dataframe.head(head))
    print("##################### Tail ######################")
    print(dataframe.tail(tail))
    print("##################### NA ########################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


train_df = load("train")
test_df = load("test")

check_df(train_df)
check_df(test_df)

train_df.drop(['Id'], axis=1, inplace=True)
test_df.drop(['Id'], axis=1, inplace=True)

y = train_df['SalePrice'].reset_index(drop=True)  # Datayı birleştirmek için tutmamız gerek
y.shape

train_df["SalePrice"].describe()

sns.distplot(train_df.SalePrice)
plt.show()

sns.boxplot(train_df["SalePrice"])
plt.show()

cat_cols, num_cols, cat_but_car = grab_col_names(train_df)


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()


for col in num_cols:
    num_summary(train_df, col, plot=False)


# Korelasyon Analizi
def target_correlation_matrix(dataframe, corr_th=0.5, target="SalePrice"):
    """
    Bağımlı değişken ile verilen threshold değerinin üzerindeki korelasyona sahip değişkenleri getirir.
    :param dataframe:
    :param corr_th: eşik değeri
    :param target:  bağımlı değişken ismi
    :return:
    """
    corr = dataframe.corr()
    corr_th = corr_th
    try:
        filter = np.abs(corr[target]) > corr_th
        corr_features = corr.columns[filter].tolist()
        sns.clustermap(dataframe[corr_features].corr(), annot=True, fmt=".2f")
        plt.show()
        return corr_features
    except:
        print("Yüksek threshold değeri, corr_th değerinizi düşürün!")


correlated_features = target_correlation_matrix(train_df, corr_th=0.5,
                                                target="SalePrice")  # salary price ile corr featurelar ileride feature oluşturmamız için gerekli
train_df.corr()

for col in cat_cols:
    if test_df[col].nunique() != train_df[col].nunique():
        print(f"{col} has not equal unique feature ")
        print(f"{col} \n unique elements train :{train_df[col].nunique()} test:{test_df[col].nunique()}\n")
# train ve test data setlerindeki kategorik değişkenlerin unique veri sayısı birbirinden farklı bu yüzden label encoding yaparken sıkıntı çıkmaması adına iki datasetini birleştirip data cleaning işlemini bu şekilde yapıyorum

# Features
train_features = train_df.drop(['SalePrice'], axis=1)
test_features = test_df

train_features.shape
test_features.shape

features = pd.concat([train_features, test_features]).reset_index(drop=True)
features.shape

cat_cols, num_cols, cat_but_car = grab_col_names(features)

#################################
# Missing Values (Eksik Değerler)
#################################

features.isnull().values.any()  # null değer olduğu gözüküyor
features.isnull().sum()


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


na_cols = missing_values_table(features, True)

# Alley, BsmtQual: BsmtCond: BsmtExposure: BsmtFinType1: BsmtFinType2: FireplaceQu: GarageType: GarageFinish: GarageQual: GarageCond: PoolQC: Fence: MiscFeature:
# datasetindeki bu değişkenlerin hepsi null olarak alınmış fakat bir ev bunlara sahip olmayabilir yani direk eksik veri diyemeyiz
# bu yüzden bu değişkenlerdeki nan değerlerinin %45dan fazla ratioya sahip olanları Na olarak güncelliyorum

cat_has_Na = ["Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "FireplaceQu",
              "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"]

for col in cat_has_Na:
    features[col] = np.where(
        (features[col].isnull().sum() / features[col].shape[0] * 100 > 45) & (features[col].isnull() == True), "Na",
        features[col])

na_cols = missing_values_table(features, True)

# geriye kalan eksik değerleri doldurma
# numeric değişkenleri median ile
features = features.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)
# kategorik değişkenleri mode ile doldurma
features = features.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 25) else x, axis=0)
features.isnull().values.any().sum()

cat_cols, num_cols, cat_but_car = grab_col_names(features)
features.shape
features.head()

###############################
# Yeni değişkenler oluşturma
###############################

correlated_features
# OverallQual = Evin genel malzemesini ve bitişini değerlendirir
# YearBuilt = Orijinal yapım tarihi
# YearRemodAdd = Tadilat tarihi (tadilat veya ilave yapılmamışsa inşaat tarihi ile aynı)
# TotalBsmtSF =  Bodrum alanının toplam metrekaresi
# 1stFlrSF = Birinci Kat metrekare
# GrLivArea = Üst sınıf (zemin) yaşam alanı metrekare
# FullBath = Bodrum tam banyolar
# TotRmsAbvGrd = Zemin üzerindeki toplam oda sayısı (banyo dahil değildir)
# GarageCars = Araba kapasitesindeki garajın büyüklüğü
# GarageArea = Garajın metrekare cinsinden büyüklüğü

features['NEWNESS'] = features['YearRemodAdd'] - features['YearBuilt']  # Evin yenilik durumu
features["GrLivAreaANDTotalBsmtSF"] = features["GrLivArea"] + features["TotalBsmtSF"]  #
features["TotRmsAbvGrdANDFullBath"] = features["TotRmsAbvGrd"] + features["FullBath"]
features["GarageCarsANDFullBath"] = features["GarageCars"] * features["FullBath"]
features["1stFlrSFANDGrLivArea"] = features["1stFlrSF"] + features["GrLivArea"]


# Rare Encoding

def rare_analyser(dataframe, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe)}), end="\n\n\n")


rare_analyser(features, cat_cols)


def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


features = rare_encoder(features, 0.01)
features.head()
features.shape
cat_cols, num_cols, cat_but_car = grab_col_names(features)


# (Label Encoding)

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in features.columns if features[col].dtype not in [int, float]
               and features[col].nunique() == 2]

for col in binary_cols:
    label_encoder(features, col)

cat_cols, num_cols, cat_but_car = grab_col_names(features)


# One Hot Encoder
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


ohe_cols = [col for col in features.columns if 25 >= features[col].nunique() > 2]

features = one_hot_encoder(features, ohe_cols)

cat_cols, num_cols, cat_but_car = grab_col_names(features)

features.head()
features.shape

train_df = features.iloc[:len(y), :]
test_df = features.iloc[len(y):, :]

train_df.head()
test_df.head()
test_df = test_df.reset_index(drop=True)

train_df.shape
test_df.shape
y.shape

train_df["SalePrice"] = y

#########################
# Aykırı değişken analizi
#########################

cat_cols, num_cols, cat_but_car = grab_col_names(train_df)


def outlier_thresholds(dataframe, col_name, q1=0.10, q3=0.90):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    print(col, check_outlier(train_df, col))

for col in num_cols:
    replace_with_thresholds(train_df, col)

for col in num_cols:
    print(col, check_outlier(train_df, col))

######## Scale #########
# Tree based model kullandığımız için aslında scaling yapmamıza gerek yok fakat yapsakta sonuçta bir etkisi olmayacaktır.

scaler = RobustScaler()
num_cols = [col for col in num_cols if col != "SalePrice"]
train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
train_df.columns
train_df.head()
##########################
# Model
##########################

y = train_df["SalePrice"]
X = train_df.drop(["SalePrice"], axis=1)

X.shape
y.shape

# LightGBM
from lightgbm import LGBMRegressor

lgbm_model = LGBMRegressor(random_state=17)

cv_results = cross_validate(lgbm_model, X, y, cv=5, scoring="neg_mean_squared_error")

cv_results["test_score"].mean()

np.mean(np.sqrt(-cross_val_score(lgbm_model, X, y, cv=5, scoring="neg_mean_squared_error")))

# Hiperparametre optimizasyonu
lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [500, 1000, 5000],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_best_grid.best_params_

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring="neg_mean_squared_error")

cv_results["test_score"].mean()

np.mean(np.sqrt(-cross_val_score(lgbm_final, X, y, cv=5, scoring="neg_mean_squared_error")))

# test data setini light gbm final modeli ile  tahminleme
pred_test_df = lgbm_final.predict(test_df)


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(20, 20))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

    return feature_imp.sort_values(by="Value", ascending=False)[0:num]["Feature"]


# En Önemli 80 feature ile tekrardan model kurma
important_features = plot_importance(lgbm_final, X, 80)
important_features = list(important_features)

X = train_df[important_features]

cv_results = cross_validate(lgbm_model, X, y, cv=5, scoring="neg_mean_squared_error")

# Skorlar
cv_results["test_score"].mean()

np.mean(np.sqrt(-cross_val_score(lgbm_model, X, y, cv=5, scoring="neg_mean_squared_error")))

# Hiperparametre optimizasyonu
lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [500, 1000, 5000],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
# lgbm_best_grid.best_score_
lgbm_best_grid.best_params_

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

# Skorlar
cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring="neg_mean_squared_error")

cv_results["test_score"].mean()

np.mean(np.sqrt(-cross_val_score(lgbm_final, X, y, cv=5, scoring="neg_mean_squared_error")))

# test data setini light gbm final modeli ile  tahminleme
pred_test_df = lgbm_final.predict(test_df[important_features])
