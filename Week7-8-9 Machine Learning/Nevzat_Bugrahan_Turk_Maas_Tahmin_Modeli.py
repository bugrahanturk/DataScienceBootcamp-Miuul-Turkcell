import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import LocalOutlierFactor

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


def load():
    data = pd.read_csv("Week7/datasets/hitters.csv")
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


df = load()
check_df(df)


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


cat_cols, num_cols, cat_but_car = grab_col_names(df)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col, True)


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()


num_summary(df, num_cols, True)

# Sadece Target değişkeni için bakarsak
num_summary(df, "Salary", True)

#################################
# Missing Values (Eksik Değerler)
#################################

df.isnull().values.any()  # null değer olduğu gözüküyor
df.isnull().sum()  # Salary değişkenin de 59 adet

# Datayı incelediğimizde "CAtBat"(Oyuncunun kariyeri boyunca topa vurma sayısı) feature'nı diğer featurlar ile inceleyip sıfır olduğu yer var mı yok mu diye bakabiliriz
# çünkü bir oyunucunun örneğin  "CHits(Oyuncunun kariyeri boyunca yaptığı isabetli vuruş sayısı) sıfır değilken CAtBat sayısı sıfır ise bir sorun olduğunu anlarız.

# CAtBat sıfır olan değeri var mı kontrolü
df["CAtBat"].isnull().values.any()  # Sıfır değeri yok varsayımımız gerçekleşmedi.


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


na_cols = missing_values_table(df, True)  # Eksik değer oranı çok da düşük değil

#############################################
# Eksik Değer Problemini Çözme
#############################################

# sayısal değişkenleri outlierlardan daha az etkilendiği için direk median ile doldurma
df = df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)
df.isnull().sum()


#########################
# Aykırı değişken analizi
#########################

def outlier_thresholds(dataframe, col_name, q1=0.5, q3=0.95):
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
    print(col, check_outlier(df, col))

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))


# Aykırı değer görülmemiştir fakat Çok Değişkenli Aykırı Değer Analizini yapmakta fayda olacaktır
def local_outlier_factor(df):
    df_numeric = df.select_dtypes(include=['float64', 'int64'])

    df_numeric.head()
    df_numeric.shape

    clf = LocalOutlierFactor(n_neighbors=20)
    clf.fit_predict(df_numeric)

    df_scores = clf.negative_outlier_factor_
    df_scores[0:5]
    np.sort(df_scores)[0:5]

    scores = pd.DataFrame(np.sort(df_scores))
    scores.plot(stacked=True, xlim=[0, 50], style='.-')
    plt.show()

    elbow_variable = int(input("Degeri giriniz:"))
    th = np.sort(df_scores)[elbow_variable]

    print(df[df_scores < th])

    print(df[df_scores < th].shape)

    print(df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T)

    print(df[df_scores < th].index)

    df.drop(axis=0, labels=df[df_scores < th].index, inplace=True)

    return df


df = local_outlier_factor(df)

############################
# ÖZELLİK ÇIKARIMI
############################
# Mantıklı yeni değişkenler üretmek için Salary değişkeni ile Korelasyon analizine odaklanabiliriz

###############################
# Yeni değişkenler oluşturma
###############################
plt.figure(figsize=(16, 5))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f")
plt.show()
df.corr()

df["AVG_CAtBat"] = df["CAtBat"] / df["Years"]  # Oyuncunun kariyeri boyunca ortalama kaç kez topa vurduğu
df["AVG_CHits"] = df["CHits"] / df["Years"]
df["AVG_CHmRun"] = df["CHmRun"] / df["Years"]
df["AVG_Cruns"] = df["CRuns"] / df["Years"]
df["AVG_CRBI"] = df["CRBI"] / df["Years"]
df["AVG_CWalks"] = df["CWalks"] / df["Years"]
df["AVG_PutOuts"] = df["PutOuts"] / df["Years"]

# Oyuncunun vurduğu topların yüzde kaçı isabetli
df["PERC_CHits_CAtBat"] = df["CHits"] / df["CAtBat"] * 100

# Oyuncunun isabetli vuruşunun yüzde kaçı sayı olmuş
df["PERC_CRuns_CHits"] = df["CRuns"] / df["CHits"] * 100

# Oyuncunun karşı oyuncuya yaptırdığı hata sayısının yüzde kaçını takımına sayı olarak kazandırmış

df["PERC_CWalks_CRuns"] = df["CWalks"] / df["CRuns"] * 100

df.head()

#############################################
# Encoding
#############################################

# Değişkenlerin tiplerine göre ayrılması işlemi
cat_cols, num_cols, cat_but_car = grab_col_names(df)


# LABEL ENCODING
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
binary_cols
cat_cols
# categorical değişkenlerin hepsi binary column olduğundan dolayı sadece binary encoding yapmamız yeterli olacaktır.


for col in binary_cols:
    df = label_encoder(df, col)

df.head()

##################################
# STANDARTLAŞTIRMA
##################################
orig_df = df.copy()

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

##########################
# Model
##########################

y = df["Salary"]
X = df.drop(["Salary"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)


reg_model = LinearRegression().fit(X_train, y_train)

##########################
# Tahmin Başarısını Değerlendirme
##########################

# Train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))

# Test RMSE
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# TRAIN RKARE
reg_model.score(X_train, y_train)
# Test RKARE
reg_model.score(X_test, y_test)

# 10 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model, X, y, cv=10, scoring="neg_mean_squared_error")))

# sabit (b - bias)
reg_model.intercept_

# coefficients (w - weights)
reg_model.coef_

# Model datadaki ile null olan verileri  tahmin etme

df = load()
null_df = df[df["Salary"].isnull()]
null_df = null_df.drop(["Salary"],axis=1)
cat_cols, num_cols, cat_but_car = grab_col_names(null_df)

binary_cols = [col for col in null_df.columns if null_df[col].dtypes == "O" and null_df[col].nunique() == 2]
binary_cols
cat_cols
# categorical değişkenlerin hepsi binary column olduğundan dolayı sadece binary encoding yapmamız yeterli olacaktır.


for col in binary_cols:
    null_df = label_encoder(null_df, col)

null_df.head()

#corelasyon
null_df.corr()

null_df["AVG_CAtBat"] = null_df["CAtBat"] / null_df["Years"]
null_df["AVG_CHits"] = null_df["CHits"] / null_df["Years"]
null_df["AVG_CHmRun"] = null_df["CHmRun"] / null_df["Years"]
null_df["AVG_Cruns"] = null_df["CRuns"] / null_df["Years"]
null_df["AVG_CRBI"] = null_df["CRBI"] / null_df["Years"]
null_df["AVG_CWalks"] = null_df["CWalks"] / null_df["Years"]
null_df["AVG_PutOuts"] = null_df["PutOuts"] / null_df["Years"]

null_df["PERC_CHits_CAtBat"] = null_df["CHits"] / null_df["CAtBat"] * 100
null_df["PERC_CRuns_CHits"] = null_df["CRuns"] / null_df["CHits"] * 100
null_df["PERC_CWalks_CRuns"] = null_df["CWalks"] / null_df["CRuns"] * 100

null_df[num_cols] = scaler.fit_transform(null_df[num_cols])
y_pred = reg_model.predict(null_df)
