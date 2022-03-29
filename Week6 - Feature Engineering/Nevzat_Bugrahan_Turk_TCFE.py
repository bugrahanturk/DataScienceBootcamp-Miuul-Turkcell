import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


def load():
    data = pd.read_csv("Week6/datasets/Telco-Customer-Churn.csv")
    return data


df = load()
df.head()
df.describe()


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


check_df(df)
# Total Charges(Müşteriden tahsil edilen toplam tutar) değişkeni float olması gerekirken veri içerisindeki dtype'ı object olarak gözüküyor.
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')


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

df[num_cols].describe().T
df[cat_cols].describe().T


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

# Hedef değişkene göre numerik değişkenlerin ortalaması
for col in num_cols:
    print(df.groupby("Churn").agg({col: "mean"}), end="\n\n\n")


#########################
# Aykırı değişken analizi
#########################


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
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


for col in num_cols:
    print(col, check_outlier(df, col))  # Outlier yok gözüküyor

#################################
# Missing Values (Eksik Değerler)
#################################

# eksik gozlem var mı yok mu sorgusu

df.isnull().values.any()

df.isnull().sum()


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


na_cols = missing_values_table(df, True)
# Sadece TotalCharges değişkeni eksik değer içeriyor

# Eksik Veri Yapısının İncelenmesi

msno.bar(df)
plt.show()

msno.matrix(df)
plt.show()

msno.heatmap(df)
plt.show()

# Korelasyon analizi
sns.heatmap(df[num_cols + cat_cols + cat_but_car].corr(), annot=True, fmt=".2f")
plt.show()
df.corr()

#############################################
# Eksik Değer Problemini Çözme
#############################################

# değişken sayısı az olduğu için data setinden drop edebiliriz
df = df.dropna()
df.shape
df.isnull().sum()


#############################################
# Aykırı Değer Problemini Çözme
#############################################
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

    th = np.sort(df_scores)[17]

    df[df_scores < th]

    df[df_scores < th].shape

    df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T

    df[df_scores < th].index

    df.drop(axis=0, labels=df[df_scores < th].index, inplace=True)

    return df


df = local_outlier_factor(df)

df.shape

###############################
# Yeni değişkenler oluşturunuz
###############################

df.loc[(df["SeniorCitizen"] == 1) & (df["Dependents"] == "Yes"), "New_Citizen_Dependent"] = "Senior_Dependent"
df.loc[(df["SeniorCitizen"] == 1) & (df["Dependents"] == "No"), "New_Citizen_Dependent"] = "Senior_Independent"
df.loc[(df["SeniorCitizen"] == 0) & (df["Dependents"] == "Yes"), "New_Citizen_Dependent"] = "NotSenior_Dependent"
df.loc[(df["SeniorCitizen"] == 0) & (df["Dependents"] == "No"), "New_Citizen_Dependent"] = "NotSenior_Independent"

df.loc[(df["PhoneService"] == "Yes") & (df["InternetService"] == "DSL"), "New_Phone_Internet"] = "Phone_DSL"
df.loc[(df["PhoneService"] == "Yes") & (df["InternetService"] == "Fiber optic"), "New_Phone_Internet"] = "Phone_Fiber"
df.loc[(df["PhoneService"] == "Yes") & (df["InternetService"] == "No"), "New_Phone_Internet"] = "Phone_NoIn"
df.loc[(df["PhoneService"] == "No") & (df["InternetService"] == "DSL"), "New_Phone_Internet"] = "NoPhone_DSL"
df.loc[(df["PhoneService"] == "No") & (df["InternetService"] == "Fiber"), "New_Phone_Internet"] = "NoPhone_Fiber"
df.loc[(df["PhoneService"] == "No") & (df["InternetService"] == "No"), "New_Phone_Internet"] = "NoPhone_NoIn"

df.head()

#############################################
# Encoding (Label Encoding, One-Hot Encoding, Rare Encoding)
#############################################

# Label Encoding & Binary Encoding

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]
binary_cols


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


for col in binary_cols:
    label_encoder(df, col)

df.head()

# Rare Encoding
# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.

cat_cols, num_cols, cat_but_car = grab_col_names(df)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col)


# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "Churn_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


rare_analyser(df, "Churn",
              cat_cols)  # veri incelendiğinde ration parametresi 0.01 tutulduğunda rare olarak kullanılabilecek bir değişkenin olmadığı gözlemlenmektedir dolayısıyla ratio ile ufak bir oynama yapılabilir.


# 3. Rare encoder'ın yazılması.

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


df = rare_encoder(df, 0.014)
df.head()
rare_analyser(df, "Churn", cat_cols)


# One-Hot Encoding

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


ohe_cols = [col for col in df.columns if 12 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)
df.head()

cat_cols, num_cols, cat_but_car = grab_col_names(df)

rare_analyser(df, "Churn", cat_cols)

useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis=None)]

# Kullanışsız bir değişken bulunmamaktadır
# df.drop(useless_cols, axis=1, inplace=True)

df.groupby("New_Phone_Internet_Phone_Fiber").agg({"Churn": "mean"})
df.groupby("New_Phone_Internet_Phone_DSL").agg({"Churn": "mean"})
df.groupby("New_Phone_Internet_Phone_NoIn").agg({"Churn": "mean"})

from statsmodels.stats.proportion import proportions_ztest

test_stat, pvalue = proportions_ztest(count=[df.loc[df["New_Phone_Internet_Phone_Fiber"] == 1, "Churn"].sum(),
                                             df.loc[df["New_Phone_Internet_Phone_Fiber"] == 0, "Churn"].sum()],

                                      nobs=[df.loc[df["New_Phone_Internet_Phone_Fiber"] == 1, "Churn"].shape[0],
                                            df.loc[df["New_Phone_Internet_Phone_Fiber"] == 0, "Churn"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))  # Test Stat = 26.0206, p-value = 0.0000
# Test sonucundan da anlaşılacağı üzere, proportion_ztestin hipotezi p1 ve p2 oranları arasında fark yoktur der fakat görüldüğü üzere p value 0.05den küçük olduğundan dolayı h0 reddedilir ve telefon servisi olup interneti de fiber olanların churn etme oranı daha düşüktür diyebiliriz.

##########################################
# Standart Scaler
##########################################

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()

df.head()
df.shape

#############################################
# Model
#############################################

y = df["Churn"]
X = df.drop(df[["Churn", "customerID"]], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X_train)
