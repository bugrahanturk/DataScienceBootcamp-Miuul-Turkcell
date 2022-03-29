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
    data = pd.read_csv("Week6/datasets/diabetes.csv")
    return data


df = load()


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


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()


num_summary(df, num_cols, True)

# Kategorik değişkenlere göre hedef değişkenin ortalaması, sadece outcome kategorik değişken

# hedef değişkene göre numerik değişkenlerin ortalaması
for col in num_cols:
    print(df.groupby("Outcome").agg({col: "mean"}), end="\n\n\n")


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
    print(col, check_outlier(df, col))  # Hepsinde aykırı değer olduğu görülüyor


# Aykırı değişkenlerin kendisine ulaşmak
def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


for col in num_cols:
    print("Outliers of " + col)
    print(grab_outliers(df, col), end="\n\n\n")

for col in num_cols:
    sns.boxplot(x=df[col])
    plt.show()

#################################
# Missing Values (Eksik Değerler)
#################################

# eksik gozlem var mı yok mu sorgusu
df.isnull().values.any()  # eksik değer yok fakat biz bazı featureların "Insulin,Glucose" sıfır olamayacağını bildiğimizden dolayı sıfır olan değerleri eksik değer olarak varsayacağız

df["Insulin"] = np.where(df["Insulin"] == 0, np.nan, df["Insulin"])
df["Glucose"] = np.where(df["Glucose"] == 0, np.nan, df["Glucose"])

df.isnull().values.any()

# Insulin değişkenindeki eksik gozlem sayısı
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


# Eksik Değerlerin Bağımlı Değişken ile İlişkisinin İncelenmesi

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"Outcome_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(df, "Outcome", na_cols)

#############################################
# Eksik Değer Problemini Çözme
#############################################

missing_values_table(df)


# Tahmine Dayalı Atama ile Doldurma


def fill_predicts(df):
    # değişkenlerin standartlatırılması
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    df.head()

    # knn'in uygulanması.
    from sklearn.impute import KNNImputer

    imputer = KNNImputer(n_neighbors=5)
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    df.head()

    df = pd.DataFrame(scaler.inverse_transform(df), columns=df.columns)

    return df


df = fill_predicts(df)
df.head()

df.isnull().values.any()
df.isnull().sum()


#############################################
# Aykırı Değer Problemini Çözme
#############################################

# Baskılama Yöntemi (re-assignment with thresholds)

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols

df.shape

for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))


# Çok Değişkenli Aykırı Değer Analizi: Local Outlier Factor

def local_outlier_factor(df):
    df = df.select_dtypes(include=['float64', 'int64'])
    df.head()
    df.shape

    clf = LocalOutlierFactor(n_neighbors=20)
    clf.fit_predict(df)

    df_scores = clf.negative_outlier_factor_
    df_scores[0:5]
    np.sort(df_scores)[0:5]

    scores = pd.DataFrame(np.sort(df_scores))
    scores.plot(stacked=True, xlim=[0, 50], style='.-')
    plt.show()

    th = np.sort(df_scores)[5]

    df[df_scores < th]  # Glukoz değeri sıfır olan değerler içeriyor

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

df.head()
df.describe().T

df['New_BMI'] = pd.cut(x=df['BMI'], bins=[0, 18.5, 24.9, 29.9, 51],
                       labels=["Underweight", "Healthy", "Overweight", "Obese"])

df.loc[df['Age'] <= 21, 'NEW_AGE_CAT'] = 'young'
df.loc[(df['Age'] > 21) & (df['Age'] < 50), 'NEW_AGE_CAT'] = 'mature'
df.loc[df['Age'] >= 50, 'NEW_AGE_CAT'] = 'senior'

df.loc[(df["New_BMI"] == "Underweight") & (df["NEW_AGE_CAT"] == "mature"), "New_Age_BMI"] = "Under_Weight_Mature"
df.loc[(df["New_BMI"] == "Underweight") & (df["NEW_AGE_CAT"] == "senior"), "New_Age_BMI"] = "Under_Weight_Senior"
df.loc[(df["New_BMI"] == "Healthy") & (df["NEW_AGE_CAT"] == "mature"), "New_Age_BMI"] = "Healthy_Mature"
df.loc[(df["New_BMI"] == "Healthy") & (df["NEW_AGE_CAT"] == "senior"), "New_Age_BMI"] = "Healthy_Senior"
df.loc[(df["New_BMI"] == "Overweight") & (df["NEW_AGE_CAT"] == "mature"), "New_Age_BMI"] = "Overweight_Mature"
df.loc[(df["New_BMI"] == "Overweight") & (df["NEW_AGE_CAT"] == "senior"), "New_Age_BMI"] = "Overweight_Senior"
df.loc[(df["New_BMI"] == "Obese") & (df["NEW_AGE_CAT"] == "mature"), "New_Age_BMI"] = "Obese_Mature"
df.loc[(df["New_BMI"] == "Obese") & (df["NEW_AGE_CAT"] == "senior"), "New_Age_BMI"] = "Obese_Senior"
df.loc[(df["New_BMI"] == "Underweight") & (df["NEW_AGE_CAT"] == "young"), "New_Age_BMI"] = "Under_Weight_Young"
df.loc[(df["New_BMI"] == "Healthy") & (df["NEW_AGE_CAT"] == "young"), "New_Age_BMI"] = "Healthy_Young"
df.loc[(df["New_BMI"] == "Overweight") & (df["NEW_AGE_CAT"] == "young"), "New_Age_BMI"] = "Overweight_Young"
df.loc[(df["New_BMI"] == "Obese") & (df["NEW_AGE_CAT"] == "young"), "New_Age_BMI"] = "Obese_Young"

cat_cols, num_cols, cat_but_car = grab_col_names(df)
cat_cols = [col for col in cat_cols if col != "Outcome"]
df.shape
df.head()

#############################################
# Encoding (Label Encoding, One-Hot Encoding, Rare Encoding)
#############################################

# Label Encoding & Binary Encoding

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]
binary_cols


# binary ve kategorik bir değişkenimiz olmadığından label encoding yapmıyoruz

# Rare Encoding
# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.

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
                            "Outcome_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


rare_analyser(df, "Outcome", cat_cols)


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


new_df = rare_encoder(df, 0.01)
new_df.head()
rare_analyser(new_df, "Outcome", cat_cols)

df = new_df


# One-Hot Encoding

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


ohe_cols = [col for col in df.columns if 12 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)
df.head()

cat_cols, num_cols, cat_but_car = grab_col_names(df)

rare_analyser(df, "Outcome", cat_cols)

useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis=None)]

df.drop(useless_cols, axis=1, inplace=True)

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

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

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
