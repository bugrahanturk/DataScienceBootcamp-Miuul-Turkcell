import numpy as np
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

def load():
    data = pd.read_csv("Week9/dataset/flo_data_20K.csv")
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

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

def high_correlated_cols(dataframe, plot=False, corr_th=0.50):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
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

############
# Data Prep
############

df = load()
df.head()

check_df(df)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in cat_cols:
    num_summary(df, col, plot=True)

for col in num_cols:
    num_summary(df, col, plot=False)

# Korelasyon analizi
high_correlated_cols(df,plot=False)


corr = df[num_cols].corr()
corr

sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()

# Aykırı değer analizi
for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))


date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

df.isnull().values.any() # Eksik değer bulunmuyor
df.isnull().sum()

analysis_date = dt.datetime(2021,6,1)

df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

df["recency"] = (analysis_date - df["last_order_date"]).astype('timedelta64[D]')
df.rename(columns={"order_num_total": "frequency","customer_value_total": "monetary"}, inplace=True)

df["recency_score"] = pd.qcut(df['recency'], 5, labels=[5, 4, 3, 2, 1])
df["frequency_score"] = pd.qcut(df['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
df["monetary_score"] = pd.qcut(df['monetary'], 5, labels=[1, 2, 3, 4, 5])

df["RF_SCORE"] = (df['recency_score'].astype(int) + df['frequency_score'].astype(int))
df["RFM_SCORE"] = (df['recency_score'].astype(int) + df['frequency_score'].astype(int) + df['monetary_score'].astype(int))

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

df['segment'] = df['RF_SCORE'].replace(seg_map, regex=True)

df.head()
df.info()
# date columns çıkarma
date_cols = [col for col in df.columns if is_datetime(df[col])]

df.drop(["master_id","interested_in_categories_12"], axis=1, inplace=True)
df.drop(date_cols, axis=1, inplace=True)
df.head()
df.info()

cat_cols, num_cols, cat_but_car = grab_col_names(df)

#Encoding
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

df.head()
df.info()

################################
# K-Means
################################
sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)

kmeans = KMeans()
ssd = []
K = range(1, 30)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(df)
    ssd.append(kmeans.inertia_)

plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSR/SSD")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show()

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df)
elbow.show()

elbow.elbow_value_

################################
# Final Cluster'ların Oluşturulması
################################

kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df)

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_

clusters_kmeans = kmeans.labels_

k_means_clustered_df = load()

k_means_clustered_df["cluster"] = clusters_kmeans

k_means_clustered_df.head()

k_means_clustered_df["cluster"] = k_means_clustered_df["cluster"] + 1

k_means_clustered_df.groupby("cluster").agg(["count","mean","median"])

k_means_clustered_df.to_csv("clusters.csv")

################################
# Hierarchical Clustering
################################

hc_average = linkage(df, "average")

plt.figure(figsize=(7, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           truncate_mode="lastp", # 'lastp' ->p tane küme kalana kadar böl , none denirse en son gözlem birimine kadar böler ve anlamsız gözükür plot
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.show()

################################
# Kume Sayısını Belirlemek
################################

plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_average)
plt.axhline(y=3.5, color='r', linestyle='--')
plt.show()

# Cluster sayısını grafik incelenerek 8 olarak seçilebilir

################################
# Final Modeli Oluşturmak
################################

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=8, linkage="average")

clusters = cluster.fit_predict(df)

hi_data = load()

hi_data["hi_cluster_no"] = clusters

hi_data["hi_cluster_no"] = hi_data["hi_cluster_no"] + 1

hi_data.head()

# Adım 3: Her bir segmenti istatistiksel olarak inceleyeniz
hi_data.groupby("hi_cluster_no").agg(["count","mean","median"])
# son hal
last_data = load()

last_data["hi_cluster_no"] = hi_data["hi_cluster_no"]
last_data["kmeans_cluster"] = k_means_clustered_df["cluster"]
last_data.head()
last_data.to_csv()