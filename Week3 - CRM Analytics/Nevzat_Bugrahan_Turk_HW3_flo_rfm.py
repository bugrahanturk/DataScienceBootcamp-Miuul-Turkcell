import datetime as dt
import pandas as pd

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

#############################################
# Görev 1: Veriyi Anlama ve Hazırlama
#############################################

df_ = pd.read_csv(r"C:\Users\bugra\OneDrive\Masaüstü\turkcell_ds_bootcamp\Week3\datasets\flo_data_20K.csv")
df = df_.copy()
df.shape

df.head(10)
df.columns
df.dropna(inplace=True)

df.describe().T  # betimsel
df.isnull().sum()
df.dtypes

df["num_total_omnichannel"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["value_total_omnichannel"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

for col in df.columns:
    if "date" in col:
        df[col] = pd.to_datetime(df[col])

df.dtypes

df.groupby('order_channel').agg({'master_id': "count",
                                 'num_total_omnichannel': "mean",
                                 'value_total_omnichannel': "mean"})

df["value_total_omnichannel"].sort_values(ascending=False).head(10)

df["num_total_omnichannel"].sort_values(ascending=False).head(10)


def create_pre_process(data):
    data = df
    df.head(10)
    df.columns
    df.dropna(inplace=True)

    df.describe().T  # betimsel
    df.isnull().sum()
    df.dtypes

    df["num_total_omnichannel"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df["value_total_omnichannel"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

    for col in df.columns:
        if "date" in col:
            df[col] = pd.to_datetime(df[col])
    df.dtypes
    df.groupby('order_channel').agg({'master_id': "count",
                                     'num_total_omnichannel': "mean",
                                     'value_total_omnichannel': "mean"})

    df["value_total_omnichannel"].sort_values(ascending=False).head(10)

    df["num_total_omnichannel"].sort_values(ascending=False).head(10)


create_pre_process(df)

#############################################
# Görev 2: RFM Metriklerinin Hesaplanması
#############################################

today_date = dt.datetime(2022, 1, 6)
type(today_date)

# recency = müşterinin yeniliği
# frequency = müsterinin yaptığı toplam satın alma
# monetary = müsterinin yaptığı toplam satın alma neticesinde müşterinin bıraktığı toplam parasal değer

rfm = df.groupby('master_id').agg({'last_order_date': lambda date: (today_date - date.max()).days,
                                   'value_total_omnichannel': "sum",
                                   "num_total_omnichannel": "sum"})

rfm.head()

rfm.columns = ['recency', 'frequency', 'monetary']

rfm.describe().T

rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])

# 0-100, 0-20, 20-40, 40-60, 60-80, 80-100

rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

rfm["monetary_score"] = pd.qcut(rfm['monetary'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

rfm["RF_SCORE"] = (rfm['recency_score'].astype(str) +
                   rfm['frequency_score'].astype(str))

rfm.describe().T

#############################################
# RF Skorunun Hesaplanması
#############################################


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

rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)

new_df = pd.merge(df, rfm, on="master_id")

df_a = new_df[(new_df["segment"] == "champions") | (new_df["segment"] == "loyal_customers")
              & (new_df["monetary"].mean() > 250)
              & (new_df["interested_in_categories_12"].str.contains("KADIN"))].index

new_df_a = pd.DataFrame()
new_df_a["case_a"] = df_a

new_df_a.to_csv("new_df_a.csv")

df_b = new_df[(new_df["segment"] == "about_to_sleep") | (new_df["segment"] == "new_customers")
                & (new_df["interested_in_categories_12"].str.contains("ERKEK") )
                & (new_df["interested_in_categories_12"].str.contains("COCUK"))].index

new_df_b = pd.DataFrame()
new_df_b["case_b"] = df_b

new_df_b.to_csv("new_df_b.csv")

