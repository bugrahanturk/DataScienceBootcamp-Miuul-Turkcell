############################################
# GÖREV 1: Veriyi Hazırlama
############################################


import pandas as pd

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

# Adım 1: Online RetailII veri setinden 2010-2011 sheet’ini okutunuz.
df_ = pd.read_excel("Week4/datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df.head()

df.describe().T
df.isnull().sum()
df.shape


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# Adım 2: StockCode’uPOST olan  gözlem birimlerini dropediniz. (POST her faturaya eklenen bedel, ürünü ifade etmemektedir.)
# Adım 3: Boş değer içeren gözlem birimlerini dropediniz.
# Adım 4: Invoiceiçerisinde C bulunan değerleri veri setinden çıkarınız. (C faturanın iptalini ifade etmektedir.)
def data_prep(dataframe):
    dataframe.drop(dataframe[dataframe["StockCode"] == "POST"].index, inplace=True)
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Price"] > 0]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    # Adım 6: Priceve Quantitydeğişkenlerinin aykırı değerlerini inceleyiniz, gerekirse baskılayınız
    replace_with_thresholds(dataframe, "Price")
    replace_with_thresholds(dataframe, "Quantity")
    return dataframe


# Adım 5: Pricedeğeri sıfırdan küçük olan gözlem birimlerini filtreleyiniz.
df[df["Price"] < 0]
df[df["Quantity"] < 0]

df = data_prep(df)
df.isnull().sum()
df.describe().T


############################################
# GÖREV 2: Alman Müşteriler Üzerinden Birliktelik Kuralları Üretme
############################################

def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)


def create_rules(dataframe, id=True, country="Germany"):
    dataframe = dataframe[dataframe['Country'] == country]
    dataframe = create_invoice_product_df(dataframe, id)
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules


def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)


df = df_.copy()

df = data_prep(df)
rules = create_rules(df)

check_id(df, 21987)
check_id(df, 23235)
check_id(df, 22747)


def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]


############################################
# GÖREV 2: Sepet İçerisindeki Ürün Id’leriVerilen Kullanıcılara Ürün Önerisinde Bulunma
############################################

recomended_item = arl_recommender(rules, 21987, 1)
check_id(df, recomended_item[0])

recomended_item = arl_recommender(rules, 23235, 1)
check_id(df, recomended_item[0])

recomended_item = arl_recommender(rules, 22747, 1)
check_id(df, recomended_item[0])
