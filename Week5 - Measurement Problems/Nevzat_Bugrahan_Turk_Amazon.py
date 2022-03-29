import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

############################################
# GÖREV 1: Average Rating'i güncel yorumlara göre hesaplayınız ve var olan average rating ile kıyaslayınız.
############################################

# Data Preprocessing
df = pd.read_csv("Week5/datasets/amazon_review.csv")
df.head(15)
df.describe().T
df.isnull().sum()

# Adım 1: Ürünün ortalama puanı
df["overall"].mean()

# Adım 2: Tarihe Göre Ağırlıklı Puan Ortalamasını Hesaplayınız

df.info()
# reviewTime değişkenini tarih değişkeni olarak tanıtma
df["reviewTime"] = pd.to_datetime(df["reviewTime"])

# reviewTime'ın max değerini current_date olarak kabul etme
current_date = df["reviewTime"].max()

# her bir puan-yorum tarihi ile current_date'in farkını gün cinsinden ifade ederek yeni değişken oluşturma
df["days"] = (current_date - df["reviewTime"]).dt.days

# Gün cinsinden ifade edilen değişkeni quantile fonksiyonu ile 4'e bölüp (3 çeyrek verilirse 4 parça çıkar) çeyrekliklerden gelen değerlere göre ağırlıklandırma yapma
quantiles = df["days"].quantile([.25, .50, .75, ])
# df["day_diff"].quantile([.25, .50, .75])

quantiles = quantiles.tolist()
q1 = quantiles[0]
q2 = quantiles[1]
q3 = quantiles[2]

df.loc[df["days"] <= q1, "overall"].mean()
df.loc[(df["days"] > q1) & (df["days"] <= q2), "overall"].mean()
df.loc[(df["days"] > q2) & (df["days"] <= q3), "overall"].mean()
df.loc[(df["days"] > q3), "overall"].mean()


# Adım3 : time_based_weighted_average fonksiyonunu kullanarak, day_diff'i gün sayısına göre veya quartile değerlerine göre parçalayıp ağırlıklandırınız ve her bir zaman diliminin ortalamasını karşılaştırıp yorumlayınız
def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[df["days"] <= q1, "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["days"] > q1) & (dataframe["days"] <= q2), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["days"] > q2) & (dataframe["days"] <= q3), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["days"] > q3), "overall"].mean() * w4 / 100


w1, w2, w3, w4 = 28, 26, 24, 22

time_based_weighted_average(df)
# Adım4: Ağırlandırılmış puanlamada her bir zaman diliminin ortalamasını karşılaştırıp yorumlayınız:
df.loc[df["days"] <= q1, "overall"].mean() * w1 / 100
df.loc[(df["days"] > q1) & (df["days"] <= q2), "overall"].mean() * w2 / 100
df.loc[(df["days"] > q2) & (df["days"] <= q3), "overall"].mean() * w3 / 100
df.loc[(df["days"] > q3), "overall"].mean() * w4 / 100
# verilen ağırlık değerine göre puanlamaların önemi,değeri de değişmektedir.

############################################
# GÖREV 2: Ürün için ürün detay sayfasında görüntülenecek 20 review'i belirleyiniz
############################################

# Adım1: helpful_no değişkenini üretme
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]


# Adım2: score_pos_neg_diff, score_average_rating ve wilson_lower_bound skorlarını hesapalyıp veriye ekleme
def score_pos_neg_diff(pos, neg):
    return pos - neg


df["score_pos_neg_diff"] = df.apply(lambda x: score_pos_neg_diff(x["helpful_yes"], x["helpful_no"]), axis=1)
df.head(5)


def score_average_rating(pos, neg):
    if pos + neg == 0:
        return 0
    return pos / (pos + neg)


df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)
df.head(5)


def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)
df.head(5)

# Adım3: 20 yorumu belirleyiniz ve sonuçlarını yorumlayınız
df.sort_values("wilson_lower_bound", ascending=False).head(20)
df.sort_values("wilson_lower_bound", ascending=False).head(20)["reviewText"]  # yorumlar

# Yorum:
# Sonuçlara baktığımızda kullanıcıların faydalı bulduğu yorumların olması gerektiği gibi yukarda olduğu gözlemlenmektedir.
