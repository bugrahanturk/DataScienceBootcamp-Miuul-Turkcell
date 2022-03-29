import pandas as pd
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

############################################
# GÖREV 1: Veriyi Hazırlama ve Analiz Etme
############################################

control_df = pd.read_excel("Week5/datasets/ab_testing.xlsx", sheet_name="Control Group")
control_df = control_df.dropna(how="all", axis="columns")

# Adım1: kontrol ve test grubu verilerini ayrı değişkenlere atma
# Adım2: Verileri analiz etme

control_df = control_df.rename(columns={"Purchase":"Control_Purchase"})
control_df.head(5)

control_df.describe().T
control_df.shape

test_df = pd.read_excel("Week5/datasets/ab_testing.xlsx", sheet_name="Test Group")
test_df = test_df.dropna(how="all", axis="columns")
test_df = test_df.rename(columns={"Purchase":"Test_Purchase"})
test_df.head(5)
test_df.describe().T
test_df.shape


df = pd.concat([control_df, test_df], axis=1)
df.head(5)

############################################
# GÖREV 2: A/B Testinin Hipotez Tanımlanması
############################################

# Adım1: Hipotezleri kur
# H0: M1  = M2 (Kontrol grubu ve test grubu satın alma ortalamaları arasında fark yoktur.)
# H1: M1! = M2 (Kontrol grubu ve test grubu satın alma ortalamaları arasında fark vardır.)

# Adım2: Kontrol ve test grubu için purchase ortalamalarını analiz etme
df["Test_Purchase"].mean()
df["Control_Purchase"].mean()

############################################
# GÖREV 3: Hipotez Testinin Gerçekleştirilmesi
############################################

# Adım1: Normallik Varsayımı ve Varyans Homojenliği

# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1:..sağlanmamaktadır.

test_stat, pvalue = shapiro(df["Test_Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue)) # p-value = 0.1541

test_stat, pvalue = shapiro(df["Control_Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue)) # p-value = 0.5891

# p değeri 0.05'den küçük olmadığı her iki durum için de gözlenmektedir dolayısıyla H0'ı reddedemeyiz yani Normallik Varsayımı sağlanmaktadır

# Varsayım sağlandığı için Varyans Homojenliği Varsayımını incelememiz gerekir ve bu varsayımı incelemek için levene testi kullanılır.
# Varsayım Kontrolü
test_stat, pvalue = levene(df["Test_Purchase"], df["Control_Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue)) # p-value = 0.1083

# p-value 0.05 den büyük olduğu tekrar gözlemlenmektedir yani H0 reddedilemez ve 2 varsayım da sağlanmaktadır.
# Varsayımlar sağlandığı için bağımsız iki örneklem ttesti (parametrik test) yapılır.

test_stat, pvalue = ttest_ind(df["Test_Purchase"], df["Control_Purchase"], equal_var=True) # iki varsayım da sağlandırı için True
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue)) # p-value = 0.3493

#Yorum:
# p-value değeri ttest sonucu görüldüğü üzere 0.05'den büyük bir değer geldi bu yüzden H0 reddedilemez diyebiliriz bu da bize her ne kadar kontrol ve test grupları ortalamasında matematiksel olarak fark olduğu gözükse de kurduğumuz hipotez doğrultusunda "Kontrol grubu ve test grubu satın alma ortalamaları arasında farkın olmadığını istatistiksel olarak kanıtlamaktadır".
# Test olarak ttesti kullanıldı çünkü yukarıdan da görüldüğü üzere 2 varsayım da sağlanmaktadır.
# Elde edilen test sonuçlarına göre Kontrol grubu ve test grubu satın alma ortalamaları arasında fark yoktur bu yüzden Facebook'un tanıttığı yeni teklif verme türü olan average bidding'in maximum bidding'den daha fazla dönüşüm getirdiği söylenilemez. Yani eski teklif verme türü halen kullanılabilir.

