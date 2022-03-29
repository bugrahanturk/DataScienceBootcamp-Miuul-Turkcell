import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)


######################################
# Görev 1:  Veri Hazırlama
######################################

def create_user_movie_df():
    import pandas as pd
    # Adım 1:   movie, rating veri setlerini okutunuz.
    movie = pd.read_csv('Week4/datasets/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('Week4/datasets/movie_lens_dataset/rating.csv')
    # Adım 2:  rating veri setine Id’lerea it film isimlerini ve türünü movie veri setinden ekleyiniz.
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    # Adım3:  Toplam oy kullanılma sayısı 1000'in altında olan filmlerin isimlerini listed etutunuz ve veri setinden çıkartınız.
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    # Adım4: index'te userID'lerin sutunlarda film isimlerinin ve değer olarak ratinglerin bulunduğu dataframe için pivot table oluşturunuz
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df


# Adım5:  Yapılan tüm işlemleri fonksiyonlaştırınız.
user_movie_df = create_user_movie_df()


######################################
# Görev 2:  Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
######################################

def user_based_recommender(random_user, user_movie_df, ratio=60, cor_th=0.65, score=3.5):
    # Adım 2:  Seçilen kullanıcıya ait gözlem birimlerinden oluşan random_user_df adında yeni bir dataframe oluşturunuz
    random_user_df = user_movie_df[user_movie_df.index == random_user]

    # Adım3:  Seçilen kullanıcıların oy kullandığı filmleri movies_watched adında bir listeye atayınız
    movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

    ######################################
    # Görev 3:  Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişilmesi
    ######################################

    # Adım 1:  Seçilen kullanıcının izlediği fimlere ait sutunları user_movie_df'ten seçiniz ve movies_watched_df adında yeni bir data frame oluşturunuz
    movies_watched_df = user_movie_df[movies_watched]

    # Adım 2:  Her bir kullancının seçili user'in izlediği filmlerin kaçını izlediğini bilgisini taşıyan user_movie_count adında yeni bir dataframe oluşturunuz.
    user_movie_count = movies_watched_df.T.notnull().sum()

    user_movie_count = user_movie_count.reset_index()

    user_movie_count.columns = ["userId", "movie_count"]

    # Adım3:  Seçilen kullanıcının oy verdiği filmlerin yüzde 60 ve üstünü izleyenlerin kullanıcı id’lerinden users_same_movies adında bir liste oluşturunuz
    perc = len(movies_watched) * ratio / 100

    ######################################
    # Görev 4:  Öneri Yapılacak Kullanıcı ile En Benzer Kullanıcıların Belirlenmesi
    ######################################

    # Adım 1: user_same_movies listesi içerisindeki seçili user ile benzerlik gösteren kullanıcıların id’lerinin bulunacağı şekilde movies_watched_df dataframe’ini filtreleyiniz.
    users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]

    final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                          random_user_df[movies_watched]])

    # Adım 2: Kullanıcıların birbirleri ile olan korelasyonlarının bulunacağı yeni bircorr_dfdataframe’i oluşturunuz.
    corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()

    corr_df = pd.DataFrame(corr_df, columns=["corr"])

    corr_df.index.names = ['user_id_1', 'user_id_2']

    corr_df = corr_df.reset_index()

    # Adım3: Seçili kullanıcı ile yüksek korelasyona sahip(0.65’in üzerinde olan) kullanıcıları filtreleyerek top_users adında yeni bir data frame oluşturunuz.
    top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= cor_th)][
        ["user_id_2", "corr"]].reset_index(drop=True)

    top_users = top_users.sort_values(by='corr', ascending=False)

    top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

    rating = pd.read_csv('Week4/datasets/the_movies_dataset/ratings.csv')
    # Adım4:  top_users dataframe’ine rating veriseti ile merge ediniz
    top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')

    top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]

    ######################################
    # Görev 5:  Weighted Average Recommendation Score'un Hesaplanması ve İlk 5 Filmin Tutulması
    ######################################

    # Adım 1:   Her bir kullanıcının corr ve rating değerlerinin çarpımından oluşan weighted_rating adında yeni bir değişken oluşturunuz.
    top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

    top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})

    # Adım 2:  Film id’sive her bir filme ait tüm kullanıcıların weighted rating’lerinin ortalama değerini içeren recommendation_df adında yeni bir dataframe oluşturunuz.
    recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})

    recommendation_df = recommendation_df.reset_index()

    recommendation_df[recommendation_df["weighted_rating"] > score]

    # Adım3:  recommendation_df içerisinde weighted rating'i 3.5'ten büyük olan filmleri seçiniz ve weighted rating’e göre sıralayınız.
    movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > score].sort_values(
        "weighted_rating", ascending=False)
    # Adım4:  movie veri setinden film isimlerini getiriniz ve tavsiye edilecek ilk 5 filmi seçiniz
    movie = pd.read_csv('Week4/datasets/movie_lens_dataset/movie.csv')

    return movies_to_be_recommend.merge(movie[["movieId", "title"]])[:5]


# Adım 1:Rastgele bir kullanıcı id’si seçiniz
random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)
user_based_recommender(random_user, user_movie_df, ratio=60, cor_th=0.65, score=3)


######################################
# ITEM BASED
# Görev 1:  Kullanıcının izlediği en son ve en yüksek puan verdiği filme göre item-based öneri yapınız.
######################################

def item_based_recommender(user_movie_df):
    # Adım 1:   movie, rating veri setlerini okutunuz
    movie = pd.read_csv('Week4/datasets/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('Week4/datasets/the_movies_dataset/ratings.csv')

    df = movie.merge(rating, how="left", on="movieId")
    df.head()

    # Adım 2:  Seçili kullanıcının 5 puan verdiği filmlerden puanı en güncel olan filmin id'sinin alınız.
    rated_movie = df[(df["userId"] == random_user) & (df["rating"] == 5)].sort_values(by="timestamp", ascending=False)[
                  :1]
    rated_movie_id = rated_movie["movieId"].item()
    rated_movie_name = rated_movie["title"].item()

    # Adım3:  User based recommendation bölümünde oluşturulan user_movie_df dataframe’ini seçilen film id’sine göre filtreleyiniz
    movie_name = user_movie_df[rated_movie_name]
    # Adım 4:  Filtrelenen dataframe’i kullanarak seçili filmle diğer filmlerin korelasyonunu bulunuz ve sıralayınız.
    return user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(5)


# Adım5:  Seçili film’in kendisi haricinde ilk 5 film’I öneri olarak veriniz
item_based_recommender(user_movie_df)
