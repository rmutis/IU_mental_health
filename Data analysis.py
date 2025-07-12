from functions import graphic, km_method, gmm_method, get_top_ten, cluster_desc
from itertools import product
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import string

# Set display of max. rows and columns to unlimited
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Import the raw data. The file must be in current working directory
raw_data = pd.read_csv("mental-heath-in-tech-2016_20161114.csv")

# First look at data
print(raw_data.head())
print("Shape of dataframe ", raw_data.shape)

# Count the amount of na-values per column
no_value = pd.DataFrame(raw_data.isna().sum(), columns=["number_na"])
print(no_value)

# Print the amount of different answers per question
overview = pd.DataFrame({
    "Column": raw_data.columns,
    "Different_answers": [raw_data[col].nunique(dropna=False) for col in raw_data.columns]
})
print(overview)

# The columns titles contain the question. For better handling we replace them with double letter e.g. AA
# For traceability reasons there will be also a translation table containing the letter and question
processed_data = raw_data.copy()
questions = processed_data.columns.tolist()
alphabet = list(string.ascii_uppercase)
combo = ["".join(x) for x in product(alphabet, repeat=2)]
new_title = combo[:len(questions)]
processed_data.columns = new_title
translation = pd.DataFrame({
    "original": questions,
    "translated": new_title
})
#translation.to_excel("Column translation table.xlsx")

# Print the different answers given per column. Except BL and BN due to free test answers
for column in processed_data.columns:
    if column not in ["BL", "BN"]:
        print(f"Answers in column ",column)
        print(processed_data[column].value_counts(dropna=False))
        print()

# No mapping for AA, AC, AD, AQ, AY, CA as they already have a numerical binary answer
columns_mapping = ["AB", "AE", "AF", "AG", "AH", "AI", "AJ", "AK", "AL", "AM", "AN", "AO", "AP", "AR", "AS", "AT",
                   "AU", "AV", "AW", "AX", "AZ", "BA", "BB", "BC", "BD", "BE", "BF", "BG", "BH", "BI", "BJ", "BK", "BM",
                   "BO", "BP", "BQ", "BR", "BS", "BT", "BU", "BV", "BY", "CB", "CC", "CD", "CE", "CK"]

# DROP AD, AQ - AX, BS to vast amount of na
# DROP CG, CI = State
# DROP BW, BX, BZ = Diagnosis question, previous questions are sufficient
# Treat na in a different manner: AB – AR, AZ – BJ
# TFIDF: BL, BN
# Hot encoding: CF, CH, CJ

# Replace columns with few answers with numeric answers
mapping = {
    # Column AB
    "1-5": 0, "6-25": 0.2, "26-100": 0.4, "100-500": 0.6, "500-1000": 0.8, "More than 1000": 1.0,
    # Column AE, AF, AG, AH, AI, AK, AL, AM, AN, AO, AP, AV, AW, BK, BM, BT, BU, BV, BY:
    "No": 0.0, "Not eligible for coverage / N/A": 0.0, "Not applicable to me": 0.0, "I am not sure": 0.5, "Maybe": 0.5,
    "Unsure": 0.5, "I don't know": 0.5, "Yes": 1.0,
    # Column AJ:
    "Very easy": 0.0, "Somewhat easy": 0.25, "Neither easy nor difficult": 0.5, "Somewhat difficult": 0.75,
    "Very difficult": 1.0,
    # Column AR:
    "No, I don't know any": 0.0, "I know some": 0.5, "Yes, I know several": 1.0,
    # Column AS, AU, BD: the answer "Not applicable to me" will be treated as NaN
    "No, because it would impact me negatively": 0.0, "No, because it doesn't matter": 0.0,
    "Sometimes, if it comes up": 0.5, "Sometimes": 0.5, "Yes, always": 1.0,
    # Column AX:
    "1-25%": 0.0, "26-50%": 1/3, "51-75%": 2/3, "76-100%": 1.0,
    # Column AZ, BB, BC, BI:
    "No, none did": 0.0, "None did": 0.0, "Some did": 0.5, "Yes, they all did": 1.0,
    # Column BA:
    "N/A (not currently aware)": 0.0, "No, I only became aware later": 0.0, "I was aware of some": 0.5,
    "Yes, I was aware of all of them": 1.0,
    # Column BE, BF, BJ:
    "None of them": 0.0, "Some of them": 0.5, "Yes, all of them": 1.0,
    # Column BG, BH:
    "No, at none of my previous employers": 0.0, "Some of my previous employers": 0.5,
    "Yes, at all of my previous employers": 1.0,
    # Column BO:
    "No, it has not": 0.0, "No, I don't think it would": 0.25, "Yes, I think it would": 0.75, "Yes, it has": 1.0,
    # Column BP:
    "No, they do not": 0.0, "No, I don't think they would": 0.25, "Yes, I think they would": 0.75, "Yes, they do": 1.0,
    # Column BQ: the answer "Not applicable to me (I do not have a mental ..." will be treated as NaN
    "Not open at all": 0.0, "Somewhat not open": 0.25, "Neutral": 0.5, "Somewhat open": 0.75, "Very open": 1.0,
    # Column BR:
    "Maybe/Not sure": 0.5, "Yes, I observed": 1.0, "Yes, I experienced": 1.0,
    # Column CB:
    "Never": 0.0, "Rarely": 0.25, "Often": 0.75, "Always": 1.0,
    # Column CD:
    3: 0, 15: 0, 17: 0, 19: 0, 20: 0,
    21: 0.25, 22: 0.25, 23: 0.25, 24: 0.25, 25: 0.25, 26: 0.25, 27: 0.25, 28: 0.25, 29: 0.25, 30: 0.25, 31: 0.25,
    32: 0.25, 33: 0.25, 34: 0.25, 35: 0.25, 36: 0.25, 37: 0.25, 38: 0.25, 39: 0.25, 40: 0.25,
    41: 0.5, 42: 0.5, 43: 0.5, 44: 0.5, 45: 0.5, 46: 0.5, 47: 0.5, 48: 0.5, 49: 0.5, 50: 0.5, 51: 0.5, 52: 0.5,
    53: 0.5, 54: 0.5, 55: 0.5, 56: 0.5, 57: 0.5, 58: 0.5, 59: 0.5,
    61: 0.75, 62: 0.75, 63: 0.75, 65: 0.75, 66: 0.75, 70: 0.75, 74: 0.75,
    99: 1, 323: 1,
    # Column CE:
    "Male":	0, "male": 0, "Male ": 0, "M": 0, "man": 0, "Cis male": 0, "Male.": 0, "male 9:1 female, roughly": 0,
    "Male (cis)": 0, "nb masculine": 0, "Malr": 0, "Sex is male": 0, "Dude": 0,
    "I'm a man why didn't you make this a drop down question. You should of asked sex? And I would of answered yes "
    "please. Seriously how much text can this take?": 0, "mail": 0, "M|": 0, "cisdude": 0, "cis man": 0,
    "Male/genderqueer": 0.5, "Male (trans, FtM)": 0.5, "Bigender": 0.5, "non-binary": 0.5, "Transitioned, M2F": 0.5,
    "Genderfluid (born female)": 0.5, "Other/Transfeminine": 0.5, "Androgynous": 0.5, "mtf": 0.5, "genderqueer": 0.5,
    "Human": 0.5, "Genderfluid": 0.5, "Enby": 0.5, "Other": 0.5, "Queer": 0.5, "Agender": 0.5, "Fluid": 0.5,
    "Nonbinary": 0.5, "Unicorn": 0.5, "genderqueer woman":	0.5, "Genderflux demi-girl": 0.5, "Transgender woman": 0.5,
    "none of your business": 0.5,
    "Female": 1, "I identify as female.": 1, "female ":	1, "Female assigned at birth ":	1, "F ": 1, "Woman": 1,
    "fm": 1, "Cis female ": 1, "Female or Multi-Gender Femme": 1, "female/woman": 1, "Cisgender Female": 1, "fem": 1,
    "Female (props for making this a freeform field, though)": 1, " Female": 1, "Cis-woman": 1,
    "female-bodied; no feelings about gender": 1, "AFAB": 1
    }
for x in columns_mapping:
    processed_data[x] = processed_data[x].map(mapping)


hot_encoding_columns = ["CF", "CH"]
one_hot_simple = processed_data[hot_encoding_columns].copy()
encoder = OneHotEncoder()
for x in hot_encoding_columns:
    one_hot_encoded = encoder.fit_transform(one_hot_simple[[x]]).toarray()
    one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out())
    print("Encoding: for column ", x, " following shape is added: ", one_hot_encoded_df.shape)
    processed_data = pd.concat([processed_data, one_hot_encoded_df], axis=1)
# Drop the original columns CF, CH
processed_data = processed_data.drop(columns=["CF", "CH"])

one_hot_difficult = processed_data["CJ"].copy()
encoded = one_hot_difficult.str.get_dummies(sep='|')
encoded.columns = [f"CJ_{c.strip()}" for c in encoded.columns]
print("Encoding: for column CJ following shape is added: ", encoded.shape)
processed_data = pd.concat([processed_data, encoded], axis=1)
# Drop the original columns CJ
processed_data = processed_data.drop(columns=["CJ"])


na_pro_dat = (processed_data.isna().sum()/len(processed_data))*100
all_columns = set(na_pro_dat.index)
na_df = pd.DataFrame(index=sorted(all_columns))
na_df["original data"] = na_pro_dat

na_df = na_df.fillna(0).astype(int)
#na_df.to_excel("Export NA values beginning.xlsx", index=True)




# If the question “Do you have previous employers?” (AY) was answered with 0 (=no) there are na-values in columns AZ – BJ. This occurred 169 times.
# If the question “Are you self-employed?” (AA) was answered with 1 (=yes) there are na-values in the columns AB – AP. This occurred 287 times.
# If the question “Are you self-employed?” (AA) was answered with 0 (=no) there are na-values in the columns AQ – AX. This occurred 1146 times.

columns1 = ["AZ", "BA", "BB", "BC", "BD", "BE", "BF", "BG", "BH", "BI", "BJ"]
columns2 = ["AB", "AC", "AD", "AE", "AF", "AG", "AH", "AI", "AJ", "AK", "AL", "AM", "AN", "AO", "AP"]
columns3 = ["AQ", "AR", "AS", "AT", "AU", "AV", "AW", "AX"]
condition1 = processed_data["AY"] == 0
condition2 = processed_data["AA"] == 1
condition3 = processed_data["AA"] == 0
processed_data.loc[condition1, columns1] = processed_data.loc[condition1, columns1].fillna(1.5)
processed_data.loc[condition2, columns2] = processed_data.loc[condition2, columns2].fillna(1.5)
processed_data.loc[condition3, columns3] = processed_data.loc[condition3, columns3].fillna(1.5)

# Count the amount of na-values per column
na_pro_dat = (processed_data.isna().sum()/len(processed_data))*100
all_columns = set(na_pro_dat.index)
na_df = pd.DataFrame(index=sorted(all_columns))
na_df["original data"] = na_pro_dat

na_df = na_df.fillna(0).astype(int)
#na_df.to_excel("Export NA values middle.xlsx", index=True)

na_columns = ["AD", "AF", "AT", "AV", "AX", "BQ", "BR", "BS", "CE"]
for x in na_columns:
    impute = SimpleImputer(missing_values=np.nan, strategy="mean")
    impute = impute.fit(processed_data[[x]])
    impute = impute.transform(processed_data[[x]])
    processed_data[x] = impute

na_pro_dat = (processed_data.isna().sum()/len(processed_data))*100
all_columns = set(na_pro_dat.index)
na_df = pd.DataFrame(index=sorted(all_columns))
na_df["original data"] = na_pro_dat

na_df = na_df.fillna(0).astype(int)
#na_df.to_excel("Export NA values end.xlsx", index=True)


# DROP AD, AQ - AX, BS to vast amount of na -> without these information
# DROP CG, CI = State -> OK
# DROP BW, BX, BZ = Diagnosis question, previous questions are sufficient -> OK



processed_data = processed_data.drop(columns=["BW", "BX", "BZ", "CG", "CI"])



freetext_answers = ["BL", "BN"]
free_text = processed_data[freetext_answers].copy()
for x in free_text:
    vectorizer = TfidfVectorizer(stop_words="english")
    vector = vectorizer.fit_transform(free_text[x].fillna("").astype(str))
    tfidf_df = pd.DataFrame(vector.toarray(), columns=[f"{x}_{word}" for word in vectorizer.get_feature_names_out()])
    print("TFIDF: for column ", x, " following shape is added: ", tfidf_df.shape)
    processed_data_text = pd.concat([processed_data, tfidf_df], axis=1)
# Drop the original columns BL and BN
processed_data = processed_data.drop(columns=["BL", "BN"])
processed_data_text = processed_data_text.drop(columns=["BL", "BN"])

processed_data = processed_data.astype(float)
processed_data_text = processed_data_text.astype(float)



scaler = MinMaxScaler()
processed_data = pd.DataFrame(scaler.fit_transform(processed_data), columns=processed_data.columns)
processed_data_text = pd.DataFrame(scaler.fit_transform(processed_data_text), columns=processed_data_text.columns)

na_pro_dat = (processed_data.isna().sum()/len(processed_data))*100
all_columns = set(na_pro_dat.index)
na_df = pd.DataFrame(index=sorted(all_columns))
na_df["original data"] = na_pro_dat

na_df = na_df.fillna(0).astype(int)
#na_df.to_excel("Export NA values3.xlsx", index=True)

results = pd.DataFrame(columns=["method", "data_source", "k", "BIC", "silhouette", "most_important_features",
                                "cluster_description"])




data_sources = {"processed_data": processed_data, "processed_data_text": processed_data_text}

reduced_data_sources = {}
labels_data_sources = {}


for name, dataframe in data_sources.items():
    for x in range(2,8):
        print("Start with K-Means Cluster = ", x, "Datasource = ", name)
        centroids, labels_km = km_method(x, dataframe)
        reduced_data_km, top_importance_km = get_top_ten(centroids, dataframe)
        centroids, labels_km = km_method(x, reduced_data_km)
        labels_data_sources[f"km_{x}"] = labels_km
        reduced_data_sources[f"km_{x}"] = reduced_data_km
        sil_score_km = silhouette_score(reduced_data_km, labels_km)
        cluster_km = cluster_desc(reduced_data_km, labels_km)

#        reduced_df_gmm.loc[:,"cluster"] = opt_labels_gmm
#        cluster_sum_gmm = reduced_df_gmm.groupby("cluster").mean()
#        model = KMeans(n_clusters=x, random_state=42)
#        model.fit_predict(dataframe)
#        importance_km = centroids.std(axis=0).sort_values(ascending=False)
#        top_importance_km = importance_km.iloc[:10].index
#        opt_model_km = KMeans(n_clusters=x, random_state=42)
#        reduced_df_km = dataframe[top_importance_km].copy()
#        opt_labels_km = opt_model_km.fit_predict(reduced_df_km)
        # TODO: calculate elbow method and extract best value -> write in table
#        reduced_df_km.loc[:,"cluster"] = opt_labels_km
#        cluster_sum_km = reduced_df_km.groupby("cluster").mean()

        print("Start with GMM Cluster = ", x, "Datasource = ", name)
        means, labels_gmm, bic = gmm_method(x, dataframe)
        reduced_data_gmm, top_importance_gmm = get_top_ten(means, dataframe)
        means, labels_gmm, bic = gmm_method(x, reduced_data_gmm)
        labels_data_sources[f"gmm_{x}"] = labels_gmm
        reduced_data_sources[f"gmm_{x}"] = reduced_data_gmm
        sil_score_gmm = silhouette_score(reduced_data_gmm, labels_gmm)
        cluster_gmm = cluster_desc(reduced_data_gmm, labels_gmm)

        # importance_gmm = means.std(axis=0).sort_values(ascending=False)
        # top_importance_gmm = importance_gmm.iloc[:10].index
        # reduced_df_gmm = dataframe[top_importance_gmm].copy()
        # opt_model_gmm = mixture.GaussianMixture(n_components=x)
        # opt_model_gmm.fit(reduced_df_gmm)
        # opt_labels_gmm = opt_model_gmm.predict(reduced_df_gmm)
        # labels_data_sources[f"gmm_{x}"] = opt_labels_gmm
        # reduced_data_sources[f"gmm_{x}"] = reduced_df_gmm
        # sil_score_gmm = silhouette_score(reduced_df_gmm, opt_labels_gmm)

        # reduced_df_gmm.loc[:,"cluster"] = opt_labels_gmm
        # cluster_sum_gmm = reduced_df_gmm.groupby("cluster").mean()

        gmm_km_results = pd.DataFrame([{"method": "K-Means", "data_source": name, "k": x, "BIC": "not applicable",
                                        "silhouette": sil_score_km, "most_important_features": top_importance_km,
                                        "cluster_description": cluster_km},
                                       {"method": "GMM", "data_source": name, "k": x, "BIC": bic,
                                        "silhouette": sil_score_gmm, "most_important_features": top_importance_gmm,
                                        "cluster_description": cluster_gmm}])
        results = pd.concat([results, gmm_km_results], ignore_index=True)

    print("Start with DBSCAN and Datasource = ", name)
    for x in np.arange(1, 3.5, 0.1):
        model = DBSCAN(eps=x, min_samples=5)
        labels_dbs = model.fit_predict(dataframe)
        outliers = np.sum(labels_dbs == -1)
        if outliers / len(dataframe) < 0.1:
            print("Start to obtain best values")
            print("DBS: Less than 10% outlier with eps = ", x)
            dbscan_data = dataframe.copy()
            dbscan_data["Labels"] = labels_dbs
            dbscan_data = dbscan_data[dbscan_data["Labels"] != -1]
            dbscan_data_no_label = dbscan_data.drop("Labels", axis=1)
            means = dbscan_data.groupby("Labels").mean()
            reduced_data_dbs, top_importance_dbs = get_top_ten(means, dbscan_data)
            k = dbscan_data["Labels"].nunique(dropna=False)
            dbs_results = pd.DataFrame([{"method": "DBSCAN", "data_source": name, "k": k,
                                         "silhouette": "not applicable", "BIC": "not applicable",
                                         "most_important_features": top_importance_dbs,
                                         "cluster_description": means}])
            results = pd.concat([results, dbs_results], ignore_index=True)
            graphic(dbscan_data_no_label, dbscan_data["Labels"], "DBSCAN", k)
            break

    # reduced_data_km, top_importance_km = get_top_ten(centroids, dataframe)
    #
    #
    # importance_dbs = means.std(axis=0).sort_values(ascending=False)
    # top_importance_dbs = importance_dbs.iloc[:10].index
    # means_top = means[top_importance_dbs]
    # # print(means_top)

    gmm_results = results[(results["method"] == "GMM") & (results["data_source"] == name)]
    gmm_results = gmm_results.sort_values(by="BIC", ascending=True).reset_index(drop=True)
    for x in range(0,4):
        sil_score = gmm_results.iloc[x, 4]
        if sil_score >= 0.5:
            value_sil_gmm = sil_score
            k_value = gmm_results.iloc[x, 2]
            print("Found an optimum for GMM at k = ", k_value, "with silhoette score of ", value_sil_gmm, " and BIC of",
                 gmm_results.iloc[x, 3])
            graphic(reduced_data_sources[f"gmm_{k_value}"], labels_data_sources[f"gmm_{k_value}"], "GMM", k_value)
            break

    value_sil_km = results[(results["method"] == "K-Means") & (results["data_source"] == name)]["silhouette"].max()
    print(value_sil_km)
    if value_sil_km >= 0.5:
        index_sil_km = results[(results["method"] == "K-Means") & (results["data_source"] == name)]["silhouette"].idxmax()
        k_value = results.iloc[index_sil_km, 2]
        print("Found an optimum for KM at k = ", k_value, "and silhouette score of ", value_sil_km)
        graphic(reduced_data_sources[f"km_{k_value}"], labels_data_sources[f"km_{k_value}"], "K-Means", k_value)

cluster_summary = pd.DataFrame()







from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np

# neigh = NearestNeighbors(n_neighbors=5)
# nbrs = neigh.fit(processed_data)
# distances, indices = nbrs.kneighbors(processed_data)
#
# # Sortierte Abstände zum k-nächsten Nachbarn
# distances = np.sort(distances[:, 4])  # 4 = 5. Nachbar (0-basierter Index)
# plt.plot(distances)
# plt.ylabel("Abstand zum 5. Nachbarn")
# plt.xlabel("Datenpunkt (sortiert)")
# plt.title("k-Distance Plot (für eps)")
# plt.grid(True)
# plt.show()



#print(results)

#print(centroids[importance[:20].index])

results.to_excel("model_results12.xlsx", index=False)

