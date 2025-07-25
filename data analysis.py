from functions import graphic, km_method, gmm_method, get_top_ten, cluster_desc, count_na
from itertools import product
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
import string

# Set display of max. rows and columns to unlimited
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Import the raw data. The file must be in current working directory
raw_data = pd.read_csv("mental-heath-in-tech-2016_20161114.csv")

# First look at data
print(raw_data.head())
print("Shape of dataframe ", raw_data.shape)

# The feature titles contain the question. For better handling we replace them with double letter e.g. AA
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
translation.to_excel("feature_translation_table.xlsx")

# Count the amount of na-values per column
count_na(processed_data)

# Export data to Excel for further investigation of na-values
raw_data.to_excel("raw_data.xlsx")

# Set a fixed na-value for follow-up questions that cannot be answered due to answer in initial question
columns1 = ["AZ", "BA", "BB", "BC", "BD", "BE", "BF", "BG", "BH", "BI", "BJ"]
columns2 = ["AB", "AC", "AD", "AE", "AF", "AG", "AH", "AI", "AJ", "AK", "AL", "AM", "AN", "AO", "AP"]
columns3 = ["AQ", "AR", "AS", "AT", "AU", "AV", "AW", "AX"]
condition1 = processed_data["AY"] == 0
condition2 = processed_data["AA"] == 1
condition3 = processed_data["AA"] == 0
processed_data.loc[condition1, columns1] = processed_data.loc[condition1, columns1].fillna(1.5)
processed_data.loc[condition2, columns2] = processed_data.loc[condition2, columns2].fillna(1.5)
processed_data.loc[condition3, columns3] = processed_data.loc[condition3, columns3].fillna(1.5)

# Print the different answers given per column. Except BL and BN due to free test answers
for x in processed_data.columns:
    if x not in ["BL", "BN"]:
        print("Answers in column ", x)
        print(processed_data[x].value_counts(dropna=False))

# Features relevant for mapping. No mapping for AA, AC, AD, AQ, AY, CA as they already have a numerical binary answer
columns_mapping = ["AB", "AE", "AF", "AG", "AH", "AI", "AJ", "AK", "AL", "AM", "AN", "AO", "AP", "AR", "AS", "AT",
                   "AU", "AV", "AW", "AX", "AZ", "BA", "BB", "BC", "BD", "BE", "BF", "BG", "BH", "BI", "BJ", "BK", "BM",
                   "BO", "BP", "BQ", "BR", "BS", "BT", "BU", "BV", "BY", "CB", "CC", "CE", "CK"]

# Definition of mapping to replace original answer to scaled numeric value
mapping = {
    # Column AE, AF, AG, AH, AI, AK, AL, AM, AN, AO, AP, AV, AW, BK, BM, BT, BU, BV, BY:
    "No": 0.0, "Not eligible for coverage / N/A": 0.0, "Not applicable to me": 0.0, "I am not sure": 0.5, "Maybe": 0.5,
    "Unsure": 0.5, "I don't know": 0.5, "Yes": 1.0,
    # Column AJ:
    "Very easy": 0.0, "Somewhat easy": 0.25, "Neither easy nor difficult": 0.5, "Somewhat difficult": 0.75,
    "Very difficult": 1.0,
    # Column AR:
    "No, I don't know any": 0.0, "I know some": 0.5, "Yes, I know several": 1.0,
    # Column AS, AU, BD:
    "No, because it would impact me negatively": 0.0, "No, because it doesn't matter": 0.0,
    "Sometimes, if it comes up": 0.5, "Sometimes": 0.5, "Yes, always": 1.0,
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
    # Column AB
    "1-5": 0, "6-25": 0.2, "26-100": 0.4, "100-500": 0.6, "500-1000": 0.8, "More than 1000": 1.0,
    # Column AX:
    "1-25%": 0.0, "26-50%": 1/3, "51-75%": 2/3, "76-100%": 1.0,
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
    "female-bodied; no feelings about gender": 1, "AFAB": 1,
    # Keep the already translated na-values
    1.5: 1.5
    }
# Apply mapping on the data
for x in columns_mapping:
    processed_data[x] = processed_data[x].map(mapping)

# Perform one hot encoding on the features CF and CH
hot_encoding_columns = ["CF", "CH"]
one_hot_simple = processed_data[hot_encoding_columns].copy()
encoder = OneHotEncoder()
for x in hot_encoding_columns:
    one_hot_encoded = encoder.fit_transform(one_hot_simple[[x]]).toarray()
    one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out())
    print("Encoding: for column ", x, " following shape is added: ", one_hot_encoded_df.shape)
    processed_data = pd.concat([processed_data, one_hot_encoded_df], axis=1)
# Drop the original features CF, CH and other features that are too detailed
processed_data = processed_data.drop(columns=["CF", "CH", "CG", "CI", "BW", "BX", "BZ"])

# Separate the entries of feature CJ and add a new feature for every unique value
one_hot_difficult = processed_data["CJ"].copy()
encoded = one_hot_difficult.str.get_dummies(sep='|')
encoded.columns = [f"CJ_{c.strip()}" for c in encoded.columns]
print("Encoding: for column CJ following shape is added: ", encoded.shape)
processed_data = pd.concat([processed_data, encoded], axis=1)
# Drop the original feature CJ
processed_data = processed_data.drop(columns=["CJ"])

# Count na values
count_na(processed_data)

# Process the following features and replace their na-values with the mean
na_features = ["AD", "AF", "AT", "AV", "AX", "BQ", "BR", "BS", "CE"]
for x in na_features:
    impute = SimpleImputer(missing_values=np.nan, strategy="mean")
    impute = impute.fit(processed_data[[x]])
    impute = impute.transform(processed_data[[x]])
    processed_data[x] = impute

# Apply binning on the column CD
processed_data["CD"] = processed_data["CD"].replace([3, 323], np.nan)
processed_data["CD"] = processed_data["CD"].fillna(processed_data["CD"].mean())
processed_data["CD"] = pd.cut(processed_data["CD"], bins=5, labels=[0.0, 0.25, 0.5, 0.75, 1.0])

# Store the free text features in a separate dataframe. Drop them in original dataframe
text_data = processed_data[["BL", "BN"]].copy()
processed_data = processed_data.drop(columns=["BL", "BN"])

# Convert all remaining columns to float and perform scaling
processed_data = processed_data.astype(float)
# scaler = RobustScaler()
# processed_data = pd.DataFrame(scaler.fit_transform(processed_data), columns=processed_data.columns)

# Apply TFIDF on the free text answers and add the results to a new dataframe
processed_data_text = pd.DataFrame()
for x in text_data:
    vectorizer = TfidfVectorizer(stop_words="english")
    vector = vectorizer.fit_transform(text_data[x].fillna("").astype(str))
    tfidf_df = pd.DataFrame(vector.toarray(), columns=[f"{x}_{word}" for word in vectorizer.get_feature_names_out()])
    print("TFIDF: for column ", x, " following shape is added: ", tfidf_df.shape)
    processed_data_text = pd.concat([processed_data, tfidf_df], axis=1)

# Preparation for modeling: create new dataframes to store results
results = pd.DataFrame(columns=["method", "data_source", "k", "BIC", "silhouette", "most_important_features",
                                "cluster_description"])
reduced_data_sources = {}
labels_data_sources = {}

# Perform modeling on 2 different datasets from k-values 2 to 7
data_sources = {"processed_data": processed_data, "processed_data_text": processed_data_text}
for name, dataframe in data_sources.items():
    for x in range(2, 8):
        # Modeling with K-Means, function from functions.py is used
        print("Start with K-Means Cluster = ", x, "Datasource = ", name)
        centroids, labels_km = km_method(x, dataframe)
        # Get ten most important features for modeling training and retrain the model only with these features
        reduced_data_km, top_importance_km = get_top_ten(centroids, dataframe)
        centroids, labels_km = km_method(x, reduced_data_km)
        # Store the data for later dimension reduction
        labels_data_sources[f"km_{x}"] = labels_km
        reduced_data_sources[f"km_{x}"] = reduced_data_km
        # Get Silhouette Score and cluster description
        sil_score_km = silhouette_score(reduced_data_km, labels_km)
        cluster_km = cluster_desc(reduced_data_km, labels_km)
        # Modeling now with GMM, only further difference to K-Means is calculation of BIC-value
        print("Start with GMM Cluster = ", x, "Datasource = ", name)
        means, labels_gmm, bic = gmm_method(x, dataframe)
        reduced_data_gmm, top_importance_gmm = get_top_ten(means, dataframe)
        means, labels_gmm, bic = gmm_method(x, reduced_data_gmm)
        # Store the data for later dimension reduction
        labels_data_sources[f"gmm_{x}"] = labels_gmm
        reduced_data_sources[f"gmm_{x}"] = reduced_data_gmm
        # Get Silhouette Score and cluster description
        sil_score_gmm = silhouette_score(reduced_data_gmm, labels_gmm)
        cluster_gmm = cluster_desc(reduced_data_gmm, labels_gmm)
        # The results for current k-value are now added as 2 new lines to result dataframe
        gmm_km_results = pd.DataFrame([{"method": "K-Means", "data_source": name, "k": x, "BIC": "not applicable",
                                        "silhouette": sil_score_km, "most_important_features": top_importance_km,
                                        "cluster_description": cluster_km},
                                       {"method": "GMM", "data_source": name, "k": x, "BIC": bic,
                                        "silhouette": sil_score_gmm, "most_important_features": top_importance_gmm,
                                        "cluster_description": cluster_gmm}])
        results = pd.concat([results, gmm_km_results], ignore_index=True)

    # Application of DBSCAN method, find optimal value of eps for less than 10% outliers
    print("Start with DBSCAN and Datasource = ", name)
    for x in np.arange(1, 10, 0.1):
        model = DBSCAN(eps=x, min_samples=5)
        labels_dbs = model.fit_predict(dataframe)
        outliers = np.sum(labels_dbs == -1)
        k = len(np.unique(labels_dbs)) -1
        if outliers / len(dataframe) < 0.05 and k < 8:
            print("Start to obtain best values")
            print("DBS: Less than ", outliers/len(dataframe)*100, "% outliers and ", k, " clusters with eps = ", x)
            # Copy dataframe, add the labels and remove all outlier (-1)
            dbscan_data = dataframe.copy()
            dbscan_data["Labels"] = labels_dbs
            dbscan_data = dbscan_data[dbscan_data["Labels"] != -1]
            cluster_count = dbscan_data["Labels"].value_counts()
            print("Following cluster and amount of members ", cluster_count)
            # Need of separate dataframe without label for later dimension reduction
            dbscan_data_no_label = dbscan_data.drop("Labels", axis=1)
            # Get means of each label for cluster description for most important features. Identify amount of labels
            means = dbscan_data.groupby("Labels").mean()
            reduced_data_dbs, top_importance_dbs = get_top_ten(means, dbscan_data)
            # New line in results dataframe to be added
            dbs_results = pd.DataFrame([{"method": "DBSCAN", "data_source": name, "k": k,
                                         "silhouette": "not applicable", "BIC": "not applicable",
                                         "most_important_features": top_importance_dbs,
                                         "cluster_description": means}])
            results = pd.concat([results, dbs_results], ignore_index=True)
            # Apply MDS dimension reduction from functions.py
            graphic(dbscan_data_no_label, dbscan_data["Labels"], "DBSCAN", k, name)
            break

    # Get the results from GMM and sort them by lowest BIC
    gmm_results = results[(results["method"] == "GMM") & (results["data_source"] == name)]
    gmm_results = gmm_results.sort_values(by="BIC", ascending=True).reset_index(drop=True)
    # Identify the silhouette score for lowest BIC and consider as optimum if >= 0.5. If not used next BIC value.
    for x in range(0, 4):
        sil_score = gmm_results.iloc[x, 4]
        if sil_score >= 0.5:
            value_sil_gmm = sil_score
            k_value = gmm_results.iloc[x, 2]
            print("Found an optimum for GMM at k = ", k_value, "with silhouette score of ", value_sil_gmm,
                  " and BIC of", gmm_results.iloc[x, 3])
            cluster_count = np.unique(labels_data_sources[f"gmm_{k_value}"], return_counts=True)
            print("Following cluster and amount of members ", cluster_count)
            # Apply MDS dimension reduction from functions.py
            graphic(reduced_data_sources[f"gmm_{k_value}"], labels_data_sources[f"gmm_{k_value}"], "GMM",
                    k_value, name)
            break

    # Get the results from K-Means by looking for best silhouette score
    value_sil_km = results[(results["method"] == "K-Means") & (results["data_source"] == name)]["silhouette"].max()
    print(value_sil_km)
    if value_sil_km >= 0.5:
        index_sil_km = results[(results["method"] == "K-Means") &
                               (results["data_source"] == name)]["silhouette"].idxmax()
        k_value = results.iloc[index_sil_km, 2]
        print("Found an optimum for KM at k = ", k_value, "and silhouette score of ", value_sil_km)
        cluster_count = np.unique(labels_data_sources[f"km_{k_value}"], return_counts=True)
        print("Following cluster and amount of members ", cluster_count)
        # Apply MDS dimension reduction from functions.py
        graphic(reduced_data_sources[f"km_{k_value}"], labels_data_sources[f"km_{k_value}"], "K-Means",
                k_value, name)

# Export to Excel to have a look at the different models and their results including cluster description
results.to_excel("model_results.xlsx", index=False)
