# -*- coding: utf-8 -*-
"""
@author: Marisa Paone
Class: CS677
Facilitator: Sarah Cameron
Date: 8/13/23
Term Project

This script looks at all shark attacks ever recorded and predicts if they were fatal or not with logistic regression, Naive Bayesian,
Decision Tree, Random Forest, and KNN. It also looks at total attacks by country, activity, and shark species

"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection \
    import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import tree

ticker = 'shark_attack_incident_log.xlsx'
here = os.path.abspath(__file__)
input_dir = os.path.abspath(os.path.join(here, os.pardir))
df = pd.read_excel(os.path.join(input_dir, ticker))


try:
    # -------------Question 1-------------

    print(df)
    df.drop(range(6879,6910), inplace= True)

    df_select = df[['Date', 'Year', 'Type', 'Country', 'Area', 'Activity', 'Name', 'Sex ', 'Age', 'Injury', 'Fatal (Y/N)', 'Time', 'Species ']]
    print()
    print(df_select)

    # filtering out provoked attacks
    df_unprovoked = df_select.copy()
    df_unprovoked = df_select[df_select['Type'] == 'Unprovoked']

    df_unprovoked = df_unprovoked.dropna(subset=['Age'])
    df_unprovoked.drop(df_unprovoked[df_unprovoked['Year'] == 0].index, inplace = True)
    print(df_unprovoked)

    # Replacing country values with all capitalized values
    df_unprovoked['Country'] = df_unprovoked['Country'].str.upper()
    # Filtering out values like 20s, 30s, 40s
    df_unprovoked['Age'] = df_unprovoked['Age'].str.replace('s', '')
    # dropping not a number values for the Age column
    df_unprovoked = df_unprovoked.dropna(subset=['Age'])

    df_unprovoked['Age'] = pd.to_numeric(df_unprovoked['Age'], errors = 'coerce')
    df_unprovoked = df_unprovoked.dropna(subset = ['Age'])
    df_unprovoked['Age'] = df_unprovoked['Age'].astype(int)

    df_unprovoked['Class'] = np.where(df_unprovoked['Fatal (Y/N)'] == 'Y', 1, 0)

    # 25 most frequent shark types - White, Bull, Tiger, Wobbegong, Blacktip
    n = 25
    frequent_species = df_unprovoked['Species '].value_counts()[:n].index.tolist()
    print('\n25 Most Frequent Shark Species/Sizes that Attack\n',frequent_species)

    # filtering species data since each cell may differ in text
    df_unprovoked['Species '] = df_unprovoked['Species '].replace(r'.*tiger.*', 'Tiger shark', regex = True)
    df_unprovoked['Species '] = df_unprovoked['Species '].replace(r'.*Tiger.*', 'Tiger shark', regex=True)
    df_unprovoked['Species '] = df_unprovoked['Species '].replace(r'.*bull.*', 'Bull shark', regex=True)
    df_unprovoked['Species '] = df_unprovoked['Species '].replace(r'.*Bull.*', 'Bull shark', regex=True)
    df_unprovoked['Species '] = df_unprovoked['Species '].replace(r'.*white .*', 'White shark',  regex=True)
    df_unprovoked['Species '] = df_unprovoked['Species '].replace(r'.*White .*', 'White shark', regex=True)
    df_unprovoked['Species '] = df_unprovoked['Species '].replace(r'.*wobbegong.*', 'Wobbegong shark', regex=True)
    df_unprovoked['Species '] = df_unprovoked['Species '].replace(r'.*blacktip.*', 'Blacktip shark', regex=True)
    # Filling the values with just a ' ' and nothing else with NaN so that I can fill them with an unknown shark
    df_unprovoked['Species '] = df_unprovoked['Species '].replace(r'^\s*$', np.nan, regex = True)
    df_unprovoked['Species '] = df_unprovoked['Species '].fillna('Unknown')

    df_unprov_fatal = df_unprovoked[df_unprovoked['Class'] == 1]
    df_unprov_notfatal = df_unprovoked[df_unprovoked['Class'] == 0]

    print('Fatal attacks recorded:', len(df_unprov_fatal))
    print('Non Fatal attacks:', len(df_unprov_notfatal))

    # 15 most frequent shark species that attack
    n = 15
    frequent_species = df_unprovoked['Species '].value_counts()[:n].index.tolist()
    print('15 Most Frequent Shark Species/Sizes that Attack (after filtering data):\n',frequent_species)

    # 10 most frequent activies
    n = 10
    # we want the data from "scuba diving, free diving" etc.
    df_unprovoked['Activity'] = df_unprovoked['Activity'].replace(r'.*diving.*', 'Diving', regex=True)
    frequent_activities = df_unprovoked['Activity'].value_counts()[:n].index.tolist()
    print('10 Most Frequent Activities that end in a Shark Attack\n',frequent_activities)

    # Removing all other shark attacks that aren't from the 10 most common activities
    df_unprovoked['Activity'] = df_unprovoked['Activity'].astype(str)
    df_unprovoked = df_unprovoked[df_unprovoked['Activity'].isin(frequent_activities)]
    print(df_unprovoked)

    def calc_rates(cm):
        TP = cm[0, 0]
        FP = cm[0, 1]
        FN = cm[1, 0]
        TN = cm[1, 1]
        TPR = (TP / (TP + FN))
        TNR = (TN / (FP + TN))
        print('TPR = ', TPR)
        print('TNR = ', TNR)

    # Prediction/Classification of Fatalities are based on Country of Attack, the activity being performed,
    # Sex (M/F) of person, and the Age of the person being attacked

    input_data = df_unprovoked[['Country', 'Activity', 'Sex ', 'Age', 'Species ']]
    dummies = [pd.get_dummies(df_unprovoked[c]) for c in input_data.columns]
    binary_data = pd.concat(dummies, axis = 1)
    X  = binary_data[0:len(df_unprovoked)].values
    le = LabelEncoder()
    Y = df_unprovoked['Class'].values
    log_reg_classifier = LogisticRegression()

    # Training and Testing Sets

    x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.5, random_state=3)

    #--------------Logistic Regression--------------
    log_reg_classifier.fit(x_train, y_train)
    predict_LR = log_reg_classifier.predict(x_test)
    accuracy_LR = np.mean(predict_LR == y_test)
    print('\nLogistic Regression Accuracy:' , accuracy_LR)
    cm_LR = confusion_matrix(y_test, predict_LR)
    print(cm_LR)
    calc_rates(cm_LR)

    # --------------Naive Bayesian--------------
    NB_classifier = MultinomialNB().fit(x_train, y_train)
    predict_NB = NB_classifier.predict(x_test)
    accuracy_NB = accuracy_score(y_test, predict_NB)
    print('\nNaive Bayesian Accuracy:', accuracy_NB)
    cm_NB = confusion_matrix(y_test, predict_NB)
    print(cm_NB)
    calc_rates(cm_NB)

    # --------------Decision Tree--------------
    DT_classifier = tree.DecisionTreeClassifier()
    DT_classifier = DT_classifier.fit(x_train, y_train)
    predict_DT = DT_classifier.predict(x_test)
    print('\nDecision Tree Accuracy:', accuracy_score(y_test, predict_DT))
    cm_DT = confusion_matrix(y_test, predict_DT)
    print(cm_DT)
    calc_rates(cm_DT)

    #-------------- used from homework 5 to compute the best random forest to use--------------

    data = pd.DataFrame(columns=['estimators', 'depth', 'error rate', 'accuracy', 'pred'])
    print('\nRandom Forest: ')

    for i in range(10):
        for j in range(5):
            RF_classifier = RandomForestClassifier(n_estimators=i + 1, max_depth=j + 1, criterion='entropy')
            RF_classifier.fit(x_train, y_train)
            prediction_RF = RF_classifier.predict(x_test)
            error_rate = np.mean(prediction_RF != y_test)
            new_row = {'estimators': i + 1, 'depth': j + 1, 'error rate': error_rate,
                       'accuracy': accuracy_score(y_test, prediction_RF), 'pred': prediction_RF}
            data.loc[len(data)] = new_row

    # printing entire random forest dataframe (without predictions)
    print(data.iloc[:, [0, 1, 2, 3]])

    # plotting error rates
    one = data[data['depth'] == 1]
    two = data[data['depth'] == 2]
    three = data[data['depth'] == 3]
    four = data[data['depth'] == 4]
    five = data[data['depth'] == 5]

    plt.plot(one['estimators'], one['error rate'], color='red', marker='o', label='max depth = 1')
    plt.plot(two['estimators'], two['error rate'], color='green', marker='o', label='max depth = 2')
    plt.plot(three['estimators'], three['error rate'], color='blue', marker='o', label='max depth = 3')
    plt.plot(four['estimators'], four['error rate'], color='purple', marker='o', label='max depth = 4')
    plt.plot(five['estimators'], five['error rate'], color='orange', marker='o', label='max depth = 5')
    plt.title('Random Forest for Country, Activity, Species, Age, and Sex', fontsize=10)
    plt.xlabel('n_estimators')
    plt.ylabel('error rate')
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.show()

    # Calculating best accuracy row and lowest error row (they are always the same)
    best_accuracy = data.iloc[[data['accuracy'].idxmax()]]
    print('Best Accuracy Row for Random Forest: \n', best_accuracy.iloc[:, [0, 1, 2, 3]])
    lowest_error = data.iloc[[data['error rate'].idxmin()]]
    print('Lowest Error Row for Random Forest: \n', lowest_error.iloc[:, [0, 1, 2, 3]])

    # Calculating confusion matrix
    cm_RF = confusion_matrix(y_test, best_accuracy['pred'].iloc[0])
    print('Confusion matrix for n =', int(best_accuracy['estimators'].iloc[0]), 'and depth =',
          int(best_accuracy['depth'].iloc[0]), '\n', cm_RF)
    calc_rates(cm_RF)

    # --------------KNN--------------

    scaler = StandardScaler()
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.fit_transform(y_test)
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # initialize the kNN classifier and fit X and Y
    knn_classifier = KNeighborsClassifier(n_neighbors=1)
    knn_classifier.fit(x_train, y_train)
    predict_KNN = knn_classifier.predict(x_test)
    print("\nAccuracy for kNN with k =", 1, 'is:', accuracy_score(y_test, predict_KNN))
    cm_KNN = confusion_matrix(y_test, predict_KNN)
    print(cm_KNN)
    calc_rates(cm_KNN)

    knn_classifier = KNeighborsClassifier(n_neighbors=2)
    knn_classifier.fit(x_train, y_train)
    predict_KNN = knn_classifier.predict(x_test)
    print("\nAccuracy for kNN with k =", 2, 'is:', accuracy_score(y_test, predict_KNN))
    cm_KNN = confusion_matrix(y_test, predict_KNN)
    print(cm_KNN)
    calc_rates(cm_KNN)

    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    knn_classifier.fit(x_train, y_train)
    predict_KNN = knn_classifier.predict(x_test)
    print("\nAccuracy for kNN with k =", 3, 'is:', accuracy_score(y_test, predict_KNN))
    cm_KNN = confusion_matrix(y_test, predict_KNN)
    print(cm_KNN)
    calc_rates(cm_KNN)

    knn_classifier = KNeighborsClassifier(n_neighbors=4)
    knn_classifier.fit(x_train, y_train)
    predict_KNN = knn_classifier.predict(x_test)
    print("\nAccuracy for kNN with k =", 4, 'is:', accuracy_score(y_test, predict_KNN))
    cm_KNN = confusion_matrix(y_test, predict_KNN)
    print(cm_KNN)
    calc_rates(cm_KNN)

    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    knn_classifier.fit(x_train, y_train)
    predict_KNN = knn_classifier.predict(x_test)
    print("\nAccuracy for kNN with k =", 5, 'is:', accuracy_score(y_test, predict_KNN))
    cm_KNN = confusion_matrix(y_test, predict_KNN)
    print(cm_KNN)
    calc_rates(cm_KNN)


    # Taking a look at the top 15 countries with the most recorded shark attacks
    n = 15
    frequent_countries = df_unprovoked['Country'].value_counts()[:n].index.tolist()
    print('\n15 Most Frequent Shark Attack Countries:', frequent_countries)

    # Removing all other countries that aren't from the 30 most common countries for readibility purposes
    df_unprovoked_top_countries = df_unprovoked[df_unprovoked['Country'].isin(frequent_countries)]
    country_counts = df_unprovoked_top_countries['Country'].value_counts()

    #plot country vs number of attacks
    plt.figure(figsize=(10,6))
    plt.bar(country_counts.index, country_counts.values)
    plt.xlabel('Country')
    plt.ylabel('Number of Shark Attacks')
    plt.title('Shark Attacks by Country')
    plt.xticks(rotation = 45)
    plt.tight_layout()
    plt.savefig('Shark_Attacks_By_Country')
    plt.show()

    # plot species vs number of attacks
    frequent_species.remove('Unknown')
    print('\n15 Most Frequent Shark Attack Species:', frequent_species)
    df_unprovoked_top_species = df_unprovoked[df_unprovoked['Species '].isin(frequent_species)]
    species_counts = df_unprovoked_top_species['Species '].value_counts()

    plt.figure(figsize=(10, 6))
    plt.bar(species_counts.index, species_counts.values)
    plt.xlabel('Species')
    plt.ylabel('Number of Shark Attacks')
    plt.title('Shark Attacks by Species/Size')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('Shark_Attacks_By_Species')
    plt.show()

    # plotting activites vs shark attack counts
    print('\n10 Most Frequent Shark Attack Activies:', frequent_activities)
    df_unprovoked_top_activities = df_unprovoked[df_unprovoked['Activity'].isin(frequent_activities)]
    activity_counts = df_unprovoked_top_species['Activity'].value_counts()

    plt.figure(figsize=(10, 6))
    plt.bar(activity_counts.index, activity_counts.values)
    plt.xlabel('Activity')
    plt.ylabel('Number of Shark Attacks')
    plt.title('Shark Attacks by Activity')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('Shark_Attacks_By_Activity')
    plt.show()


    # taking a look at the most fatal attacks and what shark species they were
    n = 8
    frequent_fatal_species = df_unprov_fatal['Species '].value_counts()[:n].index.tolist()
    print('\n8 Most Frequent Fatal Shark Species:', frequent_fatal_species)
    frequent_fatal_species.remove('Unknown')
    df_unprovoked_fatal_species = df_unprov_fatal[df_unprov_fatal['Species '].isin(frequent_fatal_species)]
    fatal_species_counts = df_unprovoked_fatal_species['Species '].value_counts()

    plt.figure(figsize=(10, 6))
    plt.bar(fatal_species_counts.index, fatal_species_counts.values)
    plt.xlabel('Species')
    plt.ylabel('Number of Shark Attacks')
    plt.title('Fatal Shark Attacks by Species')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('Fatal_Shark_Attacks_By_Species')
    plt.show()

    # taking my best classifier and comparing it with just Country  activity and shark species

    input_data = df_unprovoked[['Country', 'Activity', 'Species ']]
    dummies = [pd.get_dummies(df_unprovoked[c]) for c in input_data.columns]
    binary_data = pd.concat(dummies, axis=1)
    X = binary_data[0:len(df_unprovoked)].values
    le = LabelEncoder()
    Y = df_unprovoked['Class'].values

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=3)

    RF_classifier = RandomForestClassifier(n_estimators=int(best_accuracy['estimators'].iloc[0]), max_depth=int(best_accuracy['depth'].iloc[0]), criterion='entropy')
    RF_classifier.fit(x_train, y_train)
    prediction_RF = RF_classifier.predict(x_test)
    error_rate = np.mean(prediction_RF != y_test)
    print('\nAccuracy for Random Forest without Age and Sex',accuracy_score(y_test, prediction_RF))
    cm_RF = confusion_matrix(y_test, prediction_RF)
    print(cm_RF)
    calc_rates(cm_RF)


except Exception as e:
    print(e)
    print('failed to read stock data for file: ', ticker)