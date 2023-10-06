import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, \
                            precision_score, \
                            recall_score, \
                            f1_score, \
                            confusion_matrix, \
                            ConfusionMatrixDisplay

warnings.simplefilter(action='ignore', category=FutureWarning)

############################# Training  #########################
############################# Manipolazione Dataset #########################

Features = ['age', 'workclass', 'fnlwgt', 'education', 'education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']
df = pd.read_csv("Datasets/adult.data")

# Aggiungo la prima riga del dataset manualmente dato che non compare nel dataframe tramite la funzione pd.read_csv
# Manually adding the first row of the dataset since i noticed it is missing in the pd.read_csv import

df.loc[-1] = ['39', ' State-gov', '77516', ' Bachelors', '13', ' Never-married', ' Adm-clerical', ' Not-in-family', ' White', ' Male', '2174', '0', '40', ' United-States', ' <=50K']
df.index = df.index + 1
df.sort_index(inplace=True) 

df.columns = Features

# Visualizzo info relative al Dataset, principalmente alla ricerca di dati Null
print(df.info(),"\n")

# So che i missing values sono noti come "?" avendo cercato i distinti valori delle diverse Features.
# I know that all the missing values are marked as "?" 

# Tramite questo ciclo ho trovato quali features contengono dati ignoti.
# Thanks to this for loop i find which features contain "?" values

for i in Features:
    if ((' ?' in df[i].unique()) == True) :   #unique mi "ritorna" i valori della feature sotto forma di array
        print("Missing values occured in" , i , "feature")

# Sostituito valori mancanti con i valori più frequenti 
'''
# First Draft - calcolato la moda 
print("\n", df['native-country'].value_counts(),"\n")
print(df['workclass'].value_counts(),"\n")
print(df['occupation'].value_counts(),"\n")

# First Draft - Sostituito valori mancanti con i valori più frequenti ######

#df['native-country'] = df['native-country'].replace(' ?','United-States')
#df['workclass'] = df['workclass'].replace(' ?','Private')
#df['occupation'] = df['occupation'].replace(' ?','Prof-specialty')
'''
# Rimozione delle istanze contenenti valori mancanti 
# Removing instances containing missing values
df.drop(df[df['native-country'] == ' ?'].index,inplace=True)
df.drop(df[df['workclass'] == ' ?'].index,inplace=True)
df.drop(df[df['occupation'] == ' ?'].index,inplace=True)

###################################################################################################################################################

# Fattorizzo tutte le feature tramite il ciclo
# per ogni featue : df['workclass'], _ = pd.factorize(df['workclass'], sort=True)

for i in Features:
    df[i], _ = pd.factorize(df[i], sort=True)

'''
# First draft - Prima di passare a lavorare sui dati, converto i valori del dataframe da Str a Int 
# df['age'] = pd.to_numeric(df['age'], errors='coerce')

for i in Features:
    df[i] = pd.to_numeric(df[i], errors='coerce')

# [Verifica], se la correlation matrix mi aiuta a trovare un subset con precisione migliore

corr_matrix = df.corr()
sea.heatmap(corr_matrix, annot=True)
plt.show()
print(corr_matrix['income'])

show_feat = ['age', 'workclass', 'fnlwgt', 'education', 'education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']
sea.pairplot(df[show_feat])

Features = ['age', 'education-num','relationship','sex','capital-gain','hours-per-week']
'''

############################# Estrazione Target -- Estracting Target feature ######################################################

y = df['income'].values
y, _ = pd.factorize(y, sort=True)
X = df.drop(['income'], axis=1).values

############################# Scalamento dei dati -- Scaling ###########################################################################

scaler = StandardScaler().fit(X)
X_train=  scaler.transform(X)

############################# Definizione dei modelli -- Choosing models ###########################################################################

models = [
          LogisticRegression(multi_class='multinomial', solver='saga', class_weight='balanced', max_iter=100),
          DecisionTreeClassifier(class_weight='balanced'),
          KNeighborsClassifier(weights='distance')
         ]

models_names = [
                'Softmax Reg.',
                'DT',
                'KNN'
                ]


models_hparameters = [
                      {'penalty': ['l1', 'l2']},  # Softmax Reg
                      {'criterion': ['entropy', 'gini']},  # DT ---- C: 1 di default
                      {'n_neighbors': [31], 'weights' : ['uniform', 'distance']},  # K-NN ---- dopo vari test miglior K più ricorrente intorno ai 31
                     ]


chosen_hparameters = []
estimators = []

for model, model_name, hparameters in zip(models, models_names, models_hparameters):
        print('\n', model_name)
        clf = GridSearchCV(estimator=model, param_grid=hparameters, scoring='accuracy')
        clf.fit(X_train, y)
        chosen_hparameters.append(clf.best_params_)
        estimators.append((model_name, clf))
        print('Accuracy:  ', clf.best_score_)


clf_stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
scores = cross_validate(clf_stack, X_train, y, cv=5, scoring=('f1_weighted', 'accuracy'))

print('\n############ Ensemble  ############ \n')

print('The cross-validated weighted F1-score of the Stacking Ensemble is ', np.mean(scores['test_f1_weighted']))
print('The cross-validated Accuracy of the Stacking Ensemble is ', np.mean(scores['test_accuracy']))
print("\n")

############################# Scelta finale del modello #############################

final_model= clf_stack

############################# Training finale con tutto il dataset di training!  #########################

final_model.fit(X_train, y)

############################# Testing  #########################
############################# Manipolazione Dataset di Testing #########################

df = pd.read_csv("Datasets/adult.test")
df = df.reset_index(drop=False)

df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']


for i in Features:
    if ((' ?' in df[i].unique()) == True) :   #unique mi "ritorna" i valori della feature sotto forma di array
        print("Missing values occured in" , i , "feature")

'''
# First Draft - calcolato la moda 
print("\n", df['native-country'].value_counts(),"\n")
print(df['workclass'].value_counts(),"\n")
print(df['occupation'].value_counts(),"\n")

# First Draft - Sostituito valori mancanti con i valori più frequenti ######

df['native-country'] = df['native-country'].replace(' ?','United-States')
df['workclass'] = df['workclass'].replace(' ?','Private')
df['occupation'] = df['occupation'].replace(' ?','Prof-specialty')
'''

# Rimozione delle istanze contenenti valori mancanti 
df.drop(df[df['native-country'] == ' ?'].index,inplace=True)
df.drop(df[df['workclass'] == ' ?'].index,inplace=True)
df.drop(df[df['occupation'] == ' ?'].index,inplace=True)

##########################################################################################################################

# Fattorizzo tutte le feature 
# df['workclass'], _ = pd.factorize(df['workclass'], sort=True)

for i in Features:
    df[i], _ = pd.factorize(df[i], sort=True)

'''
# First draft
#Prima di passare a lavorare sui dati, converto i valori del dataframe da Str a Int 
#df['age'] = pd.to_numeric(df['age'], errors='coerce')

for i in Features:
    df[i] = pd.to_numeric(df[i], errors='coerce')
'''
############################# Estrazione Target ###############################################################################

y_test = df['income'].values
y_test, _ = pd.factorize(y_test, sort=True)
X = df.drop(['income'], axis=1).values

############################# Trasformazione dei dati ############################################################################

X_test=  scaler.transform(X)

############################# Prediction e  valutazione ###############################################################################

y_pred = final_model.predict(X_test)

###################################################################################################################################################

# ROC curve

y_pred_proba = final_model.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

# Confusion Matrix

predictions = final_model.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred, labels=final_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, 
                               display_labels=final_model.classes_)
disp.plot()
plt.show()

##########################################################################################################################

print('\n-------------------------------------------------------------------------------------------------------- /')
print('Final Testing RESULTS')
print('Accuracy is ', accuracy_score(y_test, y_pred))
print('Precision is ', precision_score(y_test, y_pred, average='weighted'))
print('Recall is ', recall_score(y_test, y_pred, average='weighted'))
print('F1-Score is ', f1_score(y_test, y_pred, average='weighted'))
print('AUC: ', auc)
#print("\nconfusion matrix for our final Model ",'\n',conf_mat)



