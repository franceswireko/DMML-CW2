import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import plot_roc_curve
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
import seaborn as sns


#Load the dataset
facebook = pd.read_csv('CW2_Facebook_metrics.csv')

#Display first 3 rows of the dataset
print(facebook.head)
print(facebook.shape)
print(facebook.dtypes)
print(facebook.describe())
print(facebook.info())

#identify missing values
print("Missing Values")
print(facebook.isna().sum())

#replacing missing values using fillna() function
facebook.fillna(0, inplace=True)

#check replacement of missing values
print("Missing Values")
print(facebook.isna().sum())

#select features and target variable
#first method
#split data into inputs and targets
X = facebook.drop(columns = ['Category','Type','Post Month', 'Post Weekday','Post Hour', 'Paid', 'Lifetime Post Consumers', 'Total Interactions'])
y = facebook['Category']
print(X, y)

print(X.columns)

# feature_cols = ['Page total likes','Lifetime Post Total Reach','Lifetime Post Total Impressions','Lifetime Engaged Users ','Lifetime Post Consumptions ',
# 'Lifetime Post Impressions by people who have liked your Page ','Lifetime Post reach by people who like your Page ',
# 'Lifetime People who have liked your Page and engaged with your post ','Comments','Likes','Shares']
# X = facebook[feature_cols] # Features
# y = facebook.Category # Target variable
# print(X, y)
# print(X.columns)


# #normalise the data
scaler = StandardScaler()
X1 = scaler.fit_transform(X)

#linear regression feature importance
model = LinearRegression()
model.fit(X1,y)
importance = model.coef_
for i, v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
#plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.savefig('Facebook_Features.png', dpi=1050)
plt.show()

# split X and y into training and testing sets
# split the data using 30%
X_train,X_test,y_train,y_test=train_test_split(X1,y, test_size=0.3,random_state=42)
# print("Testing Input")
# print(X1_test)
# print("Testing Output")
# #print(y_test)
print(y_test.count())


# Build the models
# b)# Create KNN classifier
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
knn_score = accuracy_score(y_test, y_pred_knn)
print("Accuracy score (KNN): ", knn_score)
cm1 = confusion_matrix(y_test, y_pred_knn)
print('KNN Confusion Matrix', cm1)

# c) Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
log_score = accuracy_score(y_test, y_pred_log)
print("Accuracy score (LR): ", log_score)
cm2 = confusion_matrix(y_test, y_pred_log)
print('LR Confusion Matrix')
print(cm2)

# d) random forest machine
rf_model=RandomForestClassifier(n_estimators=50,min_samples_split=10)
rf_model.fit(X_train,y_train)
y_pred_rf = rf_model.predict(X_test)
rf_score = accuracy_score(y_test, y_pred_rf)
print("Accuracy score (Random Forest): ", rf_score)
cm3 = confusion_matrix(y_test, y_pred_rf)
print('Random Forest Confusion Matrix')
print(cm3)

# e) support vector machine
svc_model = SVC(kernel='poly', degree=8)
svc_model.fit(X_train, y_train)
y_pred_svc = svc_model.predict(X_test)
svc_score = accuracy_score(y_test, y_pred_svc)
print("Accuracy score (Support Vector Machine): ", svc_score)
cm4 = confusion_matrix(y_test, y_pred_svc)
print('Support Vector Machine Confusion Matrix')
print(cm4)

# #SGD
sgd_model = SGDClassifier(random_state=42)
sgd_model.fit(X_train, y_train)
y_pred_sgd = sgd_model.predict(X_test)
sgd_score = accuracy_score(y_test, y_pred_sgd)
print("Accuracy score (SGD): ", sgd_score)
cm = confusion_matrix(y_test, y_pred_sgd)
print('SGD Confusion Matrix')
print(cm)


#f) #Ensemble
estimators=[('knn', knn_model), ('rf', rf_model), ('log_reg', log_model)]
#create our voting classifier, inputting our models
ensemble = VotingClassifier(estimators, voting='soft')
#fit model to training data
ensemble.fit(X_train, y_train)
#test our model on the test data
ens_score = ensemble.score(X_test, y_test)
ens_pred = ensemble.predict(X_test)

#Ensemble Accuracy Score
print('Ensemble Accuracy Score:', ens_score)

#Ensemble Confusion Matrix
cm_ensemble = confusion_matrix(y_test, ens_pred)
print(cm_ensemble)
# # visualize cfm
class_names=[1,2,3]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cm_ensemble), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
plt.savefig('Ensemble_heatmap.png', dpi=1080)

# # # Precision, Recall, F1 Score Report for Ensemble
# print('Ensemble Performance Report')
# print(classification_report(y_test, ens_pred))


# Tuning and performance
# #Build the ensemble using grid search to tune performance
#k-nearest neighbour
knn = KNeighborsClassifier()
#create a dictionary of all values we want to test for n_neighbors
params_knn = {'n_neighbors': np.arange(1, 25)}
#use gridsearch to test all values for n_neighbors
knn_gs = GridSearchCV(knn, params_knn, cv=5)
#fit model to training data
knn_gs.fit(X_train, y_train)
#save best model
knn_best = knn_gs.best_estimator_
#check best n_neighbors value
print(knn_gs.best_params_)
#logistic regression
log_reg = LogisticRegression()
#fit the model to the training data
log_reg.fit(X_train, y_train)
#Random Forest
rf = RandomForestClassifier()
#create a dictionary of all values we want to test for n_estimators
params_rf = {'n_estimators': [50, 100, 200]}
#use gridsearch to test all values for n_estimators
rf_gs = GridSearchCV(rf, params_rf, cv=5)
#fit model to training data
rf_gs.fit(X_train, y_train)
#save best model
rf_best = rf_gs.best_estimator_
#check best n_estimators value
print(rf_gs.best_params_)

#test the three models with the test data and print their accuracy scores
print('knn: {}'.format(knn_best.score(X_test, y_test)))
print('rf: {}'.format(rf_best.score(X_test, y_test)))
print('log_reg: {}'.format(log_reg.score(X_test, y_test)))


# #voting classifier
#create a dictionary of our models
estimators_gc=[('knn', knn_best), ('rf', rf_best), ('log_reg', log_reg)]
#create our voting classifier, inputting our models
ensemble_gc = VotingClassifier(estimators, voting='soft')
#fit model to training data
ensemble_gc.fit(X_train, y_train)
#test our model on the test data
ens_score_gc = ensemble_gc.score(X_test, y_test)
print('Voting Classifier Accuracy Score:', ens_score_gc)
#Confusion Matrix
ens_pred_gc = ensemble_gc.predict(X_test)
cm_ensemble_gc = confusion_matrix(y_test, ens_pred_gc)
print('Voting Classifier Confusion Matrix')
print(cm_ensemble_gc)

# # visualize cfm
class_names=[1,2,3]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cm_ensemble_gc), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.title('Voting Classifier Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
plt.savefig('Voting Classifier_heatmap.png', dpi=1080)

# # Precision, Recall, F1 Score Report for Voting Classifier
print('Voting Classifier Performance Report')
print(classification_report(y_test, ens_pred_gc))

# #ROC/AUROC Implementation
# plot_roc_curve(cm_ensemble_gc, X_test, ens_pred_gc)
# plt.savefig('ROC.png', dpi=1080, format='png')
