import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score,classification_report,precision_recall_fscore_support
from scipy.stats import f_oneway
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import pickle

df1=pd.read_excel('data\case_study1.xlsx')
df2=pd.read_excel('data\case_study2.xlsx')

df1=df1.loc[df1['Age_Oldest_TL']!=-99999]
columns_to_be_removed=[]
for i in df2.columns:
    if df2.loc[df2[i]==-99999].shape[0]>10000:
        columns_to_be_removed.append(i)
        
df2=df2.drop(columns_to_be_removed,axis=1)

for i in df2.columns:
    df2=df2.loc[df2[i]!=-99999]

#getting common column names 
for i in df1.columns:
    if i in df2.columns:
        print(i)
        
df=pd.merge(df1,df2,how='inner',on='PROSPECTID')

categorical_columns=[]
for i in df.columns:
    if df[i].dtype=='object' and i!='Approved_Flag':
        categorical_columns.append(i)
        
chi_val={}
for i in categorical_columns:
    chi2,pval,_,_=chi2_contingency(pd.crosstab(df[i],df['Approved_Flag']))
    chi_val[i]=pval
    
#all the categorical features have pvalue<=0.05 we will accept all

#For numerical features:
numerical_columns=[]
for i in df.columns:
    if i not in  categorical_columns and i not in ['PROSPECTID','Approved_Flag'] :
        numerical_columns.append(i)
        
# VIF check
vif_data=df[numerical_columns]
total_columns=vif_data.shape[1]
columns_to_be_kept=[]
column_index=0
vif_values={}
for i in range(0,total_columns):
    vif_value=variance_inflation_factor(vif_data,column_index)
    vif_values[numerical_columns[column_index]]=vif_value
    if vif_value<=6:
        columns_to_be_kept.append(numerical_columns[i])
        column_index+=1
    else:
        vif_data=vif_data.drop([numerical_columns[i]],axis=1)
        
        
#ANOVA for columns_to_be_kept
columns_to_be_kept_numerical=[]
for i in columns_to_be_kept:
    a=list(df[i])
    b=list(df['Approved_Flag'])
    
    group_P1=[value for value,group in zip(a,b) if group=='P1']
    group_P2=[value for value,group in zip(a,b) if group=='P2']
    group_P3=[value for value,group in zip(a,b) if group=='P3']
    group_P4=[value for value,group in zip(a,b) if group=='P4']
    
    f_stats,p_value=f_oneway(group_P1,group_P2,group_P3,group_P4)
    if p_value<=0.05:
        columns_to_be_kept_numerical.append(i)
        
        
## Label encoding categorical columns

features = columns_to_be_kept_numerical + ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']
df = df[features + ['Approved_Flag']]

df.loc[df['EDUCATION']=='SSC',['EDUCATION']]=1
df.loc[df['EDUCATION']=='OTHERS',['EDUCATION']]=1
df.loc[df['EDUCATION']=='12TH',['EDUCATION']]=2
df.loc[df['EDUCATION']=='GRADUATE',['EDUCATION']]=3
df.loc[df['EDUCATION']=='UNDER GRADUATE',['EDUCATION']]=3
df.loc[df['EDUCATION']=='POST-GRADUATE',['EDUCATION']]=4
df.loc[df['EDUCATION']=='PROFESSIONAL',['EDUCATION']]=3
df['EDUCATION'] = df['EDUCATION'].astype(int)
df_encoded=pd.get_dummies(df,columns=['MARITALSTATUS','GENDER','last_prod_enq2', 'first_prod_enq2'])

# Modeling
y=df_encoded['Approved_Flag']
x=df_encoded.drop(['Approved_Flag'],axis=1)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print('Radom Forest classifier')
rf=RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(x_train,y_train)
ypred=rf.predict(x_test)
print(classification_report(y_test, ypred,target_names=['P1','P2','P3','P4']))    
    
print("Decision Tree classifier")    
dt=DecisionTreeClassifier(max_depth=10,min_samples_split=10)
dt.fit(x_train,y_train)
ypred1=dt.predict(x_test)
print(classification_report(y_test, ypred1,target_names=['P1','P2','P3','P4']))
    

print('XGBoost')
xgb_classifier = xgb.XGBClassifier(objective='multi:softmax',  num_class=4)
y = df_encoded['Approved_Flag']
x = df_encoded. drop ( ['Approved_Flag'], axis = 1 )
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)
xgb_classifier.fit(x_train, y_train)
y_pred = xgb_classifier.predict(x_test)
print(classification_report(y_test, y_pred,target_names=['P1','P2','P3','P4'])) 


# hyperparameter tuning
param_grid={
    'colsample_bytree':[0.1,0.3,0.5,0.7,0.9],
    'learning_rate':[0.001,0.01,0.1,1],
    'max_depth':[3,5,8,10],
    'alpha':[1,10,100],
    'n_estimators':[10,50,100]
}
y = df_encoded['Approved_Flag']
x = df_encoded. drop ( ['Approved_Flag'], axis = 1 )
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)
xgb_classifier = xgb.XGBClassifier(objective='multi:softmax',  num_class=4,n_jobs=-1)
clf = GridSearchCV(xgb_classifier, param_grid)
clf.fit(x_train,y_train)
model=clf.best_estimator_
pickle.dump(model,open('model.sav','wb'))
print(classification_report(y_test,model.predict(x_test),target_names=['P1','P2','P3','P4']))