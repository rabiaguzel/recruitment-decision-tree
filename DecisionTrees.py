
# In[ ]:
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv("DecisionTreesdataset.csv")


# In[ ]:


df.head()

# In[ ]:


duzetme_mapping = {'Y': 1, 'N': 0}

df['IseAlindi'] = df['IseAlindi'].map(duzetme_mapping)
df['SuanCalisiyor?'] = df['SuanCalisiyor?'].map(duzetme_mapping)
df['Top10 Universite?'] = df['Top10 Universite?'].map(duzetme_mapping)
df['StajBizdeYaptimi?'] = df['StajBizdeYaptimi?'].map(duzetme_mapping)
duzetme_mapping_egitim = {'BS': 0, 'MS': 1, 'PhD': 2}
df['Egitim Seviyesi'] = df['Egitim Seviyesi'].map(duzetme_mapping_egitim)
df.head()


# Sonuc sütunu
# In[ ]:



y = df['IseAlindi']
X = df.drop(['IseAlindi'], axis=1)


# In[ ]:


X.head()


# Decision Tree

# In[ ]:


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,y)


# In[ ]:



# Predict
# 5 yıl deneyimli, hazlihazırda bir yerde çalışan ve 5 eski şirkette çalışmış olan, eğitim seviyesi Lisans
print (clf.predict([[5, 1, 5, 0, 0, 0]]))


# In[ ]:


# Toplam 1 yıllık iş deneyimi, 3 kez iş değiştirmiş çok iyi bir okul mezunu şuan çalışmıyor
print (clf.predict([[1, 0, 3, 0, 1, 0]]))


# In[ ]:


# Toplam 9 yıllık iş deneyimi, 5 kez iş değiştirmiş çok iyi bir okul mezunu değil şuan çalışıyor
print (clf.predict([[9, 1, 5, 0, 0, 0]]))


# In[ ]:


# Toplam 15 yıllık iş deneyimi, 6 kez iş değiştirmiş iyi bir okul mezunu şuan çalışmıyor
print (clf.predict([[15, 0, 6, 1, 1, 1]]))

## Toplu Öğrenme: Random Forest

# 10 tane decision tree birleşiminden oluşan bir Random Forest kullanarak tahmin:


# In[ ]:



rnd_fr_clf = RandomForestClassifier(n_estimators=20)
rnd_fr_clf = rnd_fr_clf.fit(X, y)
print (rnd_fr_clf.predict([[10, 1, 4, 0, 0, 0]]))
print (rnd_fr_clf.predict([[10, 0, 4, 0, 0, 0]]))