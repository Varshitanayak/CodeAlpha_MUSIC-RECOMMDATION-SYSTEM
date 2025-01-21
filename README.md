# CodeAlpha_MUSIC-RECOMMDATION-SYSTEM
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


data = {
    'user_id':[2, 3, 4, 5, 1, 2, 3, 4, 5,6],
    'song_id':[102, 103, 104, 105, 106, 107, 108, 109, 110,111],
    'play_count':[1, 10, 2, 0, 4, 3, 8, 1, 0,2],
    'liked':[0, 1, 0, 0, 1, 1, 1, 0, 0,1],
    'last_played_days_ago':[30, 1, 20, 60,25,3, 5, 1, 45, 90],
    'is_repeated_play':[10, 1, 0, 0, 1, 1, 1,0,0, 0]
       }


df=pd.DataFrame(data)


X=df[['play_count','liked','last_played_days_ago']]
y=df['is_repeated_play']


X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=42)


model=LogisticRegression()
model.fit(X_train, y_train)


y_pred=model.predict(X_test)


accuracy=accuracy_score(y_test, y_pred)
report=classification_report(y_test, y_pred)


print("Model Accuracy:",accuracy)
print("Classification Report:")
print(report)


new_data = pd.DataFrame({
    'play_count': [6,3],
    'liked': [1, 0],
    'last_played_days_ago': [4,30]
})


new_predictions=model.predict(new_data)
print("Predicted repeated play for new data:",new_predictions)
