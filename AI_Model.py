import pandas as pd
import matplotlib.pyplot as plt #for drawing 
import seaborn as sns
from sklearn.model_selection import train_test_split # AI Start from here
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# 1- loding File
df = pd.read_csv("./data/synthetic_students_500_AI_.csv", dtype={"AI_RecommendedAction": "object"})

#because now it type is float64 this from pandas
# print(df.head())

# 2- clean data
#a- fill non exit nale in int col with -1
#print(df.isnull().sum())
num_cols = ['Age', 'AttendanceRate', 'AcademicPerformance', 'CounselingSessions', 'HotlineCalls']
df[num_cols] = df[num_cols].fillna(-1)
#fill non exit nale in text col with unkown
text_cols = ['Name', 'BehaviorIncidents']
df[text_cols] = df[text_cols].fillna("unknown")
#print(df.isnull().sum())

#b-  to insure that i use correct datatype
#print(df.dtypes) #to insure that i use correct datatype

#c- check the repeatness
# print(df[df.duplicated(subset=['StudentID'])])
df.drop_duplicates(subset=['StudentID'], inplace=True)

#d- make sure that the we have appropate data that AI can hadle it
df['GenderNum'] = df['Gender'].map({'M':0, 'F':1})
df['BehaviorIncidentsNum'] = df['BehaviorIncidents'].factorize()[0] # it give num for each catagoary

#E- check anomaly value 
df['AttendanceRate'] = df['AttendanceRate'].clip(0, 100)
df['AcademicPerformance'] = df['AcademicPerformance'].clip(0, 100)
df['CounselingSessions'] = df['CounselingSessions'].clip(0)

#F save all cleaness
#df.to_csv("synthetic_students_500_cleaned.csv", index=False)

#3- analyse the relation between the catagories
'''
print(df.describe())
print(df['Gender'].value_counts())
print(df['BehaviorIncidents'].value_counts())

df['AttendanceRate'].hist(bins=20)
plt.title('Distribution of Attendance Rate')
plt.show()

bins = range(0, 101, 5) 
attendance_groups = pd.cut(df['AttendanceRate'], bins=bins, right=False)
counts = attendance_groups.value_counts().sort_index() 
print(counts)

sns.scatterplot(x='AttendanceRate', y='Age', data=df)
plt.show()
sns.scatterplot(x='AttendanceRate', y='Gender', data=df)
plt.show()

sns.scatterplot(x='AcademicPerformance', y='Age', data=df)
plt.show()
sns.scatterplot(x='AcademicPerformance', y='Gender', data=df)
plt.show()

sns.scatterplot(x='AcademicPerformance', y='BehaviorIncidents', data=df)
plt.show()

sns.scatterplot(x='Gender', y='BehaviorIncidents', data=df)
plt.show()

sns.scatterplot(x='Gender', y='CounselingSessions', data=df)
plt.show()


df['AcademicPerformance'].hist(bins=20)
plt.title('Distribution of Academic Performance')
plt.show()

sns.scatterplot(x='AttendanceRate', y='AcademicPerformance', data=df)
plt.show()

sns.scatterplot(x='AcademicPerformance', y='BehaviorIncidents', data=df)
plt.show()
'''
'''
#4-AI MODEl

def assign_action(row):
    if row['AttendanceRate'] < 60 or row['AcademicPerformance'] < 50 or row['BehaviorIncidents'] == "عنف":
        return "Immediate Intervention"
    elif row['CounselingSessions'] > 2:
        return "Counseling Referral"
    elif row['AttendanceRate'] >= 90 and row['AcademicPerformance'] >= 75 and row['BehaviorIncidents'] == "سلوك طبيعي":
        return "Regular Monitoring"
    else:
        return "Teacher Monitoring + Awareness Materials"
'''
def assign_action(row):
    if (
        row['AttendanceRate'] < 65  
        or row['AcademicPerformance'] < 57  
        or row['BehaviorIncidents'] == "عنيف"  
    ):
        return "Immediate Intervention"
    
    elif row['CounselingSessions'] > 2:
        return "Counseling Referral"
    
    elif row['AcademicPerformance'] > 88 or row['AttendanceRate'] > 90:
        return "Regular Monitoring"
    else:
        return "Teacher Monitoring + Awareness Materials"

df['AI_RecommendedAction'] = df.apply(assign_action, axis=1)

features = ['Age', 'GenderNum', 'AttendanceRate', 'AcademicPerformance',
            'CounselingSessions', 'BehaviorIncidentsNum', 'HotlineCalls']
X = df[features]
y = df['AI_RecommendedAction']
#training modle
# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)

# modle evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Adding Prediction
df_test = X_test.copy()
df_test['AI_RecommendedAction'] = y_test
df_test['AI_PredictedAction'] = y_pred

# 10. save result
#df_test.to_csv("students_test_predictions.csv", index=False)

#save the modle
joblib.dump(model, "decision_tree_model.pkl")
