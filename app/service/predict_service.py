"""
survival	생존여부	0 = No, 1 = Yes
PassengerId 테스트 시 문제로 제공됨
- pclass	승선권	1 = 1st, 2 = 2nd, 3 = 3rd
- sex	성별
- name 이름
Age	나이
- sibsp	동반한 형제, 자매, 배우자
- parch	동반한 부모,자식
- ticket	티켓번호
- fare	티켓의 요금
- cabin	객실번호
- embarked	승선한 항구명 C = 쉐부로, Q = 퀸즈타운, S = 사우스햄톤
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class Service:
    def __init__(self):
        self.train_file = pd.DataFrame()
        self.test_file = pd.DataFrame()
        self.id = []
        self.train = pd.DataFrame()
        self.x_train = pd.DataFrame()
        self.y_label = pd.DataFrame()
        self.label = pd.DataFrame()

    def load_csv_file(self, payload):
        print("csv 파일 불러오기 시작")
        filename = payload.context + payload.fname
        data = pd.read_csv(filename)
        print("csv 파일 불러오기 완료")
        return data
    def age_ordinal(self)-> []:
        train = self.train
        test = self.test
        train['Age'] = train['Age'].fillna(-0.5)
        test['Age'] = test['Age'].fillna(-0.5)
        bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
        labels = ['Unknown', 'Baby', 'Child', 'Teenager','Student','Young Adult','Adult','Senior']
        train['AgeGroup'] = pd.cut(train['Age'], bins, labels=labels)
        test['AgeGroup'] = pd.cut(test['Age'], bins, labels=labels)
        age_title_mappeing = {
            0: 'Unknown', 1: 'Baby', 2: 'Child', 3: 'Teenager', 4: 'Student', 5: 'Young Adult', 6: 'Adult', 7: 'Senior'
        }
        for x in range(len(train['AgeGroup'])):
            if train['AgeGroup'][x] == 'Unknown':
                train['AgeGroup'][x] = age_title_mappeing[train['Title'][x]]
        for x in range(len(test['AgeGroup'])):
            if test['AgeGroup'][x] == 'Unknown':
                test['AgeGroup'][x] = age_title_mappeing[test['Title'][x]]
        age_mapping = {
            'Unknown': 0, 'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4,'Young Adult': 5, 'Adult': 6,'Senior':7
        }
        self.train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
        self.test['AgeGroup'] = test['AgeGroup'].map(age_mapping)

    def title_nominal(self)-> []:
        combine = [self.train, self.test]
        for dataset in combine:
            dataset['Title'] = dataset.Name.str.extract('([A-Za-z])\.', expand=False)
        for dataset in combine:
            dataset['Title'] = dataset['Title'].replace(['Capt','Col','Don','Dr','Major','Rev','Jonkheer','Dona'], 'Rare')
            dataset['Title'] = dataset['Title'].replace(['Countess','Lady','Sir'], 'Royal')
            dataset['Title'] = dataset['Title'].replace(['Mile','Ms'], 'Miss')
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6, "Mne": 7}
        for dataset in combine:
            dataset['Title'] = dataset['Title'].map(title_mapping)
            dataset['Title'] = dataset['Title'].fillna(0)

    def fare_ordinal(self) -> []:
        self.train['FareBand'] = pd.qcut(self.train['Fare'], 4, labels=[1,2,3,4])
        self.test['FareBand'] = pd.qcut(self.test['Fare'], 4, labels=[1,2,3,4])
        self.train = self.train.fillna({'FareBand': 1})
        self.test = self.test.fillna({'FareBand': 1})

    def drop_feature(self, feature) -> []:
        self.train = self.train.drop([feature], axis= 1)
        self.test = self.test.drop([feature], axis= 1)

    def embarked_nominal(self)-> []:
        self.train = self.train.fillna({"Embarked" : "S"})
        city_mapping = {"S": 1, "C": 2, "Q": 3}
        self.train['Embarked'] = self.train['Embarked'].map(city_mapping)
        self.test['Embarked'] = self.test['Embarked'].map(city_mapping)

    def sex_nominal(self) ->[]:
        sex_mapping = {"male": 0, "female": 1}
        combine = [self.train, self.test]
        for dataset in combine:
            dataset['Sex'] = dataset['Sex'].map(sex_mapping)

    def learning(self):
        # print('결정트리 검증 정확도 {} % '.format(self.calculate_accuracy( DecisionTreeClassifier())))
        # print('랜덤프레스트 검증 정확도 {} % '.format(self.calculate_accuracy(RandomForestClassifier())))
        # print('KNN 검증 정확도 {} % '.format(self.calculate_accuracy(KNeighborsClassifier())))
        # print('나이브 베이즈 검증 정확도 {} % '.format(self.calculate_accuracy(GaussianNB())))
        # print('SVM 검증 정확도 {} % '.format(self.calculate_accuracy(SVC())))
        result =[
            {'DT acc': self.calculate_accuracy( DecisionTreeClassifier())},
        {'RF acc': self.calculate_accuracy(RandomForestClassifier())},
        {'KNN acc': self.calculate_accuracy(KNeighborsClassifier())},
        {'NB acc': self.calculate_accuracy(GaussianNB())},
        {'SVM acc': self.calculate_accuracy(SVC())}
        ]
        return result
    def best_learning(self):
        classfier = RandomForestClassifier()
        classfier.fit(self.x_train, self.y_label)
        prediction = classfier.predict(self.test)
        submission = pd.DataFrame(
            {'PassengerId': self.id, 'Survived': prediction}
        )
        submission.to_csv('./data/submission.csv', index=False)
    def calculate_accuracy(self, classfier):
        score = cross_val_score(classfier,
                                self.x_train,
                                self.y_label,
                                cv= KFold(n_splits=10, shuffle= True, random_state=0),
                                n_jobs=1,
                                scoring='accuracy')
        return round(np.mean(score) * 100, 2)

