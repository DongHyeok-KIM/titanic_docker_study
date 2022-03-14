import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from entity.entity import Entity
from service.predict_service import Service
#
# from entity import Entity
# from service import Service

class Controller:
    def __init__(self):
        self.entity = Entity()
        self.service = Service()

    def preprocessing(self):
        print('---------- 1. Drop PassengerId, Cabin, Ticket  ------------')
        this = self.service.drop_feature( 'PassengerId')
        this = self.service.drop_feature( 'Cabin')
        this = self.service.drop_feature( 'Ticket')
        print('---------- 2. Embarked, Sex Nominal ------------')
        this = self.service.embarked_nominal()
        this = self.service.sex_nominal()
        print('---------- 3. Fare Ordinal ------------')
        this = self.service.fare_ordinal()
        this = self.service.drop_feature('Fare')
        print('---------- 4. Title Nominal ------------')
        this = self.service.title_nominal()
        this = self.service.drop_feature('Name')
        print('---------- 5. Age Ordinal ------------')
        this = self.service.age_ordinal()
        print('---------- 6. Final Null Check ------------')
        print('train null count \n{}'.format(self.service.train.isnull().sum()))
        print('test null count \n{}'.format(self.service.test.isnull().sum()))
        print('---------- 7. Create Model ------------')
        self.service.x_train = self.service.train.drop('Survived', axis=1)
        self.service.y_label = self.service.train['Survived']
        return this

    def print_menu(self):
        print('0. Exit')
        print('1. Preprocessing')
        print('2. Data Visualize')
        print('3. Modeling')
        print('4. Learning')
        print('5. Submit')
        return input('Choose One\n')
    # def strat(self):
    #     while 1:
    #         menu = self.print_menu()
    #         print('Menu : %s ' % menu)
    #         if menu == '0':
    #             print('Stop')
    #             break
    #         if menu == '1':
    #             self.entity.context = './data/'
    #             self.entity.fname = 'train.csv'
    #
    #             self.service.train = self.service.load_csv_file(self.entity)
    #             self.entity.fname = 'test.csv'
    #             self.service.test = self.service.load_csv_file(self.entity)
    #             self.service.id = self.service.test['PassengerId']
    #         if menu == '2': pass
    #             #view = View()
    #             #temp = view.create_train()
    #             #view.plot_survived_dead(temp)
    #             # view.plot_sex(temp)
    #             # view.bar_chart(temp, 'Pclass')
    #         if menu == '3':
    #             this = self.preprocessing()
    #         if menu == '4':
    #             self.service.learning()
    #         if menu == '5':
    #             self.service.best_learning()
    def strat(self):
        self.entity.context = './app/data/'
        self.entity.fname = 'train.csv'
        self.service.train = self.service.load_csv_file(self.entity)
        self.entity.fname = 'test.csv'
        self.service.test = self.service.load_csv_file(self.entity)
        self.service.id = self.service.test['PassengerId']
        self.preprocessing()
        result = self.service.learning()
        return result



