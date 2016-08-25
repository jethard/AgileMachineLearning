from titanicforest import wrapper_for_titanic_Random_Forest_code

import pandas as pd
from pandas import Series,DataFrame
import csv as csv
from sklearn.ensemble import RandomForestClassifier
#from sklearn.cross_validation import train_test_split

class TestTitanicForest(TestCase):
    def test_junk(self):
        relative_test_set_size = 0.2

        # 20ish % is 'working'
        # 50ish % was where I thought I was stuck for a while
        # 95ish % is the highest I was getting
        #   Note that you want the test to pass consistently.  So whatever this
        # threshold is should be the worst-case-scenario training minimum
        accuracy_target = 0.75  # percent


        data =  pd.read_csv("data.csv")
        targets = data['Survived']

        from sklearn.cross_validation import train_test_split

        train_x, test_x, train_y, test_y = train_test_split(
            data, targets, test_size=relative_test_set_size)

        score = wrapper_for_titanic_Random_Forest_code(train_x, train_y, test_x, test_y)
        self.assertGreaterEqual(score, accuracy_target)
