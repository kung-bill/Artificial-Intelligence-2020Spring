# Import packages here
import numpy as np
from sklearn.linear_model import LinearRegression

class Predictor:
    def __init__(self, dataset_path='salary_data.csv'):
        self.test = 0
        self.dataset_data, self.dataset_target = self.read_csv(dataset_path)
        self.model = self.train()

    @staticmethod
    def read_csv(file_path='salary_data.csv'):
        # Implement your CSV file reading here
        # returns data, target
        # Both outputs should be in numpy array format with type np.float64
        # You may reshape the array if necessary
        temp = np.loadtxt(file_path, dtype=str, delimiter=',')
        temp = temp[1:]
        YearsExperience = []
        Salary = []
        for i in range(len(temp)):
            YearsExperience.append(temp[i][0])
            Salary.append(temp[i][1])
        return np.array(YearsExperience, dtype=np.float64).reshape((len(temp), 1)), np.array(Salary, dtype=np.float64)

    def train(self):
        # returns sklearn's fitted LinearRegression model
        # Remember to pass self.dataset_data and self.dataset_target as its parameters
        reg = LinearRegression()
        model = reg.fit(self.dataset_data, self.dataset_target)
        return model
    def predict(self, x):
        # returns model's prediction given x as input
        return self.model.predict(x)

    def write_prediction(self, x, write_path='prediction.txt'):
        # opens a file using write_path with a writeable access
        # write all the outputs from the model's prediction to the file
        # You must write the output line by line instead of writing its numpy array or list object
        # This method does not return anything
        fp = open(write_path, 'w')
        for data in x:
            fp.write('{0:.2f}\n'.format(data))
        fp.close()


if __name__ == '__main__':
    # You may test your program here
    # Anything residing in this block will not be graded
    P = Predictor()
    x, y = P.read_csv()
    print('x')
    print(x)
    print('y')
    print(y)
    print('P.model.coef_ = ', P.model.coef_)
    print('P.model.intercept_ = ', P.model.intercept_)
    
    print(np.linalg.norm(np.dot(x, P.model.coef_) + P.model.intercept_ - y))
    print(np.linalg.norm(np.dot(x, np.array(15000)) + 25792.20019866871 - y))
    print(np.dot(x, P.model.coef_) + P.model.intercept_)
    print(y)
    print('P.model.predict(x)')
    print(P.model.predict(x))
    
    print(P.write_prediction(P.model.predict(x)))