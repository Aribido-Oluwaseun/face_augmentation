"""
This file uses CNN to predict the age of people from their pictures
"""

class PredictAge:

    def train(X, y, param):
        """
        This function develops a model from a bunch of images and their labels and retuns a model
        :param X: A numpy array of images. Each row represents an image
        :param y: A python list containing the ages corresponding to the index of X
        :param param: A dictionary cache of parameters to be passed into the CNN
        :return: A Model
        """
        # todo: develop this function
        pass

    def predict(model, X_test):
        """
        This function predicts ages from the test data
        :param model: takes in the model developed from trainning
        :param X_test: the images whose ages are to be predicted
        :return: the predictions
        """


class CNN:

    def __init__(self, data, label, params):
        self.data = data
        self.params = params
        self.label = label

    def run(self):
        """

        :return: a CNN model
        """
        pass


def main():
    pass

if __name__ == '__main__':
    main()
