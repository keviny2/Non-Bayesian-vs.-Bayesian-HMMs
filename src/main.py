import numpy as np
from cross_validation import CrossValidation


if __name__ == "__main__":

    cv_iter = 10
    train_bayesian = np.array([None] * cv_iter)
    test_bayesian = np.array([None] * cv_iter)

    train_maxlik = np.array([None] * cv_iter)
    test_maxlik = np.array([None] * cv_iter)

    num_training = 1000
    num_test = 1000

    bayesian = False
    if bayesian == True:
        num_iter = 100
        num_burnin = 100

        for i in range(cv_iter):

            model = CrossValidation(bayesian=bayesian, num_training=num_training, num_test=num_test)
            rate, state_path, test_set, test_state_path = model.train(num_iter=num_iter,
                                                                      num_burnin=num_burnin)
            train_bayesian[i] = rate
            test_bayesian[i] = model.test(state_path, test_set, test_state_path)
            print(train_bayesian[i])
            print(test_bayesian[i])

        print(train_bayesian)
        print(test_bayesian)
        train_rate = np.mean(train_bayesian)
        test_rate = np.mean(test_bayesian)
        print("The training rate using Bayesian is: %f" %train_rate)
        print("The test rate using Bayesian is: %f" %test_rate)

    else:
        i = 0
        while i < cv_iter:
            try:
                model = CrossValidation(bayesian=bayesian)
                rate, state_path, test_set, test_state_path = model.train(num_training=num_training, num_test=num_test)
                train_maxlik[i] = rate
                test_maxlik[i] = model.test(state_path, test_set, test_state_path)
                print(train_maxlik)
                print(test_maxlik)
                i += 1
            except:
                continue

        train_rate = np.mean(train_maxlik)
        test_rate = np.mean(test_maxlik)
        print("The training rate using Maxlike is: %f" % train_rate)
        print("The test rate using Maxlike is: %f" % test_rate)


