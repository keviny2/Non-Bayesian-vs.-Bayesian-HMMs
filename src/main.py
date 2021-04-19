import numpy as np
from cross_validation import CrossValidation


if __name__ == "__main__":

    train_bayesian = np.array([None] * 100)
    test_bayesian = np.array([None] * 100)

    train_maxlik = np.array([None] * 100)
    test_maxlik = np.array([None] * 100)

    bayesian = True
    if bayesian == True:
        num_training = 1000
        num_test = 200
        num_iter = 100
        num_burnin = 100

        for i in range(100):

            model = CrossValidation(bayesian = bayesian)
            rate, state_path, test_set, test_state_path = model.train(num_training=1000,
                                                                      num_test=200,
                                                                      num_iter=num_iter,
                                                                      num_burnin=num_burnin)
            train_bayesian[i] = rate
            test_bayesian[i] = model.test(state_path, test_set, test_state_path)
            print(train_bayesian[i])
            print(test_bayesian[i])

        print(train_bayesian)
        train_rate = np.mean(train_bayesian)
        test_rate = np.mean(test_bayesian)
        print("The training rate using Bayesian is: %d" %train_rate)
        print("The test rate using Bayesian is: %d" %test_rate)

    else:
        for i in range(100):
            try:
                model = CrossValidation(bayesian = bayesian)
            except:
                pass
            train_maxlik[i] = model.train(num_obs = 1000)
            test_bayesian[i] = model.test()


        train_rate = np.mean(train_maxlik)
        test_rate = np.mean(test_maxlik)
        print("The training rate using Maxlike is: %d" % train_rate)
        print("The test rate using Maxlike is: %d" % test_rate)


    # # TODO: (SHERRY) finish up cv
    # for i in range(100):
    #     # ... arr = np.array()
    #     model = CrossValidation(bayesian = bayesian)
    #     state_path, test_set, test_state_path = model.train(num_training=1000,
    #                                                         num_test=200,
    #                                                         num_iter=100,
    #                                                         num_burnin=100)
    #     model.test(state_path, test_set, test_state_path)

