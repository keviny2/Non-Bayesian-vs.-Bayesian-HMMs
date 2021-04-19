from CrossValidation import CrossValidation

if __name__ == "__main__":

    '''
    MaxLike
    '''
        # model = crossValidation()
        # train_rate = model.training_rate(MaxLike=True, BayesianLike = False)
        # print(train_rate)  #0.803
        # test_rate = model.test_rate(MaxLike=True, BayesianLike=False)
        # print(test_rate)  #0.845


    '''
    BayesLike
    '''
    bayesian = True

    # TODO: (SHERRY) finish up cv
    for i in range(100):
        # ... arr = np.array()
        model = CrossValidation(bayesian = bayesian)
        train_rate = model.train(num_obs=1000)
        print(train_rate)
        test_rate = model.test()
        print(test_rate)