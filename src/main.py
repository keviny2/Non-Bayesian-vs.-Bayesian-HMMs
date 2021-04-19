from cross_validation import CrossValidation

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
        state_path, test_set, test_state_path = model.train(num_training=1000,
                                                            num_test=200,
                                                            num_iter=100,
                                                            num_burnin=100)
        model.test(state_path, test_set, test_state_path)