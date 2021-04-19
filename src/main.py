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

    model = CrossValidation(bayesian = True)
    train_rate = model.train()
    print(train_rate)
    test_rate = model.test()
    print(test_rate)