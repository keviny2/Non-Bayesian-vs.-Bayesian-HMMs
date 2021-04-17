from CrossValidation import crossValidation

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

    model = crossValidation()
    train_rate = model.training_rate(MaxLike=False, BayesianLike = True)
    print(train_rate)
    test_rate = model.test_rate(MaxLike =True, BayesianLike = True)
    print(test_rate)