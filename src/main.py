from crossValidation import crossValidation
import Plot

# maxlik
cv = crossValidation()
cv.training_rate(Maxlike=True, Bayesian = False)
