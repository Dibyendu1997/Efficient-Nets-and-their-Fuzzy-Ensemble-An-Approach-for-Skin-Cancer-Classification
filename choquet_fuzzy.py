import numpy as np
from sympy import solve, symbols

def get_lambda(measures):
    g1, g2, g3 = measures
    x = symbols('x')
    lmbd = solve((g1 * g2 * g3) * x ** 3 + 
                 (g1 * g2 + g2 * g3 + g1 * g3) * x **2  + 
                 (g1 + g2 + g3-1)*x , x)
    
    return lmbd[1]
    
def get_lambda1(measures):
    g1, g2, g3 ,g4,g5 = measures
    x = symbols('x',real=True)
    lmbd = solve((g1*g2*g3*g4*g5)*x**5 +
    (g1*g2*g3*g4 + g1*g2*g3*g5+ g2*g3*g4*g5 + g1*g3*g4*g5 + g1*g2*g4*g5)*x**4+
    (g1*g2*g3 + g1*g2*g4 + g1*g2*g5 +g1*g3*g5 +g1*g4*g5 +g2*g3*g4 +g2*g4*g5 +g2*g3*g5 + g3*g4*g5 +g1*g3*g4) * x ** 3 + 
                 (g1*g2 + g1*g3 + g1*g4+g1*g5 +g2*g3 +g2*g4+g2*g5+g3*g4+g3*g5+g4*g5) * x ** 2 + 
                 (g1 + g2 + g3 + g4+ g5 - 1) * x, x)
    for root in lmbd :
        if root >-1 and root != 0:
            return root    
    #return result
def get_lambda2(measures):
    g1, g2 = measures
    x = symbols('x')
    lmbd = solve((g1 * g2 ) * x **2  + 
                 (g1 + g2-1)*x , x)
    
    return lmbd[1]
def choquet_fuzzy_integral(X, lmbd):
	print("x",X)
	print("sorted",np.sort(X, order="prediction_score")[::-1])
	sorted_data = np.sort(X, order="prediction_score")[::-1]
	f_prev = sorted_data[0][1]
	print("f_prev",f_prev)
	print("1st prediction",sorted_data[0][0] * sorted_data[0][1])
	pred = sorted_data[0][0] * sorted_data[0][1]
	print("length of sorted data",len(sorted_data))
	for i in range(1, len(sorted_data)):
		f_cur = f_prev + sorted_data[i][1] + lmbd * sorted_data[i][1] * f_prev
		pred = pred + sorted_data[i][0] * (f_cur - f_prev)
		f_prev = f_cur
	return pred

def sugeno_fuzzy_integral(X, measures):
    return np.amax(np.minimum(np.take(X, np.arange(0, X.shape[0]), 0), measures), axis=0)

def ensemble(model_predictions, measures, mode='choquet'):
    models_count = len(model_predictions)
    print("model counts",models_count)
    assert models_count == len(measures)
    
    lmbd = get_lambda2(measures)
    dtype = [('prediction_score', float), ('fuzzy_measure', float)]
    final_predictions = list()
    print("length of model prediction",len(model_predictions[0])) 
    for i in range(len(model_predictions[0])):
        if mode == 'choquet':
        	#print("A")
        	print(model_predictions[1])
        	print(model_predictions[1][i])
        	score_values = [(model_predictions[j][i], measures[j]) for j in range(models_count)]
        	#print(score_values)
        	data_belong = np.array(score_values, dtype=dtype)
        	print("data belong",data_belong)
        	x_belong_agg = choquet_fuzzy_integral(data_belong, lmbd)
        	print(x_belong_agg)
        else:
            score_values = [model_predictions[j][i] for j in range(models_count)]
            data_belong = np.array(score_values)
            x_belong_agg = sugeno_fuzzy_integral(data_belong, measures)
        final_predictions.append(x_belong_agg)
   
    return np.array(final_predictions).argmax(axis=0)
