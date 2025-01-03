import numpy as np
def predictor(sys,dia):
    gender = 1
    height = 162
    weight = 70.6
    ap_hi = sys
    ap_low = dia 
    smoke = 0
    alco = 0
    active = 1
    age = 40
    
    para = np.array([[gender, height, weight, ap_hi, ap_low,smoke, alco, active, age]])
    weights = np.array([[4.02893908e-01, -4.58755977e-02 , 1.94876975e-02 , 3.70367864e-02,
   3.13136279e-04, -3.14886534e-01 ,-2.89958223e-01, -2.19491849e-01,
   4.43183818e-02]])
    z =  np.dot(para.T,weights) + (-1.28722219)
    sigmoid = 1 / (1 + np.exp(-1*z))
    return sigmoid