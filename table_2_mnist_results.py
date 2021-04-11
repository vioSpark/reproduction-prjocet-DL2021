import yaml
import statistics

def open_results_file(filename):
    with open(filename, 'r') as stream:
        try:
            #print(yaml.safe_load(stream))
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return data

FILENAME = "results.yml"

if __name__ == "__main__":
    results = open_results_file(FILENAME)
    
    models = []

    for res in results:
        models.append(res['model'])

    #remove duplicates
    models = list(dict.fromkeys(models))

    models_acc = dict.fromkeys(models)

    for model in models:
        acc = []
        for res in results:
            if res['model'] == model:
                acc.append((1-res['acc'])*100)
        # print(len(acc))
        # print(model)
        # print(acc)
        print(f"Erorrs for: {model}: \n {acc}")
        # print(f"max: {max(acc)}")
        # print(f"min: {min(acc)}")
        # print(statistics.stdev(acc))
        models_acc[model] = sum(acc)/len(acc)
        print()
    
    print("Average error for SCALAR SESN")
    for k, val in models_acc.items():
        if 'scalar' in k:
            print(k, val)

    print()
    print("Average error for Vector SESN")
    for k, val in models_acc.items():
        if 'vector' in k:
            print(k, val)



### OUTPUT ###
# Erorrs for: mnist_ses_scalar_28: 
#  [1.9299999999999984, 2.2279999999999966, 1.9880000000000009, 2.0020000000000038, 2.298, 2.1039999999999948, 1.7020000000000035, 1.8240000000000034, 1.8979999999999997, 1.741999999999999, 1.868000000000003, 1.9260000000000055]

# Erorrs for: mnist_ses_scalar_56: 
#  [1.5959999999999974, 1.8399999999999972, 1.7179999999999973, 1.8020000000000036, 1.8240000000000034, 1.7920000000000047, 1.4859999999999984, 1.6000000000000014, 1.5739999999999976, 1.6859999999999986, 1.539999999999997, 1.7199999999999993]

# Erorrs for: mnist_ses_vector_28: 
#  [1.9320000000000004, 2.2460000000000035, 2.183999999999997, 2.1639999999999993, 2.2179999999999978, 2.2639999999999993, 1.7839999999999967, 1.8859999999999988, 1.756000000000002, 1.8859999999999988, 1.737999999999995, 1.9920000000000049]

# Erorrs for: mnist_ses_vector_56: 
#  [1.768000000000003, 1.6959999999999975, 1.756000000000002, 1.770000000000005, 1.7920000000000047]

# Erorrs for: mnist_ses_scalar_28p: 
#  [2.1059999999999968, 2.198, 1.759999999999995, 1.9220000000000015, 2.0000000000000018, 1.8480000000000052, 1.761999999999997, 1.8900000000000028]

# Erorrs for: mnist_ses_scalar_56p: 
#  [1.644000000000001, 1.854, 1.756000000000002, 1.761999999999997, 1.8379999999999952, 2.0399999999999974, 1.432, 1.6880000000000006, 1.532, 1.532, 1.539999999999997, 1.4839999999999964]

# Erorrs for: mnist_ses_vector_28p: 
#  [1.9279999999999964, 2.2639999999999993, 2.080000000000004, 2.210000000000001, 2.2340000000000027, 2.281999999999995, 1.8399999999999972, 1.8279999999999963, 1.8480000000000052, 1.6979999999999995, 1.8360000000000043]

# Erorrs for: mnist_ses_vector_56p: 
#  [1.6800000000000037, 1.7839999999999967, 1.744000000000001, 1.6199999999999992, 1.7260000000000053, 1.7979999999999996, 1.3859999999999983, 1.5499999999999958, 1.4419999999999988, 1.3660000000000005, 1.4040000000000052]

# Average error for SCALAR SESN
# mnist_ses_scalar_28 1.9591666666666674
# mnist_ses_scalar_56 1.6814999999999998
# mnist_ses_scalar_28p 1.9357499999999999
# mnist_ses_scalar_56p 1.6751666666666658

# Average error for Vector SESN
# mnist_ses_vector_28 2.0041666666666664
# mnist_ses_vector_56 1.7564000000000024
# mnist_ses_vector_28p 2.004363636363637
# mnist_ses_vector_56p 1.5909090909090908