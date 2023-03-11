import tensorflow as tf
import tensorflow.keras as keras
from keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def regression_model(
        input_size=32,
        output_size=5,
        size_dense=8,
        dropout_rate=0.15,
        l1=1e-3,
        l2=1e-4,
        learning_rate=1e-3,
        mae_weight=2
):
    def custom_KL_MAE_loss(y_true, y_pred):
        y_true = tf.multiply(y_true, [1, 1, 1, 1, 0.01])
        y_pred = tf.multiply(y_pred, [1, 1, 1, 1, 0.01])
        kl = tf.keras.metrics.kl_divergence(y_true, y_pred)
        mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
        return kl + mae * mae_weight

    input_layer = Input(shape=input_size)
    x = Dense(size_dense, activation='relu', kernel_regularizer=regularizers.L1L2(l1, l2))(input_layer)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(size_dense, activation='sigmoid', kernel_regularizer=regularizers.L1L2(l1, l2))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    output_layer = Dense(output_size, activation='softmax')(x)

    model = Model(input_layer, output_layer)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=custom_KL_MAE_loss,  # "kullback_leibler_divergence",
                  metrics=['mae']
                  )
    return model

def l1_loss(y_true, y_pred):
    return np.sum(np.abs(y_true-y_pred))


def make_final_prediction(p, min_distance = 0.06, distance_step=0.0025, f=mean_absolute_error, verbose=0):
            
    ## calculate MAE for each pair within 20 predictions, 
    ## and select the predictions with highest number of similar predictions (mae<0.06)
    
    def count_similar_vectors(p, min_distance):
        distance = {}

        # calculate l1-distance/5 for each pair
        for i in range(len(p)):
            distance[i] = []
            for j in range(len(p)):
                distance[i].append(f(p[i], p[j]))

        # count similar vectors
        n_similar_node = [len([n for n in distance[i] if n<min_distance])-1 for i in range(len(p))]
        if verbose:
            print(f"--max number of similar vector for min_distance {min_distance}: {max(n_similar_node)}/{len(p)-1}")
        
        most_neighbors_nodes = [i for i in range(len(p)) if n_similar_node[i] == max(n_similar_node)]
        most_neighbors_dists = p[most_neighbors_nodes]
        return most_neighbors_dists
    
    def l1_distance_between_vectors(most_neighbors_dists):
        mae_of_similar_nodes = []
        for i in range(len(most_neighbors_dists)):
            for j in range(len(most_neighbors_dists)):
                mae_of_similar_nodes.append(f(most_neighbors_dists[i], most_neighbors_dists[j]))
        return mae_of_similar_nodes
    
    # make sure all vectors are similar
    while True:
        most_neighbors_dists = count_similar_vectors(p, min_distance)
        if len(most_neighbors_dists) <= 1:
            break
        mae_of_similar_nodes = l1_distance_between_vectors(most_neighbors_dists)
        if max(mae_of_similar_nodes) <= min_distance:
            break
        min_distance -= distance_step
    
    if verbose:
        print("-", np.array(most_neighbors_dists))
    g_pred = np.mean(np.array(most_neighbors_dists), axis = 0)
    if verbose:
        print("-mean: ", g_pred, "\n")
    
    return g_pred

def select_best_model_and_predict(
    score,
    emb_size,
    X_test_transformed,
    best_n = 2,
    min_mae_val=0.1,
    ):

    predictions = []
    best_mae_val = []

    for k in range(10):
        sc = score[score.k==k].sort_values('mae_val').head(best_n)
        sc = sc[sc.mae_val < min_mae_val]
        sc['l1'] = [str(i) if i != 0.0 else '0' for i in sc.l1]
        sc['l2'] = [str(i) if i != 0.0 else '0' for i in sc.l2]
        
        for i in range(len(sc)):
            dropout_rate = sc.dropout_rate.tolist()[i]
            l1 = sc.l1.tolist()[i]
            l2 = sc.l2.tolist()[i]
            learning_rate = sc.learning_rate.tolist()[i]
            mae_weight = sc.mae_weight.tolist()[i]
            batch_size = sc.batch_size.tolist()[i]
            k = sc.k.tolist()[i]
            n_try = sc.n_try.tolist()[i]
            model_name = f"model_{emb_size}_{dropout_rate}_{l1}_{l2}_{learning_rate}_{mae_weight}_{batch_size}_{k}_{n_try}"
            
            #load corresponding model
            model_path = f"./submission/checkpoints/NN/{model_name}.h5"
            model = regression_model(
                input_size=emb_size,
                output_size=5,
                size_dense=8,
                dropout_rate=dropout_rate,
                l1=float(l1),
                l2=float(l2),
                learning_rate=learning_rate,
                mae_weight=2
            )
            model.load_weights(model_path)
            Y_test_pred = model.predict(X_test_transformed, verbose=0)
            predictions.append(Y_test_pred)
            best_mae_val.append([sc.mae_val.tolist()[i], k])
        
    return np.array(predictions)

