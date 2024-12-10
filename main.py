import joblib
import numpy as np
import pandas as pd
from feat_ext_reg import extract_features_reg
from feat_ext_cls import extract_features_cls
import math
import tensorflow as tf


def get_blood_pressure(ppg_signal):

    if len(ppg_signal) < 150:
        return None
    
    selected_features = [3, 4, 5, 6, 8, 10, 13, 14, 15, 17]
    model = joblib.load(r'reg_meta_data/knn_regressor.joblib')

    feature_scaler = joblib.load(
        r'reg_meta_data\\feature_scaler.joblib')

    target_scaler = joblib.load(r'reg_meta_data\\target_scaler.joblib')

    # create a dataframe with the extracted features and scale them
    feature_names = ['hr', 'ref_ind', 'lasi', 'crest_time', 'mnpv', 'sys_time', 'foot_time', 'pir', 'augmentation_index', 'pulse_height',
                     'pulse_width', 'hrv', 'amplitude_ratios', 'max_amplitude', 'min_amplitude', 'womersley_number', 'alpha', 'ipa', 'sys_time_ten', 'pwv']

    feat_dict = extract_features_reg(ppg_signal)
    feature = list(feat_dict.values())

    fin_feature_values = [feature[j] for j in selected_features]

    sel_col = [feature_names[i] for i in selected_features]

    # feature = [random.randint(100, 500) for _ in range(20)]
    test_df = pd.DataFrame(columns=sel_col)
    test_df.loc[len(test_df)] = np.array(fin_feature_values)
    test_df_scaled = feature_scaler.transform(test_df)

    # Prediction on the given ppg signal
    y_pred = model.predict(test_df_scaled)
    y_pred_rescaled = target_scaler.inverse_transform(y_pred.reshape(-1, 1))
    return math.ceil(y_pred_rescaled[0][0])

def predictive_classifier(ppg):
    
    ppg_len = len(ppg)

    if ppg_len >= 250:
        ppg_data = ppg[100:250]
    else:
        ppg_data = ppg


    fea_dict = extract_features_cls(ppg_data)

    features = []
    selected_features = ["crest_time","mnpv","sys_time","foot_time","augmentation_index","pulse_width","max_amplitude","min_amplitude","womersley_number","ipa"]

    for key in selected_features:
        features.append(fea_dict[key])

    features = np.array(features)
    features = features.reshape(1,-1)
    scaler = joblib.load(r"cls_meta_data\scaler.pkl")
    scaled_features = scaler.transform(features)
    classifier = tf.keras.models.load_model(r'cls_meta_data\trained_classifier.h5')
    prediction = classifier.predict(scaled_features)[0][0]
    predicted_class = (prediction >= 0.999).astype(int).flatten()

    predicted_class = np.array(predicted_class, dtype=np.int64)

    is_high_value = int(predicted_class[0])
    
    # prediction_result = {
    #     "isHigh": is_high_value,
    # }
    return is_high_value


def run_other_algo():
    print("...running other algo...")

ppg_signal = [2082.075,2084.625,2086.155,2088.195,2088.195,2085.135,2086.155,2086.92,2083.605,2083.35,2084.115,2082.585,2084.115,2081.82,2081.31,2081.565,2081.31,2081.31,2079.525,2079.525,2083.35,2084.115,2086.41,2086.41,2085.9,2086.41,2085.39,2086.92,2085.645,2083.35,2084.37,2082.585,2083.605,2082.585,2081.82,2081.82,2080.29,2080.035,2078.76,2079.78,2083.605,2082.84,2084.88,2087.94,2085.135,2086.155,2086.92,2085.9,2086.155,2085.135,2085.135,2084.625,2084.115,2085.39,2083.35,2083.605,2083.35,2081.565,2082.33,2085.9,2086.155,2088.705,2088.96,2087.94,2086.92,2087.43,2087.685,2085.135,2085.135,2085.9,2084.37,2085.135,2082.84,2082.075,-255.255,2081.565,2081.055,2079.27,2082.33,2086.665,2086.155,2086.92,2087.175,2087.43,2088.45,2086.92,2086.155,2086.92,2084.88,2085.645,2084.37,2083.605,2081.82,2082.33,2083.35,2080.545,2080.545,2081.31,2083.35,2086.41,2086.41,2088.45,2089.725,2088.705,2090.49,2089.47,2089.725,2090.49,2086.665,2087.175,2086.665,2086.155,2086.92,2086.665,2085.135,2085.645,2084.115,2085.645,2088.45,2090.745,2093.295,2093.04,2093.55,2092.785,2092.53,2092.02,2090.235,2089.47,2087.685,2089.215,2089.725,2089.725,2090.49,2087.685,2088.705,2088.195,2086.155,2088.96,2091.0,2093.04,2095.335,2093.295,2092.02,2094.06,2092.275,2092.02,2091.51,2090.49,2091.0,2089.215,2087.685,2087.175,2086.155,2085.645,2083.86,2082.585,2083.86,2085.645,2088.96,2088.45,2089.98,2093.295,2090.745,2091.765,2090.745,2091.255,2091.255,2089.47,2089.47,2091.51,2087.43,2086.92,2087.685,2084.88,2085.645,2083.86,2084.625,2089.98,2090.235,2093.04,2093.55,2092.275,2092.785,2092.785,2091.765,2091.0,2089.215,2089.98,2089.725,2090.235,2092.02,2090.49,2091.255,2089.215,2088.45,2088.45,2091.255,2094.315,2095.08,2096.865,2098.65,2097.12,2097.12,2097.375,2095.59,2096.355,2094.57,2094.825,2094.06,2093.295,2094.57,2093.295,2092.785,2093.295,2091.765,2093.55,2094.57,2094.315,2097.63,2096.61,2095.845,2094.315,2094.315,2096.355,2094.825,2093.805,2094.06,2094.315,2094.315,2092.02,2090.745,2093.04,2089.215,2089.215,2088.195,2092.53,2096.1,2096.61,2098.65,2098.14,2097.375,2098.395,2098.65,2097.63,2097.375,2094.57,2096.355,2094.315,2093.04,2095.59,2092.785,2093.295,2091.765,2091.51,2096.865,2097.375,2295.0,2296.53,2098.395,2295.255,2295.0,2098.395,2098.395,2096.355,2096.865,2096.355,2095.59,2096.355,2094.315,2092.785,2093.04,2090.235,2090.745,2093.55,2095.08,2097.63,2096.865,2097.63,2097.12,2096.61,2097.885,2095.08,2094.825,2096.1,2093.55,2095.08,2092.275,2092.53,2094.06,2093.04,2093.04,2089.47,2089.725,2093.805,2095.08,2097.375,2096.865,2097.12,2095.845,2095.845,2096.355,2093.805,2093.805,2094.06,2092.785,2093.55,2092.275,2092.275,2091.765,2091.51,2092.275,2089.98,2089.47,2093.805,2095.845,2098.395,2098.395,2295.765,2296.275,2098.905,2296.275,2295.51,2098.65,2295.255,2097.12,2097.63,2098.14,2095.845,2097.63,2095.08,2094.825,2092.785,2091.765,2095.59,2096.1,2098.14,2295.765,2296.02,2296.53,2098.905,2296.275,2097.885,2098.14,2098.14,2097.375,2098.395,2095.335,2094.06,2094.825,2092.785]
is_high = predictive_classifier(ppg_signal)
print(f"is_high: {is_high}")
if(is_high):
    BP = get_blood_pressure(ppg_signal)
    print(BP)
else:
    run_other_algo()




