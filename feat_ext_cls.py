import math
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import tensorflow as tf
import joblib

def butter_highpass(cutoff, fs, order):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def highpass_filter(data, cutoff=0.3, fs=25, order=4):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def bandpass_filter(ppg_signal, lowcut=0.5, highcut=5.0, fs=25, order=2):
    # print(f"Type of ppg signal from bandpass filter: {(ppg_signal)}")
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, ppg_signal)
    return y


def find_heart_rate(sys_peaks, fs):
    if len(sys_peaks) == 0:
        return 0
    peak_intervals = np.diff(sys_peaks) / fs
    heart_rates = 60 / peak_intervals
    average_heart_rate = np.mean(heart_rates)
    # ic(average_heart_rate)
    return average_heart_rate


def find_refliction_index(ppg_bwr, sys_peaks):
    n = len(ppg_bwr)
    inflection_peak = []
    tol = 1
    for peak in sys_peaks:
        idx = peak
        while idx+1 < n and ppg_bwr[idx] >= ppg_bwr[idx+1]:
            idx += 1
        if idx != peak and abs(idx - peak) >= tol:
            inflection_peak.append(idx)

    len_sys_peak = len(sys_peaks)
    if len_sys_peak == 0:
        return 0

    numerator, denominator = 0, 0
    for i in range(len_sys_peak):
        numerator += ppg_bwr[inflection_peak[i]]
        denominator += ppg_bwr[sys_peaks[i]]

    refliction_index = numerator/denominator
    return refliction_index


def find_lasi(ppg, fs):
    peaks = find_sys_foot_peaks(ppg,fs)
    sys,foot = [],[]
    for s,d in peaks:
        sys.append(s)
        foot.append(d)   
    

    s3 = []
    for i in range(len(sys)-1):
        half_pos = int((sys[i] + foot[i+1])/2)
        s3.append(half_pos)

    s2 = sys
    sampling_intrval = 1/fs
    min_length = min(len(s2),len(s3))
    area = 0
    for i in range(min_length):
        s2_index = s2[i]
        s3_index = s3[3]
        area += trapezoidal_area(ppg,s2_index,s3_index,sampling_intrval)
    
    area /= min_length
    return area



def find_sys_peaks(ppg_bwr, fs):
    sys_peaks, _ = find_peaks(ppg_bwr, prominence=0.6, distance=10)
    return sys_peaks


def calculate_pulse_areas(ppg_signal, fs):

    inverted_ppg_signal = -ppg_signal
    valleys, _ = find_peaks(inverted_ppg_signal, distance=10)
    # ic(valleys)
    # plot_scatter(ppg_signal,valleys)
    valleys = valleys[1:]
    A1, A2 = [], []
    for i in range(len(valleys)-1):
        start_index = i
        end_index = i+1
        pulse_segment = ppg_signal[start_index:end_index+1]
        time_segment = np.arange(start_index, end_index+1) / fs
        pulse_area = np.trapz(pulse_segment, time_segment)
        A1.append(pulse_area * 0.75)
        A2.append(pulse_area * 0.25)

    avg_a1 = sum(A1)/len(A1)
    avg_a2 = sum(A2)/len(A2)

    return avg_a2/avg_a1


def find_mnpv(ppg_signal, fs):
    sys_peaks, _ = find_peaks(ppg_signal, prominence=0.3, distance=15)
    iac = 0
    div = 0
    for i in range(1, len(sys_peaks)):
        iac += (ppg_signal[i] - ppg_signal[i-1])/fs
        div += 1
    if div == 0:
        return 0
    iac /= div
    peak_intervals = np.diff(sys_peaks) / fs
    average_interval = np.mean(peak_intervals)
    return iac/(iac + average_interval)


def find_crest_time(ppg_signal, fs):
    sys_peaks, _ = find_peaks(ppg_signal, prominence=0.3, distance=15)
    inverted_ppg_signal = -ppg_signal
    valleys, _ = find_peaks(inverted_ppg_signal, distance=10)
    crest_time = 0
    div = 0
    for a, b in zip(sys_peaks, valleys):
        # ic(a, b)
        div += 1
        crest_time += (a-b)/fs
    if div == 0:
        return 0
    crest_time /= div
    return crest_time


def find_sys_foot_peaks(ppg, fs):
    sys_peaks = find_sys_peaks(ppg, fs)
    PEAKS = []
    # store = []
    ppg_len = len(ppg)

    for peak in sys_peaks:
        idx = peak
        while idx >= 1 and ppg[idx] >= ppg[idx-1]:
            idx -= 1

        PEAKS.append([peak, idx])
        # store.append(idx)
    # plot_scatter(ppg,store)

    return PEAKS


def find_both_times(ppg, fs):
    PEAKS = find_sys_foot_peaks(ppg, fs)
    sys_time, foot_time = 0, 0
    peaks_len = len(PEAKS)

    if peaks_len == 0:
        return sys_time, foot_time

    for i in range(peaks_len):
        sys_time += ((PEAKS[i][0] - PEAKS[i][1])/fs)
    sys_time /= peaks_len

    for i in range(peaks_len - 1):
        foot_time += ((PEAKS[i+1][1] - PEAKS[i][0])/fs)
    foot_time /= peaks_len
    return sys_time, foot_time


def find_pir(ppg, fs):
    new_ppg = [-1 * item for item in ppg]
    PEAKS = find_sys_foot_peaks(new_ppg, fs)
    n = len(PEAKS)
    if n == 0:
        return 0
    # ic(PEAKS)
    pir = 0
    for s, d in PEAKS:
        pir += new_ppg[s]/new_ppg[d]
    pir /= n
    return pir

def find_augmentation_index(ppg, fs):
    peaks = find_sys_peaks(ppg, fs)
    primary_peak = ppg[peaks[0]]
    reflected_peak = ppg[peaks[1]]
    augmentation_pressure = reflected_peak - primary_peak
    pulse_pressure = np.max(ppg) - np.min(ppg)
    aix = augmentation_pressure / pulse_pressure
    return aix


def find_pulse_height(ppg, fs):
    PEAKS = find_sys_foot_peaks(ppg, fs)
    len_peaks = len(PEAKS)
    if len_peaks == 0:
        return 0
    sys_peaks, foot_peaks = [], []
    pulse_height = 0
    for a, b in PEAKS:
        sys_peaks.append(a)
        foot_peaks.append(b)
        pulse_height += ppg[a] - ppg[b]
    pulse_height /= len_peaks
    return pulse_height

def find_pulse_width(ppg, fs):
    PEAKS = find_sys_foot_peaks(ppg, fs)
    len_peaks = len(PEAKS)
    if len_peaks == 0:
        return 0
    A = [ele[0] for ele in PEAKS]
    B = [ele[1] for ele in PEAKS]
    pulse_width = []
    for i in range(len(B)-1):
        width = (B[i] - B[i+1])/fs
        pulse_width.append(width)

    return np.mean(pulse_width)

def find_hrv(ppg,fs):
    sys_peaks = find_sys_peaks(ppg,fs)
    ppi = np.diff(ppg[sys_peaks])
    hrv = np.std(ppi)
    return hrv

def find_amplitude_ratios(ppg,fs):
    sys_peaks = find_sys_peaks(ppg,fs)
    amplitude_ratios = ppg[sys_peaks] / np.roll(ppg[sys_peaks], 1)
    amplitude_ratios = amplitude_ratios[1:]
    mean_amplitude_ratios = np.mean(amplitude_ratios)
    return mean_amplitude_ratios

def find_max_min_amplitudes(ppg):
    max_amplitude = max(ppg)
    min_amplitude = min(ppg)
    return max_amplitude,min_amplitude

def find_womersley_number(ppg,fs):
    foot_peaks = find_sys_foot_peaks(ppg,fs)
    amp = [ppg[ele[1]] for ele in foot_peaks]
    return np.mean(amp)

def find_alpha(ppg,fs):
    peaks = find_sys_foot_peaks(ppg, fs)
    sys_peaks = [ele[0] for ele in peaks]
    foot_peaks = [ele[1] for ele in peaks]
    alpha = 0
    for i in range(len(sys_peaks)-1):
        alpha += (sys_peaks[i] - foot_peaks[i] - foot_peaks[i+1])
    alpha /= len(sys_peaks)
    return alpha

def trapezoidal_area(ppg, start_index, end_index, sampling_interval):
    
    segment = ppg[start_index:end_index+1]
    area = 0.0
    for i in range(len(segment) - 1):
        area += (segment[i] + segment[i+1]) / 2 * sampling_interval
    return area

def find_ipa(ppg,fs):
    peaks = find_sys_foot_peaks(ppg,fs)
    sys,foot = [],[]
    s1 = []
    tol = 0
    for s,d in peaks:
        sys.append(s)
        foot.append(d)   
        half_neg = int((s+d)/2)
        half_neg += tol
        s1.append(half_neg + tol)

    s3 = [int((sys[i] + foot[i+1])/2) for i in range(len(sys)-1)]
    

    s4 = foot[1:]
    s2 = sys


    sampling_interval = 1/fs
    min_length = min(len(s1),len(s2),len(s3),len(s4))
    AREA = 0
    total = 0
    for i in range(min_length):
        s1_index = s1[i]
        s2_index = s2[i]
        s3_index = s3[i]
        s4_index = s4[i]
        area1 = trapezoidal_area(ppg, s1_index, s2_index, sampling_interval)
        area2 = trapezoidal_area(ppg, s3_index, s4_index, sampling_interval)
        ipa_ratio = area1 / area2 if area2 != 0 else np.inf
        if ipa_ratio:
            AREA += ipa_ratio
            total += 1

    return AREA

def find_systolic_time_x(ppg, fs,val = 20):
    # ppg = ppg[::-1]
    sys_peaks = find_sys_peaks(ppg, fs)

    peak_len = len(sys_peaks)
    if peak_len == 0:
        return 0
    
    val += 15
    mul_fact = val/100

    foot = []
    sys_time = 0
    for peak in sys_peaks:
        idx = peak
        while idx >= 1 and ppg[idx] >= ppg[idx - 1]:
            idx -= 1
        add = math.ceil(idx + mul_fact * (peak - idx))
        # print(f"peak: {ppg[peak]}, 10%: {ppg[add]}")
        foot.append(add)
        if peak != add:
            sys_time += ((ppg[peak] - ppg[add])/fs)

    sys_time /= peak_len
    # ic(sys_time)
    return sys_time

def find_pwv(ppg,fs):
    # ppg = ppg[::-1]
    sys_peaks = find_sys_peaks(ppg, fs)

    peak_len = len(sys_peaks)
    if peak_len == 0:
        return 0
    foot = []
    for peak in sys_peaks:
        idx = peak
        while idx >= 1 and ppg[idx] >= ppg[idx - 1]:
            idx -= 1
        foot.append(idx)
    sampling_rate = 1/fs
    time_add = 0
    for i in range(peak_len):
        time_add += ((sys_peaks[i] - foot[i])/fs)
    distance = 1
    pwv = distance / time_add
    # ic(pwv)
    # plot_scatter(ppg,foot)
    return pwv


def extract_features_cls(ppg, fs=25):
    print("extract_features start")
    ppg = ppg[::-1]
    ppg_filtered = bandpass_filter(ppg)
    sys_peaks = find_sys_peaks(ppg_filtered, fs)
    hr = find_heart_rate(sys_peaks, fs)
    ref_ind = find_refliction_index(ppg_filtered, sys_peaks)
    lasi = find_lasi(ppg, fs)
    crest_time = find_crest_time(ppg_filtered, fs)
    mnpv = find_mnpv(ppg_filtered, fs)
    sys_time, foot_time = find_both_times(ppg_filtered, fs)
    pir = find_pir(ppg_filtered, fs)
    aix = find_augmentation_index(ppg_filtered, fs)
    pulse_height = find_pulse_height(ppg_filtered, fs)
    pulse_width = find_pulse_width(ppg_filtered, fs)
    hrv = find_hrv(ppg_filtered,fs)
    amplitude_ratios = find_amplitude_ratios(ppg_filtered,fs)
    max_amplitude, min_amplitude = find_max_min_amplitudes(ppg_filtered)
    womersley_number = find_womersley_number(ppg_filtered,fs)
    alpha = find_alpha(ppg_filtered,fs)
    ipa = find_ipa(ppg_filtered, fs)
    sys_time_ten = find_systolic_time_x(ppg_filtered, fs)
    pwv = find_pwv(ppg_filtered,fs)


    ret_dict = {
        "hr": hr,
        "ref_ind": ref_ind,
        "lasi": lasi,
        "crest_time": crest_time,
        "mnpv": mnpv,
        "sys_time": sys_time,
        "foot_time": foot_time,
        "pir": pir,
        "augmentation_index": aix,
        "pulse_height": pulse_height,
        "pulse_width": pulse_width,
        "hrv": hrv,
        "amplitude_ratios": amplitude_ratios,
        "max_amplitude":max_amplitude,
        "min_amplitude":min_amplitude,
        "womersley_number":womersley_number,
        "alpha":alpha,
        "ipa":ipa,
        "sys_time_ten":sys_time_ten,
        "pwv":pwv
    }
    print("extract_features end")

    return ret_dict