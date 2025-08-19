import os
import wfdb
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt, welch
import skfuzzy as fuzz
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random

# ===================== CONFIGURATION =====================
DATA_DIR = ""  # Path to PTB-XL dataset directory
OUTPUT_CSV = "mi_fuzzy_features_qrs_st_twave.csv"
TARGET_LABELS = ['NORM', 'IMI', 'AMI', 'LMI', 'ASMI', 'ILMI', 'ALMI']  # MI subclasses
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
TARGET_LENGTH = 100  # Fixed length for all segments after interpolation
PLOT_DIR = "Plot EKG"  # Directory to save visualizations
NUM_VISUALIZATION_RECORDS = 5  # Number of random records to visualize

# Create plot directory if not exists
os.makedirs(PLOT_DIR, exist_ok=True)

# ===================== PREPROCESSING FUNCTIONS =====================
def butter_bandpass(lowcut, highcut, fs, order=3):
    """Design bandpass Butterworth filter"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=0.5, highcut=40, fs=500, order=3):
    """Apply bandpass filter to remove noise"""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data)

def baseline_correction(signal, fs=500):
    """Remove baseline wander using high-pass filter"""
    # Strong high-pass to remove slow drifts
    b, a = butter(3, 0.5/(fs/2), btype='highpass')
    return filtfilt(b, a, signal)

def preprocess_signal(signal, fs=500):
    """Full preprocessing pipeline: baseline correction + noise filtering"""
    signal = baseline_correction(signal, fs)
    signal = bandpass_filter(signal, fs=fs)
    return signal

# ===================== FUZZY FEATURE FUNCTIONS =====================
def interpolate_segment(segment, target_length=TARGET_LENGTH):
    """Interpolate segment to fixed length"""
    if len(segment) < 2:
        return np.zeros(target_length)
    
    x_old = np.linspace(0, 1, len(segment))
    x_new = np.linspace(0, 1, target_length)
    interpolator = interp1d(x_old, segment, kind='linear', fill_value="extrapolate")
    return interpolator(x_new)

def extract_qrs_features(qrs):
    """Extract fuzzy features from QRS complex"""
    qrs = interpolate_segment(qrs)
    
    # Normalize amplitude
    baseline = np.median(qrs)
    qrs_normalized = qrs - baseline
    
    # Calculate important QRS parameters
    qrs_duration = len(qrs_normalized) / 500 * 1000  # Duration in ms (assuming fs=500)
    min_amplitude = np.min(qrs_normalized)
    max_amplitude = np.max(qrs_normalized)
    amplitude_range = max_amplitude - min_amplitude
    
    # ================= Fuzzy Logic for QRS Characteristics =================
    # Universe for QRS duration (ms)
    duration_universe = np.linspace(60, 200, 100)
    
    # Membership function: Wide QRS (abnormal)
    wide_qrs = fuzz.trapmf(duration_universe, [100, 120, 200, 200])
    
    # Universe for Q-wave amplitude (mV)
    qwave_universe = np.linspace(-1.0, 0, 100)
    
    # Membership function: Deep Q-wave (pathological)
    deep_qwave = fuzz.trapmf(qwave_universe, [-1.0, -1.0, -0.5, -0.2])
    
    # Universe for QRS fragmentation (amplitude variation)
    frag_universe = np.linspace(0, 2.0, 100)
    
    # Membership function: High fragmentation
    fragmented = fuzz.gaussmf(frag_universe, 0.5, 0.2)  # Amplitude variation > 0.5 mV
    
    # Calculate membership degrees
    wide_degree = fuzz.interp_membership(duration_universe, wide_qrs, qrs_duration)
    deepq_degree = fuzz.interp_membership(qwave_universe, deep_qwave, min_amplitude)
    
    # Calculate fragmentation as standard deviation of amplitude differences
    frag_value = np.std(np.diff(qrs_normalized))
    frag_degree = fuzz.interp_membership(frag_universe, fragmented, frag_value)
    
    # Calculate left/right amplitude ratio (for abnormality identification)
    center = len(qrs_normalized) // 2
    left_amp = np.mean(np.abs(qrs_normalized[:center]))
    right_amp = np.mean(np.abs(qrs_normalized[center:]))
    total_amp = left_amp + right_amp + 1e-8
    qrs_ratio = left_amp / total_amp

    return {
        'qrs_wide': wide_degree,
        'qrs_deep_qwave': deepq_degree,
        'qrs_fragmented': frag_degree,
        'qrs_ratio': qrs_ratio
    }

def extract_st_features(st_segment, lead_name=""):
    """Extract fuzzy features from ST segment"""
    st_segment = interpolate_segment(st_segment)
    baseline = np.median(st_segment)
    st_normalized = st_segment - baseline
    mean_val = np.mean(st_normalized)

    # Thresholds dalam satuan mV
    st_elev_thresh = 0.1  # 1 mm
    st_v2v3_thresh = 0.2  # 2 mm konservatif

    # Cek lead V2 atau V3
    if lead_name in ['V2', 'V3']:
        elev_thresh = st_v2v3_thresh
    else:
        elev_thresh = st_elev_thresh

    # Fuzzy logic untuk ST-elevasi
    st_universe = np.linspace(-0.5, 0.5, TARGET_LENGTH)
    st_elevated = fuzz.trapmf(st_universe, [elev_thresh, elev_thresh + 0.1, 0.5, 0.5])
    st_depressed = fuzz.trapmf(st_universe, [-0.5, -0.5, -0.1, -0.05])

    elevation_deg = fuzz.interp_membership(st_universe, st_elevated, mean_val)
    depression_deg = fuzz.interp_membership(st_universe, st_depressed, mean_val)

    return {
        'st_elevation': elevation_deg,
        'st_depression': depression_deg,
        'st_amplitude': np.max(st_normalized) - np.min(st_normalized),
    }


def extract_twave_features(t_wave):
    """Extract fuzzy features from T-wave"""
    t_wave = interpolate_segment(t_wave)
    
    # Normalize based on baseline
    baseline = np.median(t_wave)
    t_normalized = t_wave - baseline
    
    # Fuzzy universe for T-wave morphology
    t_universe = np.linspace(-0.5, 0.5, TARGET_LENGTH)
    
    # Inverted T-wave membership
    inverted = fuzz.gaussmf(t_universe, -0.2, 0.1)
    
    # Calculate membership values
    inv_degrees = fuzz.interp_membership(t_universe, inverted, t_normalized)
    
    return {
        't_inversion': np.max(inv_degrees),
        't_amplitude': np.max(t_normalized) - np.min(t_normalized)
    }

# ===================== SEGMENTATION AND FEATURE EXTRACTION =====================

def extract_features_one_beat(signal, r_peak, fs=500, lead_name=""):
    # QRS Complex (80ms around R-peak)
    qrs_start = max(0, r_peak - int(0.04 * fs))  # 40ms before
    qrs_end = min(len(signal), r_peak + int(0.04 * fs))  # 40ms after
    qrs = signal[qrs_start:qrs_end]

    # T Wave (120-280ms post R-peak)
    t_start = r_peak + int(0.12 * fs)
    t_end = min(len(signal), r_peak + int(0.28 * fs))
    if t_end > t_start:
        t_wave = signal[t_start:t_end]
    else:
        t_wave = np.zeros(10)
        t_start = t_end = r_peak  # fallback untuk hindari error

    # ST Segment (setelah QRS, sebelum T-wave)
    st_start = qrs_end
    st_end = t_start
    if st_end > st_start:
        st_segment = signal[st_start:st_end]
    else:
        st_segment = np.zeros(10)
        st_start = st_end = st_start  # fallback jika salah urutan

    return {
        **extract_qrs_features(qrs),
        **extract_twave_features(t_wave),
        **extract_st_features(st_segment, lead_name)
    }, (qrs_start, qrs_end, t_start, t_end, st_start, st_end)


# ===================== VISUALIZATION FUNCTIONS =====================
def plot_preprocessing(raw_signal, processed_signal, fs=500, record_id=""):
    """Visualize signal before and after preprocessing"""
    plt.figure(figsize=(15, 8))
    
    # Time vectors
    t_raw = np.arange(len(raw_signal)) / fs
    t_proc = np.arange(len(processed_signal)) / fs
    
    # Raw signal
    plt.subplot(2, 1, 1)
    plt.plot(t_raw, raw_signal, 'b-', linewidth=1.5)
    plt.title(f'Raw ECG Signal - {record_id}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (mV)')
    plt.grid(True)
    
    # Processed signal
    plt.subplot(2, 1, 2)
    plt.plot(t_proc, processed_signal, 'r-', linewidth=1.5)
    plt.title('After Preprocessing: Baseline Correction + Noise Filtering')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (mV)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{record_id}_preprocessing.png"))
    plt.close()

def plot_beat_segmentation(signal, r_peak, seg_points, fs=500, record_id="", beat_idx=0, lead_name=""):
    """Visualize beat segmentation with highlighted regions (QRS complex and T-wave only)"""
    qrs_start, qrs_end, t_start, t_end, st_start, st_end = seg_points
    
    plt.figure(figsize=(12, 6))
    
    # Create time vector
    start_point = max(0, r_peak - int(0.2 * fs))  # 200ms before R-peak
    end_point = min(len(signal), r_peak + int(0.4 * fs))  # 400ms after R-peak
    t = np.arange(start_point, end_point) / fs
    beat_signal = signal[start_point:end_point]
    
    plt.plot(t, beat_signal, 'b-', linewidth=2, label='ECG Signal')
    
    # Highlight regions
    plt.axvline(x=qrs_start/fs, color='g', linestyle='--', alpha=0.7)
    plt.axvline(x=qrs_end/fs, color='g', linestyle='--', alpha=0.7)
    plt.axvline(x=t_start/fs, color='m', linestyle='--', alpha=0.7)
    plt.axvline(x=t_end/fs, color='m', linestyle='--', alpha=0.7)
    plt.axvline(x=st_start/fs, color='m', linestyle='--', alpha=0.7)
    plt.axvline(x=st_end/fs, color='m', linestyle='--', alpha=0.7)
    
    # Add colored regions
    plt.axvspan(qrs_start/fs, qrs_end/fs, alpha=0.2, color='green', label='QRS Complex')
    plt.axvspan(t_start/fs, t_end/fs, alpha=0.2, color='purple', label='T Wave')
    plt.axvspan(st_start/fs, st_end/fs, alpha=0.2, color='orange', label='ST Segment')

    # Mark R-peak position
    plt.axvline(x=r_peak/fs, color='r', linestyle='-', alpha=0.7, label='R-peak')
    
    plt.title(f'Beat Segmentation - {record_id} (Beat {beat_idx}, Lead {lead_name})')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (mV)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{record_id}_beat{beat_idx}_{lead_name}_segmentation.png"))
    plt.close()

def plot_frequency_response(signal, fs=500, title=""):
    """Plot frequency spectrum of a signal"""
    f, Pxx = welch(signal, fs, nperseg=1024)
    plt.figure(figsize=(10, 4))
    plt.semilogy(f, Pxx, 'b-')
    plt.title(f'Frequency Spectrum: {title}')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.grid(True)
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{title.replace(' ', '_')}_spectrum.png"))
    plt.close()

# ===================== MAIN PROCESSING =====================
if __name__ == "__main__":
    # Load metadata
    df_meta = pd.read_csv(os.path.join(DATA_DIR, "ptbxl_database.csv"))
    scp_statements = pd.read_csv(os.path.join(DATA_DIR, "scp_statements.csv"), index_col=0)
    
    # Filter for diagnostic labels
    diagnostic_scp = scp_statements[scp_statements.diagnostic == 1].index
    df_meta['scp_codes'] = df_meta['scp_codes'].apply(eval)
    df_meta['diagnostic'] = df_meta['scp_codes'].apply(
        lambda x: [k for k in x.keys() if k in diagnostic_scp]
    )
    df_meta = df_meta[df_meta['diagnostic'].map(len) > 0]
    df_meta['label'] = df_meta['diagnostic'].apply(lambda x: x[0])
    df_meta = df_meta[df_meta['label'].isin(TARGET_LABELS)].reset_index(drop=True)
    
    print(f"Found {len(df_meta)} records with target labels")
    
    # Select random records for visualization
    if len(df_meta) > NUM_VISUALIZATION_RECORDS:
        viz_indices = random.sample(range(len(df_meta)), NUM_VISUALIZATION_RECORDS)
    else:
        viz_indices = range(len(df_meta))
    
    print(f"Selected {len(viz_indices)} records for visualization")
    
    # Process records
    all_features = []
    for idx, row in df_meta.iterrows():
        record_path = os.path.join(DATA_DIR, row['filename_hr'].replace('.hea', ''))
        record_id = os.path.basename(record_path)
        try:
            # Read ECG signal with all 12 leads
            signals, fields = wfdb.rdsamp(record_path)
            fs = fields['fs']
            
            # Preprocess reference lead (Lead II)
            raw_ref_signal = signals[:, 1].flatten()
            ref_signal = preprocess_signal(raw_ref_signal, fs)
            
            # Generate visualizations only for selected records
            if idx in viz_indices:
                print(f"Generating visualizations for record {record_id}")
                plot_preprocessing(
                    raw_ref_signal[:5*fs], 
                    ref_signal[:5*fs],
                    fs,
                    record_id=record_id
                )
                plot_frequency_response(raw_ref_signal, fs, f"{record_id} Raw Signal")
                plot_frequency_response(ref_signal, fs, f"{record_id} Processed Signal")
            
            # Detect R-peaks
            min_peak_distance = int(0.4 * fs)  # 400ms
            peaks, _ = find_peaks(
                ref_signal, 
                height=np.percentile(ref_signal, 95),
                distance=min_peak_distance
            )
            
            # Skip if not enough peaks
            if len(peaks) < 3:
                print(f"Skipping record {record_path}: only {len(peaks)} peaks found")
                continue
                
            # Process each beat (skip first and last)
            for i in range(1, len(peaks)-1):
                beat_dict = {
                    'beat_id': f"{row['ecg_id']}_{i}_{peaks[i]}",
                    'heart_rate': fs * 60 / (peaks[i] - peaks[i-1]),  # in bpm
                    'patient_id': row['ecg_id'],
                    'record_id': record_id,
                    'label': row['label']
                }
                
                # Process each lead for this beat
                for lead_idx in range(12):
                    raw_lead = signals[:, lead_idx].flatten()
                    lead_signal = preprocess_signal(raw_lead, fs)
                    lead_name = LEAD_NAMES[lead_idx]
                    
                    # Extract features and segmentation points
                    features, seg_points = extract_features_one_beat(lead_signal, peaks[i], fs)
                    
                    # Generate segmentation visualization for lead II in selected records
                    if idx in viz_indices and lead_idx == 1 and i == 1:
                        plot_beat_segmentation(
                            lead_signal,
                            peaks[i],
                            seg_points,
                            fs,
                            record_id=record_id,
                            beat_idx=i,
                            lead_name=lead_name
                        )
                    
                    # Add lead-specific features with prefix
                    for feat_name, feat_value in features.items():
                        beat_dict[f"{lead_name}_{feat_name}"] = feat_value
                
                all_features.append(beat_dict)
            
            if idx % 10 == 0:
                print(f"Processed {idx+1}/{len(df_meta)} records - {len(peaks)-2} beats")
                
        except Exception as e:
            print(f"Error processing record {record_path}: {str(e)}")
    
    # Save to CSV
    if all_features:
        df_output = pd.DataFrame(all_features)
        
        # Add beat count summary by label
        beat_counts = df_output['label'].value_counts().to_dict()
        print("\nBeat Count Summary by Label:")
        for label, count in beat_counts.items():
            print(f"{label}: {count} beats")
        
        df_output.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✅ Success! Saved {len(df_output)} beats to {OUTPUT_CSV}")
        print(f"Total features per beat: {len(df_output.columns)}")
        print(f"Visualizations saved to: {PLOT_DIR}")
    else:
        print("❌ No features extracted - check data paths and processing")