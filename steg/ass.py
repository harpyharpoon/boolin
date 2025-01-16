def extract_time_segment(data, sample_rate, start_time, end_time):
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    return data[start_sample:end_sample]

# Extract specific time segment (e.g., 10s to 20s)
time_segment = extract_time_segment(filtered_audio, sample_rate, start_time=10, end_time=20)
write("time_segment.wav", sample_rate, time_segment.astype(np.int16))
