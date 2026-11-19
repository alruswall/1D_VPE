import numpy as np
import matplotlib.pyplot as plt
#code for replicating HIF features in CSV form

#Buildup/shoulder 
def build_buildup_shoulder():
    # Define the time points for the buildup/shoulder phase
    #20 points per millisecond, 1 second total
    time_points = np.linspace(0, 1, 2000)  #

    # Define the amplitude of the buildup/shoulder phase
    amplitude = np.zeros_like(time_points)
    
    for i in range(len(time_points)):

# create a sinusodial waveform that 
        if time_points[i] < 0.05:
            amplitude[i] = (time_points[i]) * np.cos(2 * np.pi * time_points[i] / 0.005)
        else:
            amplitude[i] = 0.05 * np.cos(2 * np.pi * time_points[i] / 0.005)


    return time_points, amplitude

def build_intermittence():
    # Define the time points for the intermittence phase
    time_points = np.linspace(0, 1, 2000)  # 2000 time points from 0 to 1 second

    # Define the amplitude of the intermittence phase
    amplitude = np.zeros_like(time_points)
    
    # Create an intermittent waveform
    for i in range(len(time_points)):
        if (i // 100) % 2 == 0:
            amplitude[i] = 0.05 * np.cos(2 * np.pi * time_points[i] / 0.005)
        else:
            amplitude[i] = 0.0  # Low phase

    return time_points, amplitude

def build_randomness():
    # Define the time points for the randomness phase
    time_points = np.linspace(0, 1, 2000)  # 2000 time points from 0 to 1 second

    # Define the amplitude of the asymmetry phase
    amplitude = np.zeros_like(time_points)
    
    # Create a sinusodal waveform with random amplitude
    for i in range(len(time_points)):
        amplitude[i] = np.random.uniform(-0.05, 0.05) * np.cos(2 * np.pi * time_points[i] / 0.005)

    return time_points, amplitude

# building the HIF features and saving them to CSV files
def save_hif_features_to_csv():
    # Build the buildup/shoulder feature
    time_buildup, amplitude_buildup = build_buildup_shoulder()
    np.savetxt('buildup_shoulder.csv', np.column_stack((time_buildup, amplitude_buildup)), delimiter=',', header='Time,Amplitude', comments='')

    # Build the intermittence feature
    time_intermittence, amplitude_intermittence = build_intermittence()
    np.savetxt('intermittence.csv', np.column_stack((time_intermittence, amplitude_intermittence)), delimiter=',', header='Time,Amplitude', comments='')

    # Build the randomness feature
    time_randomness, amplitude_randomness = build_randomness()
    np.savetxt("randomness.csv", np.column_stack((time_randomness, amplitude_randomness)), delimiter=',', header='Time,Amplitude', comments='')

    # Plot the features
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(time_buildup, amplitude_buildup)
    plt.axis([0, 1, -0.1, 0.1])  # Set x-axis from 0 to 1 second and y-axis from -0.1 to 1.1
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Buildup/Shoulder simulation')

    plt.subplot(3, 1, 2)
    plt.plot(time_intermittence, amplitude_intermittence)
    plt.axis([0, 1, -0.1, 0.1])  # Set x-axis from 0 to 1 second and y-axis from -0.1 to 1.1
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Intermittence simulation')

    plt.subplot(3, 1, 3)
    plt.plot(time_randomness, amplitude_randomness)
    plt.axis([0, 1, -0.1, 0.1])  # Set x-axis from 0 to 1 second and y-axis from -0.1 to 1.1
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Randomness simulation')

    plt.tight_layout()
    plt.show()

#run the function to save HIF features to CSV
save_hif_features_to_csv()
