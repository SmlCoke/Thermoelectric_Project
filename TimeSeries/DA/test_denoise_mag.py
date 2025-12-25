import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_mock_data(length=100):
    """Generate mock data with noise and artificial outliers"""
    np.random.seed(42)
    x = np.linspace(0, 4 * np.pi, length)
    # Base signal: Sine wave
    y = np.sin(x)
    # Add random noise
    noise = np.random.normal(0, 0.1, length)
    data = y + noise
    
    # Inject artificial outliers
    # 1. Upward spike
    data[25] = 2.5 
    # 2. Downward spike
    data[60] = -2.5
    # 3. Smaller continuous spike
    data[80] = 1.8
    
    return pd.Series(data)

def old_method_mean_replacement(series, window=5, threshold=2.0):
    """
    Old Method: Mean Replacement
    If |x - μ| > nσ, then x = μ
    """
    s = series.copy()
    rolling = s.rolling(window=window, center=True, min_periods=1)
    mean = rolling.mean()
    std = rolling.std().replace(0, 1e-10)
    
    # Identify outliers
    outliers = np.abs(s - mean) > (threshold * std)
    
    # Replace with mean (destroys directionality)
    s[outliers] = mean[outliers]
    return s, mean, std

def new_method_clamping(series, window=5, threshold=2.0):
    """
    New Method: Boundary Clamping
    If x > μ + nσ, then x = μ + nσ
    If x < μ - nσ, then x = μ - nσ
    """
    s = series.copy()
    rolling = s.rolling(window=window, center=True, min_periods=1)
    mean = rolling.mean()
    std = rolling.std().replace(0, 1e-10)
    
    upper = mean + threshold * std
    lower = mean - threshold * std
    
    # Identify and clamp outliers
    mask_high = s > upper
    mask_low = s < lower
    
    # Replace with boundary values (preserves directionality)
    s[mask_high] = upper[mask_high]
    s[mask_low] = lower[mask_low]
    
    return s, upper, lower

def plot_comparison():
    # 1. Prepare data
    raw_data = generate_mock_data()
    window_size = 5
    z_threshold = 1.2 # Set lower for demonstration
    
    # 2. Apply both methods
    data_old, mean_old, _ = old_method_mean_replacement(raw_data, window_size, z_threshold)
    data_new, upper_new, lower_new = new_method_clamping(raw_data, window_size, z_threshold)
    
    # 3. Plotting
    plt.figure(figsize=(14, 8))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Plot raw data
    plt.plot(raw_data, color='black', linestyle='-', linewidth=1.5, label='Raw Data', zorder=1)
    plt.scatter(raw_data.index, raw_data, color='black', s=10, alpha=0.5)
    
    # Plot old method result
    plt.plot(data_old, color='tab:red', linestyle='--', linewidth=2, alpha=0.7, 
             label='Old Method: Mean Replacement')
    
    # Plot new method result
    plt.plot(data_new, color='tab:green', linestyle='-', linewidth=2, 
             label='New Method: Boundary Clamping')
    
    # Plot envelope (boundaries for new method)
    plt.plot(upper_new, color='green', linestyle=':', alpha=0.3, linewidth=1, label='Upper Bound (μ + nσ)')
    plt.plot(lower_new, color='green', linestyle=':', alpha=0.3, linewidth=1, label='Lower Bound (μ - nσ)')
    plt.fill_between(raw_data.index, lower_new, upper_new, color='green', alpha=0.05)

    # Annotate key areas
    # Annotate point 25 (Upward spike)
    plt.annotate('Outlier 1\n(Upward Spike)', xy=(25, 2.5), xytext=(25, 3.2),
                 arrowprops=dict(facecolor='black', shrink=0.05), ha='center')
    
    # Circle to highlight difference (optional)
    circle = patches.Circle((25, 1.0), radius=3, fill=False, color='blue', linestyle='--')
    # plt.gca().add_patch(circle) 

    plt.title(f'Denoising Method Comparison: Mean Replacement vs. Clamping\n(Window={window_size}, Threshold={z_threshold}σ)', fontsize=14)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend(loc='best', frameon=True, shadow=True)
    
    # Inset plot (Zoom In)
    # Create inset axes
    ax_ins = plt.axes([0.65, 0.6, 0.2, 0.2]) # [left, bottom, width, height]
    ax_ins.plot(raw_data[20:30], color='black', marker='o', markersize=4)
    ax_ins.plot(data_old[20:30], color='tab:red', linestyle='--', marker='x', label='Old')
    ax_ins.plot(data_new[20:30], color='tab:green', linestyle='-', marker='.', label='New')
    ax_ins.set_title('Zoom In')
    ax_ins.set_xticks([])
    
    plt.tight_layout()
    
    # Save or show
    output_file = 'denoise_comparison_3.svg'
    plt.savefig(output_file)
    print(f"Chart saved to: {output_file}")
    # plt.show()

if __name__ == "__main__":
    plot_comparison()