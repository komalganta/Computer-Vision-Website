import cv2
import numpy as np
from matplotlib import pyplot as plt

IMAGE_PATH = 'dataset/task2_source.jpg'
KERNEL_SIZE = 21
SIGMA = 5
SNR = 0.005 

def recover_channel(channel, ksize, sigma):
    # 1. Convert to float
    channel = channel.astype(np.float32)

    # 2. Generate Gaussian Kernel
    k_1d = cv2.getGaussianKernel(ksize, sigma)
    k_2d = np.outer(k_1d, k_1d)

    # 3. Pad kernel
    padded_k = np.zeros_like(channel)
    kh, kw = k_2d.shape
    padded_k[:kh, :kw] = k_2d

    # 4. FFT
    img_fft = np.fft.fft2(channel)
    kernel_fft = np.fft.fft2(padded_k)

    # 5. Deconvolution 
    kernel_conj = np.conj(kernel_fft)
    numerator = img_fft * kernel_conj
    denominator = (np.abs(kernel_fft) ** 2) + SNR
    
    restored_fft = numerator / denominator
    
    #Inverse FFT
    restored = np.abs(np.fft.ifft2(restored_fft))

    max_val = np.percentile(restored, 99)
    min_val = np.min(restored)
    restored = np.clip(restored, min_val, max_val)
    restored = cv2.normalize(restored, None, 0, 255, cv2.NORM_MINMAX)

    restored = restored.astype(np.uint8)
    sharpen_filter = np.array([[-1, -1, -1], 
                               [-1,  9, -1], 
                               [-1, -1, -1]])
    restored = cv2.filter2D(restored, -1, sharpen_filter)
    
    return restored

img = cv2.imread(IMAGE_PATH)
if img is None:
    print(f"Error: {IMAGE_PATH} not found.")
    exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Blur
blurred = cv2.GaussianBlur(img_rgb, (KERNEL_SIZE, KERNEL_SIZE), SIGMA)

# Restore
restored_channels = []
for i in range(3):
    restored_channels.append(recover_channel(blurred[:,:,i], KERNEL_SIZE, SIGMA))

restored = np.stack(restored_channels, axis=2)

restored = cv2.convertScaleAbs(restored, alpha=1.2, beta=-10)

# Display
plt.figure(figsize=(15, 6))
plt.subplot(1, 3, 1); plt.imshow(img_rgb); plt.title("Original")
plt.subplot(1, 3, 2); plt.imshow(blurred); plt.title("Blurred")
plt.subplot(1, 3, 3); plt.imshow(restored); plt.title("Restored (Ultra Sharp)")
plt.show()