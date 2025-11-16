import numpy as np
import matplotlib.pyplot as plt
import cv2 
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn.decomposition import PCA

image_path = r"C:\Users\Dosya\Desktop\фотки\ba57baeb21bcec06794b92c4ff56bea4.png"
image = Image.open(image_path).convert('RGB')
image_array = np.array(image)

print(f'Shape of the image: {image_array.shape}')

plt.imshow(image_array)
plt.axis('off')
plt.show()

R = image_array[:, :, 0]
G = image_array[:, :, 1]
B = image_array[:, :, 2]

print("R-канал:", R.shape)
print("G-канал:", G.shape)
print("B-канал:", B.shape)

n_components = 30
def applyPCA(x):
    
    pca = PCA(n_components = n_components)
    transformed = pca.fit_transform(x)

    return transformed, pca

R_pca, R_model = applyPCA(R)
G_pca, G_model = applyPCA(G)
B_pca, B_model = applyPCA(B)

print("R after PCA:", R_pca.shape)
print("G after PCA:", G_pca.shape)
print("B after PCA:", B_pca.shape)

RGB_combined = np.hstack([R_pca, G_pca, B_pca])

print(f'The shape of the new array: {RGB_combined}')

n_clusters = 20
randoming = 312

from sklearn.cluster import KMeans
kme = KMeans(n_clusters = n_clusters, random_state = randoming)

clusters = kme.fit_predict(RGB_combined)
print(f'Examples of clusters: {clusters[:20]}')

centers = kme.cluster_centers_
print(f'The centers: {centers}')

RGB_clustered = centers[clusters]
R_clustered = RGB_clustered[:, :n_components]
G_clustered = RGB_clustered[:,n_components:2*n_components]
B_clustered = RGB_clustered[:, 2*n_components:]

R_restored = R_model.inverse_transform(R_clustered)
G_restored = G_model.inverse_transform(G_clustered)

B_restored = B_model.inverse_transform(B_clustered)

restored_image = np.stack([R_restored, G_restored, B_restored], axis = 2)
restored_image = np.clip(restored_image, 0, 255).astype(np.uint8)
plt.imshow(restored_image)
plt.axis('off')
plt.show()
