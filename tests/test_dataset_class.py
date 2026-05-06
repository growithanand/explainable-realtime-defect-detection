import matplotlib.pyplot as plt

from src.data.dataset import MVTecBinaryDataset


dataset = MVTecBinaryDataset(
    root_dir="data/mvtec_ad/bottle",
    split="test",
    transform=None
)

print("Dataset size:", len(dataset))

image, label = dataset[0]

print("Image type:", type(image))
print("Image size:", image.size)
print("Label:", label)

plt.imshow(image)
plt.title(f"Label: {label}")
plt.axis("off")
plt.show()