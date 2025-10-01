from datasets import load_dataset
import matplotlib.pyplot as plt

data = load_dataset('derek-thomas/ScienceQA', split='test') # choose the test set
example = data[5]
print("Question:", example["question"], "\n")
print("Choices:", example["choices"])
print("Text context:", example["hint"])

# Display the image using matplotlib
image = example["image"]
print(image.format)
plt.imshow(example["image"])
plt.axis('off')  # Turn off axis numbers
plt.show()

print("Answer:", example["choices"][example["answer"]], "\n")
print("Lecture:", example["lecture"], "\n")
print("Solution:", example["solution"])