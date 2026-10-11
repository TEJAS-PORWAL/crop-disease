# 🎓 Viva Questions & Answers — Crop Disease Detection System

A practice question bank for defending this project. Grouped by topic, from
basics to deeper follow-ups an examiner is likely to ask.

---

## 1. Project overview

**Q1. In one sentence, what does your project do?**
It classifies a crop leaf image into one of several disease (or healthy)
categories using a deep-learning model, and then shows the predicted disease
with a confidence score and recommended treatment steps.

**Q2. Why is this problem important?**
Crop diseases cause major yield loss, especially for small farmers who may not
have quick access to an expert. An instant, image-based diagnosis helps catch
problems early and suggests practical management steps, reducing losses.

**Q3. What are the main components of your system?**
Five layers: a **data layer** (download/generate + load data), a **model layer**
(MobileNetV2 + classifier head), a **training layer** (two-phase training and
evaluation), an **inference layer** (CLI predictor and Streamlit web app), and a
**domain layer** (`treatments.py`) that maps each disease to advice.

**Q4. What technologies/libraries did you use and why?**
- **TensorFlow/Keras** — building, training, and loading the model.
- **MobileNetV2** — a lightweight pretrained CNN for transfer learning.
- **Streamlit** — fast way to build an interactive web UI in pure Python.
- **scikit-learn** — classification report and confusion matrix.
- **Matplotlib/seaborn** — plotting training curves and the confusion matrix.
- **Pillow/NumPy** — image loading and array handling.

---

## 2. Dataset

**Q5. Which dataset did you use?**
The **PlantVillage** dataset from Kaggle, which has 38 classes of healthy and
diseased leaf images organized one folder per class (e.g. `Tomato___Early_blight`).

**Q6. What is the folder structure your code expects?**
An ImageFolder-style layout: `data/train/<ClassName>/...` and
`data/val/<ClassName>/...`, with one subfolder per class. Keras infers the
labels from the folder names.

**Q7. How do you split the data into training and validation sets?**
`download_data.py` does a **per-class split** (default 20% validation) so every
class appears in both train and val. The split is seeded for reproducibility.

**Q8. Why a per-class split instead of a single random split?**
To guarantee each class is represented in both sets. A plain random split could,
for a small or imbalanced class, put all of its images into one side.

**Q9. What is the synthetic data generator for?**
`generate_synthetic_data.py` creates toy images (per-class colors + spots) so the
entire pipeline — train, evaluate, predict, app — can run end-to-end without the
large download. It is only for verifying the code runs; accuracy on it is not
meaningful.

**Q10. What image size do you use and why 160×160?**
160×160×3. It is a good balance: large enough to keep leaf detail, small enough
to train quickly, and a standard input size MobileNetV2 handles well.

---

## 3. Transfer learning & model

**Q11. What is transfer learning?**
Reusing a model pretrained on a large dataset (ImageNet) and adapting it to a new
task. The pretrained layers already know general visual features (edges, textures,
shapes), so we only need to train a small part on our data — faster training and
better accuracy on small datasets.

**Q12. Why MobileNetV2 specifically?**
It is lightweight and efficient (uses depthwise-separable convolutions and
inverted residual blocks), so it trains and runs fast — ideal for a web app and
for deployment on modest hardware — while still being accurate.

**Q13. Describe your model architecture layer by layer.**
Input (160×160×3) → data augmentation → MobileNetV2 preprocessing (scales to
[-1, 1]) → MobileNetV2 base (no top, ImageNet weights) → GlobalAveragePooling2D →
Dropout(0.2) → Dense(num_classes, softmax).

**Q14. Why GlobalAveragePooling2D instead of Flatten?**
It averages each feature map to a single value, drastically reducing parameters,
which lowers overfitting and model size compared to flattening + large dense layers.

**Q15. What does the Dropout layer do?**
During training it randomly zeroes 20% of activations, which prevents the network
from relying too heavily on specific neurons and reduces overfitting.

**Q16. Why softmax on the output layer?**
Softmax turns the raw scores into a probability distribution over classes that
sums to 1, so we can read off the predicted class and its confidence.

**Q17. Why keep preprocessing and augmentation inside the model graph?**
So training and inference feed the exact same raw [0, 255] input. The scaling and
augmentation are layers in the model, which means there is no chance of a
mismatch between how data is prepared at train time vs. predict time.

**Q18. Does augmentation run during inference?**
No. Augmentation layers are active only when the model runs in training mode
(`training=True` during `fit`). At predict time they pass the image through
unchanged.

---

## 4. Training strategy

**Q19. Explain your two-phase training.**
- **Phase 1 (feature extraction):** freeze the MobileNetV2 base and train only
  the new classifier head. The pretrained features are reused as-is.
- **Phase 2 (fine-tuning):** unfreeze the top layers of the base and train again
  at a very low learning rate so the pretrained features adapt slightly to leaf
  images without being destroyed.

**Q20. Why train the head first before fine-tuning?**
The head starts with random weights. If you unfroze the base immediately, the
large initial gradients from the untrained head would corrupt the valuable
pretrained weights. Training the head first stabilizes it, then fine-tuning makes
small, safe adjustments.

**Q21. Why is the fine-tuning learning rate so much lower (1e-5 vs 1e-3)?**
To make only tiny updates to the pretrained weights. A high learning rate during
fine-tuning would overwrite the useful ImageNet features and hurt performance.

**Q22. Why do you keep BatchNormalization layers frozen during fine-tuning?**
BatchNorm layers hold running statistics from ImageNet. Updating them on a small
batch of leaf images can destabilize training, so they are kept frozen for stability.

**Q23. What loss function did you use and why?**
`sparse_categorical_crossentropy` — the standard loss for multi-class, single-label
classification where labels are integers (not one-hot). "Sparse" means we pass
integer labels directly instead of one-hot vectors.

**Q24. Which optimizer did you use?**
Adam, which adapts the learning rate per parameter and generally converges
quickly and reliably with little tuning.

**Q25. What is EarlyStopping and why use it?**
It monitors validation loss and stops training when it stops improving (patience
of 5 epochs), restoring the best weights. This prevents overfitting and wasted
epochs.

**Q26. What is data augmentation and which transforms did you apply?**
Randomly transforming training images to simulate more variety: random flips,
rotation, zoom, and contrast. It improves generalization, especially on small
datasets, by exposing the model to varied versions of each leaf.

**Q27. How do you handle reproducibility?**
A fixed seed (42) is set for TensorFlow, the dataset shuffling, and augmentation,
so runs are repeatable.

---

## 5. Evaluation

**Q28. How do you evaluate the model?**
On the held-out validation set: overall accuracy, a scikit-learn classification
report (precision/recall/F1 per class), and a confusion matrix heatmap.

**Q29. What is a confusion matrix and what does it tell you?**
A table of true vs. predicted classes. The diagonal is correct predictions;
off-diagonal cells show which classes get confused with which. It reveals whether
errors are spread out or concentrated between similar-looking diseases.

**Q30. What's the difference between precision and recall?**
- **Precision:** of all images predicted as class X, how many were actually X.
- **Recall:** of all images that are truly class X, how many did we catch.
F1 is their harmonic mean, useful when you care about both.

**Q31. Why is accuracy alone not enough, especially with imbalanced classes?**
With imbalance, a model can score high accuracy by favoring the majority class
while failing on rare ones. Per-class precision/recall and the confusion matrix
expose that; accuracy hides it.

---

## 6. Inference & application

**Q32. Walk me through what happens when a user uploads an image in the app.**
The image is converted to RGB and resized to 160×160, turned into a batch, and
passed to the cached model. The model returns class probabilities; we take the
argmax as the prediction, look up its treatment, and render a result banner,
treatment card, and top-5 probabilities.

**Q33. How do you keep predicted labels aligned with training?**
At training time the class-name order is saved to `class_names.json`. Prediction
and the app reload that file, so the index the model outputs always maps to the
correct class name — even if folders are later reordered.

**Q34. What is the uncertainty threshold for?**
If the top confidence is below 50%, the app flags the result as "uncertain" and
tells the user to verify, rather than presenting a shaky guess as fact.

**Q35. Why cache the model in the app (`@st.cache_resource`)?**
Streamlit reruns the script on every interaction. Caching loads the model once
and reuses it, so predictions stay fast instead of reloading the model each time.

**Q36. How does the CLI predictor differ from the web app?**
Same core inference, different interface. `predict.py` takes an `--image` path and
prints the top-k predictions plus treatment to the terminal; `app.py` provides a
drag-and-drop web UI with styled results. Both share the model and `treatments.py`.

**Q37. Where does the treatment advice come from?**
`treatments.py` is a hand-curated dictionary mapping each PlantVillage class name
to a summary, cause, and step-by-step guidance, with a safe default for unknown
classes. It is general educational guidance, not a substitute for an expert.

---

## 7. Design & engineering decisions

**Q38. Why is everything driven by a single `config.py`?**
It is the single source of truth for paths and hyperparameters. Changing the
image size or learning rate in one place updates the whole project, and paths are
resolved relative to the file so it runs from any working directory.

**Q39. Why save the model in `.keras` format?**
It is the modern native Keras format that stores the full model (architecture +
weights + preprocessing layers) in one file, so inference needs no rebuild of the
architecture.

**Q40. How would this scale to all 38 PlantVillage classes?**
The code already supports it — just download all classes instead of filtering by
crop. The model's final Dense layer sizes itself to the number of classes found,
so no code change is needed; mainly more training time (ideally on a GPU).

**Q41. What are the limitations of your project?**
It only recognizes the classes it was trained on; an unrelated or very different
image can give a confident but wrong result. The treatment advice is generic, and
the synthetic data is only for testing. Real-world lighting/background variation
also affects accuracy.

**Q42. How could you improve it?**
Train on the full dataset, add more crops, add an "unknown/not-a-leaf" rejection
path, try a larger backbone (EfficientNet) for accuracy, apply class weighting for
imbalance, add Grad-CAM visualizations to explain predictions, and deploy with a
proper API/containerization.

---

## 8. Core ML concepts (quick-fire)

**Q43. What is overfitting and how did you combat it?**
When a model memorizes training data and fails to generalize. Combated with
dropout, data augmentation, transfer learning, and early stopping.

**Q44. What is a CNN and why is it suited to images?**
A Convolutional Neural Network uses filters that slide over the image to detect
spatial patterns (edges → textures → shapes). It is parameter-efficient and
translation-aware, making it ideal for image recognition.

**Q45. What is the difference between a parameter and a hyperparameter?**
Parameters (weights) are learned during training. Hyperparameters (learning rate,
batch size, epochs, dropout rate) are set by you before/around training and
control how learning happens.

**Q46. What does the batch size control?**
How many images are processed before the weights update. Larger batches give
smoother, more stable gradient estimates but use more memory; smaller batches
update more often and add noise that can help generalization.

**Q47. What is an epoch?**
One full pass over the entire training dataset.

**Q48. Why normalize/scale pixel values?**
Neural networks train better when inputs are in a small, consistent range.
MobileNetV2 expects values scaled to [-1, 1], which the preprocessing layer
handles inside the model.
