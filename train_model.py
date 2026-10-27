import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    RandomFlip,
    RandomRotation,
    RandomTranslation,
)
from tensorflow.keras.models import load_model


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(PROJECT_DIR, "train")
MODEL_PATH = os.path.join(PROJECT_DIR, "emotion_model.h5")
IMAGE_SIZE = (48, 48)
BATCH_SIZE = 128
TRAIN_BATCHES = 40
VALIDATION_BATCHES = 15
EPOCHS = 8


class BalancedAccuracy(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name="balanced_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.confusion = self.add_weight(
            name="confusion",
            shape=(num_classes, num_classes),
            initializer="zeros",
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        del sample_weight
        labels = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        predicted = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        matrix = tf.math.confusion_matrix(
            labels, predicted, num_classes=self.num_classes, dtype=self.dtype
        )
        self.confusion.assign_add(matrix)

    def result(self):
        class_totals = tf.reduce_sum(self.confusion, axis=1)
        recalls = tf.math.divide_no_nan(
            tf.linalg.diag_part(self.confusion), class_totals
        )
        present_classes = tf.cast(class_totals > 0, self.dtype)
        return tf.math.divide_no_nan(
            tf.reduce_sum(recalls), tf.reduce_sum(present_classes)
        )

    def reset_state(self):
        self.confusion.assign(tf.zeros_like(self.confusion))


image_extensions = (".bmp", ".gif", ".jpeg", ".jpg", ".png")
class_names = sorted(
    name for name in os.listdir(TRAIN_DIR)
    if os.path.isdir(os.path.join(TRAIN_DIR, name))
)
if len(class_names) != 7:
    raise ValueError(f"Expected 7 emotion folders, found {len(class_names)}: {class_names}")

train_paths = []
train_labels = []
validation_by_class = {}
training_counts = {}
random = np.random.default_rng(42)
for class_index, class_name in enumerate(class_names):
    class_dir = os.path.join(TRAIN_DIR, class_name)
    paths = [
        os.path.join(class_dir, filename)
        for filename in os.listdir(class_dir)
        if filename.lower().endswith(image_extensions)
    ]
    if len(paths) < 2:
        raise ValueError(f"Emotion class {class_name!r} must contain at least two images")

    random.shuffle(paths)
    validation_size = max(1, int(round(len(paths) * 0.2)))
    validation_by_class[class_index] = paths[:validation_size]
    class_train_paths = paths[validation_size:]
    training_counts[class_index] = len(class_train_paths)
    train_paths.extend(class_train_paths)
    train_labels.extend([class_index] * len(class_train_paths))

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Pretrained emotion model not found at {MODEL_PATH}; train or restore it first."
    )

base_model = load_model(MODEL_PATH, compile=False)
if base_model.input_shape[1:] != (48, 48, 1) or base_model.output_shape[-1] != len(class_names):
    raise ValueError(
        "The existing model must accept 48x48 grayscale faces and output "
        f"{len(class_names)} emotion classes; found input {base_model.input_shape} "
        f"and output {base_model.output_shape}."
    )

validation_paths = []
validation_labels = []
validation_total = sum(map(len, validation_by_class.values()))
validation_sample_size = VALIDATION_BATCHES * BATCH_SIZE
for class_index, paths in validation_by_class.items():
    sample_size = max(
        1,
        round(validation_sample_size * len(paths) / validation_total),
    )
    sample_size = min(sample_size, len(paths))
    selected = random.choice(paths, size=sample_size, replace=False)
    validation_paths.extend(selected.tolist())
    validation_labels.extend([class_index] * sample_size)


def load_image(path, label):
    image = tf.io.decode_image(
        tf.io.read_file(path),
        channels=1,
        expand_animations=False,
    )
    image.set_shape([None, None, 1])
    image = tf.image.resize(image, IMAGE_SIZE)
    return image / 255.0, label


train_dataset = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .shuffle(
        buffer_size=len(train_paths),
        seed=42,
        reshuffle_each_iteration=True,
    )
    .take(TRAIN_BATCHES * BATCH_SIZE)
    .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
validation_dataset = (
    tf.data.Dataset.from_tensor_slices((validation_paths, validation_labels))
    .shuffle(
        buffer_size=len(validation_paths),
        seed=43,
        reshuffle_each_iteration=False,
    )
    .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

validation_images = []
validation_targets = []
for images, labels in validation_dataset:
    validation_images.append(images.numpy())
    validation_targets.append(labels.numpy())
validation_images = np.concatenate(validation_images)
validation_targets = np.concatenate(validation_targets)


def get_validation_scores(model):
    predictions = np.argmax(
        model.predict(validation_images, batch_size=BATCH_SIZE, verbose=0),
        axis=1,
    )
    accuracy = float(np.mean(predictions == validation_targets))
    recalls = [
        float(np.mean(predictions[validation_targets == index] == index))
        for index in range(len(class_names))
    ]
    return accuracy, float(np.mean(recalls))


baseline_accuracy, baseline_balanced_accuracy = get_validation_scores(base_model)

largest_class = max(training_counts.values())
class_weights = {
    index: float(np.sqrt(largest_class / count))
    for index, count in training_counts.items()
}

augmentation = Sequential(
    [
        RandomFlip("horizontal"),
        RandomRotation(0.04),
        RandomTranslation(0.04, 0.04),
    ],
    name="augmentation",
)
inputs = Input(shape=(48, 48, 1))
outputs = base_model(augmentation(inputs))
model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=[
        "accuracy",
        BalancedAccuracy(num_classes=len(class_names)),
    ],
)

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_balanced_accuracy",
        mode="max",
        patience=4,
        restore_best_weights=True,
        verbose=1,
    ),
]

print("Emotion classes:", class_names)
print(
    f"Baseline validation accuracy: {baseline_accuracy:.4f}; "
    f"balanced accuracy: {baseline_balanced_accuracy:.4f}"
)
print(
    f"Fine-tuning on up to {TRAIN_BATCHES * BATCH_SIZE} randomly sampled images "
    f"and validating on {len(validation_targets)} stratified images each epoch."
)
model.fit(
    train_dataset,
    validation_data=(validation_images, validation_targets),
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
    shuffle=False,
)

candidate_accuracy, candidate_balanced_accuracy = get_validation_scores(model)
print(
    f"Fine-tuned validation accuracy: {candidate_accuracy:.4f}; "
    f"balanced accuracy: {candidate_balanced_accuracy:.4f}"
)

if (
    candidate_balanced_accuracy > baseline_balanced_accuracy
    and candidate_accuracy >= baseline_accuracy - 0.03
):
    candidate_path = os.path.join(PROJECT_DIR, "emotion_model_candidate.h5")
    if os.path.exists(candidate_path):
        raise FileExistsError(f"Remove or inspect the existing candidate model: {candidate_path}")
    model.save(candidate_path)
    os.replace(candidate_path, MODEL_PATH)
    print(f"Promoted improved model to {MODEL_PATH}")
else:
    print("Keeping the existing model because validation did not improve safely.")
