import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
#importing data from audio_features
data = np.load("Audio Features/audio_dataset.npz")
X_train = data["X_train"]
y_train = data["y_train"]
X_test = data["X_test"]
y_test = data["y_test"]
recording_ids_train = data["recording_ids_train"]
recording_ids_test = data["recording_ids_test"]
validation_recordings = [
    "clap_06.wav",
    "whistle_07.wav",
    "speech_07.wav",
    "snap_12.wav"
]
val_mask = np.isin(recording_ids_train, validation_recordings)
X_val = X_train[val_mask]
y_val = y_train[val_mask]
X_fit = X_train[~val_mask]
y_fit = y_train[~val_mask]
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)
X_fit_float = X_fit.astype(np.float32) / 127.0
X_val_float = X_val.astype(np.float32) / 127.0
X_test_float = X_test.astype(np.float32) / 127.0
#network definintion
model = MLPClassifier(
    hidden_layer_sizes=(8,),
    activation="relu",
    solver="adam",
    max_iter=2000,
    random_state=42
)
model.fit(X_fit_float, y_fit)
train_accuracy = model.score(X_fit_float, y_fit)
val_accuracy = model.score(X_val_float, y_val)
print("\n==== FLOAT MODEL ====")
print("Fit Accuracy:", train_accuracy)
print("Validation Accuracy:", val_accuracy)
y_val_pred = model.predict(X_val_float)
cm = confusion_matrix(y_val, y_val_pred)
print("\n==== VALIDATION CONFUSION MATRIX ====")
print(cm)
print("\n===== SNAP VALIDATION PREDICTIONS =====")
snap_mask = y_val == 3
snap_predictions = y_val_pred[snap_mask]
print("Actual labels:", y_val[snap_mask])
print("Predictions:", snap_predictions)
snap_scores = model.predict_proba(X_val_float[snap_mask])
print("\nSnap class probabilities:")
print(np.round(snap_scores, 3))
#split
print("\n===== FINAL DATA SPLIT =====")
print("Fit:", X_fit.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)
print("\nValidation class counts:")
print("Clap:", np.sum(y_val == 0))
print("Whistle:", np.sum(y_val == 1))
print("Speech:", np.sum(y_val == 2))
print("Snap:", np.sum(y_val == 3))
#model params
print("\n==== MODEL PARAMETERS ====")
print("Layer 1 weights:", model.coefs_[0].shape)
print("Layer 1 biases:", model.intercepts_[0].shape)
print("Layer 2 weights:", model.coefs_[1].shape)
print("Layer 2 biases:", model.intercepts_[1].shape)
#range of params
print("\n===== PARAMETER RANGES =====")
for i, weights in enumerate(model.coefs_):
    print(
        f"Layer {i+1} weights: "
        f"min={weights.min():.6f}, "
        f"max={weights.max():.6f}"
    )
for i, biases in enumerate(model.intercepts_):
    print(
        f"Layer {i+1} biases: "
        f"min={biases.min():.6f}, "
        f"max={biases.max():.6f}"
    )
W1_float = model.coefs_[0]
b1_float = model.intercepts_[0]
W1_SCALE = 32
W1_q = np.round(W1_float * W1_SCALE).astype(np.int8)
b1_q = np.round(
    b1_float * 127 * W1_SCALE
).astype(np.int32)
print("\n====QUANTIZED LAYER 1 ====")
print("W1_q range:", W1_q.min(), W1_q.max())
print("b1_q range:", b1_q.min(), b1_q.max())
#layer 1 
X_fit_int = X_fit.astype(np.int32)
W1_q_int = W1_q.astype(np.int32)
z1_q = X_fit_int @ W1_q_int + b1_q
relu1_q = np.maximum(z1_q, 0)
print("\n===== LAYER 1 INTEGER ACTIVATIONS =====")
print("Accumulator min:", z1_q.min())
print("Accumulator max:", z1_q.max())
print("ReLU max:", relu1_q.max())
print("ReLU percentiles:")
for p in [50, 90, 95, 99, 100]:
    print(f"{p}%: {np.percentile(relu1_q, p):.1f}")
QUANT_SHIFT_1 = 7
hidden_q = relu1_q >> QUANT_SHIFT_1
hidden_q = np.clip(hidden_q, 0, 127).astype(np.int8)
print("\n==== QUANTIZED HIDDEN ACTIVATIONS ====")
print("Range:", hidden_q.min(), hidden_q.max())
print("Percentiles:")
for p in [50, 90, 95, 99, 100]:
    print(f"{p}%: {np.percentile(hidden_q, p):.1f}")
print("Values saturated at 127:", np.sum(hidden_q == 127))
print("Total hidden activations:", hidden_q.size)
#layer 2
W2_float = model.coefs_[1]
b2_float = model.intercepts_[1]
W2_SCALE = 32
HIDDEN_SCALE = (127 * W1_SCALE) / (2 ** QUANT_SHIFT_1)
W2_q = np.round(W2_float * W2_SCALE).astype(np.int8)
b2_q = np.round(
    b2_float * HIDDEN_SCALE * W2_SCALE
).astype(np.int32)
print("\n===== QUANTIZED LAYER 2 =====")
print("Hidden scale:", HIDDEN_SCALE)
print("W2_q range:", W2_q.min(), W2_q.max())
print("b2_q range:", b2_q.min(), b2_q.max())
def quantized_inference(X):
    X_int = X.astype(np.int32)
    #layer 1
    z1 = X_int @ W1_q.astype(np.int32) + b1_q
    relu1 = np.maximum(z1, 0)
    hidden = relu1 >> QUANT_SHIFT_1
    hidden = np.clip(hidden, 0, 127).astype(np.int32)
    #layer 2
    logits = hidden @ W2_q.astype(np.int32) + b2_q
    predictions = np.argmax(logits, axis=1)
    return predictions, logits
y_val_q_pred, val_logits_q = quantized_inference(X_val)
quant_val_accuracy = np.mean(y_val_q_pred == y_val)
print("\n==== QUANTIZED MODEL ====")
print("Validation Accuracy:", quant_val_accuracy)
print("\n==== QUANTIZED VALIDATION CONFUSION MATRIX ====")
print(confusion_matrix(y_val, y_val_q_pred))
float_val_pred = model.predict(X_val_float)
agreement = np.mean(y_val_q_pred == float_val_pred)
print("\nFloat / Quantized agreement:", agreement)
# float test eval
float_test_accuracy = model.score(X_test_float, y_test)
float_test_pred = model.predict(X_test_float)
print("\n===== FINAL FLOAT TEST =====")
print("Test Accuracy:", float_test_accuracy)
print("Confusion Matrix:")
print(confusion_matrix(y_test, float_test_pred))
# quant test eval
y_test_q_pred, test_logits_q = quantized_inference(X_test)
quant_test_accuracy = np.mean(y_test_q_pred == y_test)
print("\n===== FINAL QUANTIZED TEST =====")
print("Test Accuracy:", quant_test_accuracy)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_q_pred))
# float & int implement
test_agreement = np.mean(y_test_q_pred == float_test_pred)
print("\nFloat / Quantized Test Agreement:", test_agreement)
W1_hw = W1_q.T
W2_hw = W2_q.T
print("\n==== HARDWARE PARAMETER SHAPES ====")
print("W1_hw:", W1_hw.shape)
print("b1_q:", b1_q.shape)
print("W2_hw:", W2_hw.shape)
print("b2_q:", b2_q.shape)
print("\n===== LAYER 1 WEIGHTS =====")
print(W1_hw)
print("\n===== LAYER 1 BIASES =====")
print(b1_q)
print("\n===== LAYER 2 WEIGHTS =====")
print(W2_hw)
print("\n===== LAYER 2 BIASES =====")
print(b2_q)
sample = X_test[0].astype(np.int32)
z1 = sample @ W1_q.astype(np.int32) + b1_q
relu1 = np.maximum(z1, 0)
hidden = relu1 >> 7
hidden = np.clip(hidden, 0, 127).astype(np.int32)
logits = hidden @ W2_q.astype(np.int32) + b2_q
prediction = np.argmax(logits)
print("\n--- RTL TEST VECTOR ---")
print("INPUT:")
print(sample.tolist())
print("HIDDEN:")
print(hidden.tolist())
print("LOGITS:")
print(logits.tolist())
print("PREDICTION:")
print(prediction)
print("TRUE LABEL:")
print(y_test[0])
print("\n--- RTL TEST VECTORS: ONE PER CLASS ---")
class_names = ["CLAP", "WHISTLE", "SPEECH", "SNAP"]
predictions, all_logits = quantized_inference(X_test)
for target_class in range(4):
    #finding a correct class example
    indices = np.where(
        (y_test == target_class) &
        (predictions == target_class)
    )[0]
    idx = indices[0]
    sample = X_test[idx].astype(np.int32)
    z1 = sample @ W1_q.astype(np.int32) + b1_q
    relu1 = np.maximum(z1, 0)
    hidden = relu1 >> 7
    hidden = np.clip(hidden, 0, 127).astype(np.int32)
    logits = hidden @ W2_q.astype(np.int32) + b2_q
    prediction = np.argmax(logits)
    print(f"\n{class_names[target_class]}")
    print("INPUT:")
    print(sample.tolist())
    print("HIDDEN:")
    print(hidden.tolist())
    print("LOGITS:")
    print(logits.tolist())
    print("PREDICTION:")
    print(prediction)