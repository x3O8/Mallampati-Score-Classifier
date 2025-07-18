import os
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import cv2

# --- 1. CONFIGURATION ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
N_CLASSES = 4
EPOCHS = 30
TTA_STEPS = 5
CLASS_NAMES = ['1', '2', '3', '4']

# MODIFIED: Paths updated for the Kaggle environment
BASE_DATA_DIR = '/kaggle/input/mallampati-data2/augmented_data'
TRAIN_DIR = os.path.join(BASE_DATA_DIR, 'train')
VAL_DIR = os.path.join(BASE_DATA_DIR, 'val')
TEST_DIR = os.path.join(BASE_DATA_DIR, 'test')

# Model weights will be saved to the writable 'working' directory
MODEL_WEIGHTS_DIR = '/kaggle/working/model_weights/'
os.makedirs(MODEL_WEIGHTS_DIR, exist_ok=True)


# --- 2. DATA PREPARATION ---

def load_data_to_dataframe(data_dir):
    filepaths = []
    labels = []
    class_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    for i, class_name in enumerate(class_dirs):
        class_dir_path = os.path.join(data_dir, class_name)
        for fname in os.listdir(class_dir_path):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepaths.append(os.path.join(class_dir_path, fname))
                labels.append(str(i))
    df = pd.DataFrame({'filepath': filepaths, 'label': labels})
    return df

def get_data_generators():
    """Creates Keras ImageDataGenerators for training, validation, and testing."""
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.15,
        brightness_range=[0.85, 1.15],
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    tta_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=5,
        width_shift_range=0.03,
        height_shift_range=0.03,
        zoom_range=0.03,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    return train_datagen, val_test_datagen, tta_datagen


# --- 3. MODEL ARCHITECTURES ---

def build_ensemble_models():
    """Builds and returns the three uncompiled models of the ensemble."""
    models = []
    
    # --- Model 1: MobileNetV2 ---
    base_mobilenet = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet'
    )
    base_mobilenet.trainable = True
    for layer in base_mobilenet.layers[:-20]:
        layer.trainable = False
        
    inputs_mnet = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = base_mobilenet(inputs_mnet, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap_mobilenet")(x)
    x = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.015))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.6)(x)
    outputs_mnet = tf.keras.layers.Dense(N_CLASSES, activation='softmax')(x)
    model_mobilenet = tf.keras.Model(inputs_mnet, outputs_mnet, name="MobileNetV2_Variant")
    models.append(model_mobilenet)

    # --- Model 2: VGG16 Variant 1 ---
    base_vgg1 = tf.keras.applications.VGG16(
        input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet'
    )
    base_vgg1.trainable = False
    for layer in base_vgg1.layers:
        if layer.name == "block5_conv3":
            layer.trainable = True
            
    inputs_vgg1 = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = base_vgg1(inputs_vgg1, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap_vgg1")(x)
    x = tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.7)(x)
    outputs_vgg1 = tf.keras.layers.Dense(N_CLASSES, activation='softmax')(x)
    model_vgg1 = tf.keras.Model(inputs_vgg1, outputs_vgg1, name="VGG16_Variant_1")
    models.append(model_vgg1)

    # --- Model 3: VGG16 Variant 2 ---
    base_vgg2 = tf.keras.applications.VGG16(
        input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet'
    )
    base_vgg2.trainable = False
    for layer in base_vgg2.layers:
        if layer.name == "block5_conv3":
            layer.trainable = True
            
    inputs_vgg2 = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = base_vgg2(inputs_vgg2, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap_vgg2")(x)
    x = tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.035))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.7)(x)
    outputs_vgg2 = tf.keras.layers.Dense(N_CLASSES, activation='softmax')(x)
    model_vgg2 = tf.keras.Model(inputs_vgg2, outputs_vgg2, name="VGG16_Variant_2")
    models.append(model_vgg2)

    return models


# --- 4. TRAINING PIPELINE ---

def train_final_models(models, train_generator, val_generator):
    histories = []
    learning_rates = [5e-5, 5e-5, 1e-4]

    for i, model in enumerate(models):
        print(f"\n--- Training Final Model: {model.name} ---")
        
        model_path = os.path.join(MODEL_WEIGHTS_DIR, f"{model.name}_final.keras")
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1
        )
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=7, restore_best_weights=True
        )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=4, min_lr=1e-6
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rates[i]),
            loss='categorical_crossentropy', metrics=['accuracy']
        )

        # REMOVED: class_weight is no longer needed as the dataset is balanced
        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=val_generator,
            callbacks=[checkpoint, early_stopping, reduce_lr]
        )
        histories.append(history)
    
    return histories


# --- 5. INFERENCE, 6. GRAD-CAM, 7. EVALUATION ---
def predict_with_ensemble_tta(models, test_df, tta_datagen):
    print("\n--- Performing Inference with Ensemble and TTA ---")
    all_ensemble_preds = []
    for _, row in test_df.iterrows():
        img_path = row['filepath']
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        model_tta_preds = []
        for model in models:
            tta_predictions = []
            tta_generator = tta_datagen.flow(img_array, batch_size=1)
            for _ in range(TTA_STEPS):
                aug_img = next(tta_generator)[0]
                pred = model.predict(np.expand_dims(aug_img, axis=0), verbose=0)
                tta_predictions.append(pred)
            avg_model_pred = np.mean(tta_predictions, axis=0)
            model_tta_preds.append(avg_model_pred)
        final_ensemble_pred = np.mean(model_tta_preds, axis=0)
        all_ensemble_preds.append(final_ensemble_pred)
    return np.squeeze(np.array(all_ensemble_preds), axis=1)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def generate_and_display_gradcam(img_path, models, alpha=0.6):
    print(f"\n--- Visualizing Grad-CAM for: {os.path.basename(img_path)} ---")
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE)
    img_array = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img_array)
    img_array_rescaled = np.expand_dims(img_array, axis=0) / 255.0
    last_conv_layers = ['out_relu', 'block5_conv3', 'block5_conv3']
    heatmaps, confidences = [], []
    for i, model in enumerate(models):
        pred = model.predict(img_array_rescaled, verbose=0)[0]
        pred_idx = np.argmax(pred)
        heatmap = make_gradcam_heatmap(img_array_rescaled, model, last_conv_layers[i], pred_index=pred_idx)
        heatmaps.append(heatmap)
        confidences.append(pred[pred_idx])
        print(f"  - {model.name} Prediction: {CLASS_NAMES[pred_idx]} (Conf: {pred[pred_idx]:.2f})")
        display_superimposed_gradcam(img.copy(), heatmap, f"{model.name} - Pred: {CLASS_NAMES[pred_idx]}", alpha)
    confidences = np.array(confidences) / sum(confidences) if sum(confidences) > 0 else np.ones(len(models)) / len(models)
    ensemble_heatmap = np.zeros_like(heatmaps[0], dtype=np.float32)
    for conf, hmap in zip(confidences, heatmaps):
        ensemble_heatmap += conf * hmap
    final_pred_probs = np.mean([model.predict(img_array_rescaled, verbose=0) for model in models], axis=0)[0]
    final_pred_idx = np.argmax(final_pred_probs)
    print(f"  - Ensemble Final Prediction: {CLASS_NAMES[final_pred_idx]} (Conf: {final_pred_probs[final_pred_idx]:.2f})")
    display_superimposed_gradcam(img.copy(), ensemble_heatmap, f"Ensemble - Pred: {CLASS_NAMES[final_pred_idx]}", alpha)

def display_superimposed_gradcam(img, heatmap, title, alpha=0.6):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

def evaluate_model(y_true, y_pred_probs):
    y_pred_labels = np.argmax(y_pred_probs, axis=1)
    cm = confusion_matrix(y_true, y_pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Ensemble Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred_labels, target_names=CLASS_NAMES, zero_division=0))
    y_true_one_hot = tf.keras.utils.to_categorical(y_true, num_classes=N_CLASSES)
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(N_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_true_one_hot[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure(figsize=(10, 8))
    for i in range(N_CLASSES):
        plt.plot(fpr[i], tpr[i], label=f'{CLASS_NAMES[i]} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Ensemble ROC Curves')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()


# --- MAIN EXECUTION ---

if __name__ == '__main__':
    # REMOVED: Google Drive mounting code is not needed on Kaggle
    
    if not os.path.isdir(TRAIN_DIR) or not os.path.isdir(VAL_DIR) or not os.path.isdir(TEST_DIR):
        print(f"Error: One or more data directories (train, val, test) not found in '{BASE_DATA_DIR}'.")
        print("Please ensure your Kaggle dataset is attached and the path is correct.")
    else:
        ensemble_models = build_ensemble_models()
        train_datagen, val_test_datagen, tta_datagen = get_data_generators()

        print("Creating data generators from directories...")
        train_generator = train_datagen.flow_from_directory(
            TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True
        )
        val_generator = val_test_datagen.flow_from_directory(
            VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
        )
        print(f"Found {train_generator.samples} images for training.")
        print(f"Found {val_generator.samples} images for validation.")
        
        # REMOVED: Class weight calculation is no longer needed
        
        # Check if pre-trained weights exist in the output directory
        all_weights_exist = all(
            os.path.exists(os.path.join(MODEL_WEIGHTS_DIR, f"{model.name}_final.keras")) 
            for model in ensemble_models
        )

        if not all_weights_exist:
            print("\n--- Starting Training Phase ---")
            train_final_models(ensemble_models, train_generator, val_generator)
        
        print("\n--- Loading Final Model Weights ---")
        for model in ensemble_models:
            model_path = os.path.join(MODEL_WEIGHTS_DIR, f"{model.name}_final.keras")
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.load_weights(model_path)
            print(f"Loaded weights for {model.name} from {model_path}")
        
        print("\n--- Preparing Test Set for Evaluation ---")
        test_df = load_data_to_dataframe(TEST_DIR)
        print(f"Found {len(test_df)} images in the test set.")
        
        y_pred_probs = predict_with_ensemble_tta(ensemble_models, test_df, tta_datagen)
        y_true = test_df['label'].astype(int).values
        evaluate_model(y_true, y_pred_probs)

        sample_images = test_df.sample(n=min(3, len(test_df)), random_state=42)
        for _, row in sample_images.iterrows():
            generate_and_display_gradcam(row['filepath'], ensemble_models)