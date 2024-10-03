import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import cm
from scipy.interpolate import griddata
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Function to evaluate models
def evaluate_model(model, X_test, y_test, model_name=''):
    # Evaluate the model on the test set to get loss and accuracy
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)  # Using model's evaluate method
    
    print(f'Test Loss: {test_loss:.4f} - Test Accuracy: {test_acc:.4f}')
    
    # Predict probabilities using the model
    y_pred_probs = model.predict(X_test)
    
    test_loss, test_acc = model.evaluate(X_test, y_test)
    
    # Convert probabilities to binary predictions
    y_pred_classes = np.argmax(y_pred_probs, axis=1)  # Convert softmax output to binary class labels

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_classes)
    precision = precision_score(y_test, y_pred_classes)
    recall = recall_score(y_test, y_pred_classes)
    f1 = f1_score(y_test, y_pred_classes)
    
    # Print classification report and confusion matrix
    report = classification_report(y_test, y_pred_classes)
    confusion_mat = confusion_matrix(y_test, y_pred_classes)
    
    print('Accuracy: ', accuracy)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F1 Score: ', f1)
    print('\nClassification Report:\n', report)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, fmt='g', cmap='coolwarm', xticklabels=['Diabetes', 'No Diabetes'], yticklabels=['Diabetes', 'No Diabetes'], linewidths=1, linecolor='black', cbar_kws={'shrink': 0.8})
    plt.title(f'Confusion Matrix ({model_name})', fontsize=15, fontweight='bold', pad=20)
    plt.xlabel('Predicted labels', fontsize=12, labelpad=10)
    plt.ylabel('True labels', fontsize=12, labelpad=10)
    plt.show()
    
    return accuracy


# Plotting function for learning curves
def plot_learning_curves(history, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plotting training accuracy vs. validation accuracy
    ax1.plot(history.history['accuracy'], color='red', label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], color='blue', label='Validation Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.set_title(f'Training Accuracy vs. Validation Accuracy ({title})')
    ax1.legend()
    ax1.grid(True)

    # Plotting training loss vs. validation loss
    ax2.plot(history.history['loss'], color='red', label='Training Loss')
    ax2.plot(history.history['val_loss'], color='blue', label='Validation Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.set_title(f'Training Loss vs. Validation Loss ({title})')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
def visualize_gridsearch(grid_result):
    # Convert the GridSearchCV results to a pandas DataFrame for easy visualization
    results_df = pd.DataFrame(grid_result.cv_results_)
    
    # Rename 'mean_test_score' to 'accuracy' for clarity
    results_df.rename(columns={'mean_test_score': 'accuracy'}, inplace=True)
    
    # Create a new DataFrame that contains the hyperparameters and their corresponding accuracy score
    params_df = results_df[['params', 'accuracy', 'rank_test_score']]
    
    # Sort the DataFrame by rank to view the best combinations first
    params_df = params_df.sort_values(by='rank_test_score', ascending=True)

    # Extract the individual hyperparameter columns for better readability
    params_df = pd.concat([params_df.drop(['params'], axis=1), pd.json_normalize(params_df['params'])], axis=1)

    # Set a custom style using seaborn
    sns.set_style('whitegrid')
    sns.set_palette('muted')

    # Create a single figure for all the plots
    fig, axes = plt.subplots(3, 2, figsize=(20, 18))
    fig.suptitle('Grid Search Results: Hyperparameter Effects on Accuracy', fontsize=18, fontweight='bold')

    # Boxplot for activation functions
    sns.boxplot(x='model__activation', y='accuracy', data=params_df, ax=axes[0, 0], palette='viridis')
    axes[0, 0].set_title('Activation Function vs Accuracy', fontsize=15, fontweight='bold')
    axes[0, 0].set_xlabel('Activation Function', fontsize=12)
    axes[0, 0].set_ylabel('Accuracy', fontsize=12)

    # Boxplot for optimizers
    sns.boxplot(x='model__optimizer', y='accuracy', data=params_df, ax=axes[0, 1], palette='coolwarm')
    axes[0, 1].set_title('Optimizer vs Accuracy', fontsize=15, fontweight='bold')
    axes[0, 1].set_xlabel('Optimizer', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)

    # Line plot for learning rate
    sns.lineplot(x='model__learning_rate', y='accuracy', data=params_df, ax=axes[1, 0], marker='o', linewidth=2, markersize=8, color='royalblue')
    axes[1, 0].set_title('Learning Rate vs Accuracy', fontsize=15, fontweight='bold')
    axes[1, 0].set_xlabel('Learning Rate', fontsize=12)
    axes[1, 0].set_ylabel('Accuracy', fontsize=12)

    # Line plot for batch size
    sns.lineplot(x='batch_size', y='accuracy', data=params_df, ax=axes[1, 1], marker='o', linewidth=2, markersize=8, color='darkorange')
    axes[1, 1].set_title('Batch Size vs Accuracy', fontsize=15, fontweight='bold')
    axes[1, 1].set_xlabel('Batch Size', fontsize=12)
    axes[1, 1].set_ylabel('Accuracy', fontsize=12)

    # Line plot for epochs
    sns.lineplot(x='epochs', y='accuracy', data=params_df, ax=axes[2, 0], marker='o', linewidth=2, markersize=8, color='seagreen')
    axes[2, 0].set_title('Epochs vs Accuracy', fontsize=15, fontweight='bold')
    axes[2, 0].set_xlabel('Epochs', fontsize=12)
    axes[2, 0].set_ylabel('Accuracy', fontsize=12)
    
    # Turn off the empty subplot (bottom-right)
    fig.delaxes(axes[2, 1])

    # Adjust layout for better spacing
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
def visualize_3d_gridsearch(grid_result):
    # Convert the GridSearchCV results to a pandas DataFrame for easy visualization
    results_df = pd.DataFrame(grid_result.cv_results_)
    
    # Rename 'mean_test_score' to 'accuracy' for clarity
    results_df.rename(columns={'mean_test_score': 'accuracy'}, inplace=True)
    
    # Create a new DataFrame that contains the hyperparameters and their corresponding accuracy score
    params_df = results_df[['params', 'accuracy', 'rank_test_score']]
    
    # Extract the individual hyperparameter columns for better readability
    params_df = pd.concat([params_df.drop(['params'], axis=1), pd.json_normalize(params_df['params'])], axis=1)
    
    # Prepare data for 3D surface plot
    x = params_df['model__learning_rate'].values
    y = params_df['batch_size'].values
    z = params_df['accuracy'].values
    c = params_df['epochs'].values  # Color will represent epochs

    # Normalize the epochs values for coloring
    scaler = MinMaxScaler()
    c_normalized = scaler.fit_transform(c.reshape(-1, 1)).flatten()

    # Create grid data for interpolation
    learning_rates = np.linspace(min(x), max(x), 50)
    batch_sizes = np.linspace(min(y), max(y), 50)
    X, Y = np.meshgrid(learning_rates, batch_sizes)
    
    # Interpolate the data using griddata to create a smoother surface
    Z = griddata((x, y), z, (X, Y), method='cubic')
    C = griddata((x, y), c_normalized, (X, Y), method='nearest')  # Interpolating epochs for coloring

    # Plotting the 3D surface
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Surface plot with color representing the epochs
    surf = ax.plot_surface(X, Y, Z, facecolors=cm.viridis(C), edgecolor='k', alpha=0.8)

    # Add labels
    ax.set_title('Impact of Learning Rate and Batch Size on Model Accuracy with Varying Epochs', fontsize=15, fontweight='bold')
    ax.set_xlabel('Learning Rate', fontsize=12)
    ax.set_ylabel('Batch Size', fontsize=12)
    ax.set_zlabel('Accuracy', fontsize=12)

    # Add a color bar for epochs
    m = cm.ScalarMappable(cmap=cm.viridis)
    m.set_array(c)
    fig.colorbar(m, ax=ax, shrink=0.5, aspect=5, label='Epochs')

    plt.show()
    
    
def calculate_flops_from_model(model):
    """
    Calculate the FLOPs for a given Keras model without relying on TensorFlow functions.

    Args:
        model (tf.keras.Model): The model for which to calculate FLOPs.

    Returns:
        float: The total number of FLOPs.
    """
    flops = 0

    # Ensure the model is built by passing a sample input if necessary
    if not model.built:
        # Use the input shape from the first layer if available
        input_shape = model.layers[0].input_shape
        sample_input = np.random.normal(size=input_shape).astype(np.float32)
        model(sample_input)

    # Go through each layer of the model
    for layer in model.layers:
        # Ensure that the kernel is accessible and use it to determine input and output size
        if hasattr(layer, 'kernel'):
            input_size = layer.kernel.shape[0]  # Number of input features
            output_size = layer.kernel.shape[1]  # Number of output units

            # FLOPs for matrix multiplication: 2 * input_size * output_size (forward + backward pass)
            # FLOPs for bias addition: output_size
            layer_flops = 2 * input_size * output_size + output_size
            flops += layer_flops

    return flops

def to_gflops(flops):
    return flops / 1e9

