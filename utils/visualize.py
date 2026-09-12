import matplotlib.pyplot as plt
import os

labels = ['NeutronPop', 'c1', 'c2', 'c3', 'c4','c5', 'c6']

def plotLosses(losses, savePath = "trainingLoss.png"):
    """
    Plots and saves Training Loss Curve

    Args: 
        losses(list): List of Epochs Losses
        savePath(str): Path to save the plot (default: "trainingLoss.png")
    """
    plt.figure(figsize=(10,10))
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(savePath) or ".", exist_ok=True)
    plt.savefig(savePath,dpi=300)
    print("Loss Plot saved at {savePath}")
    
    plt.show()

def plotPredictions(timeAxis,preds,targets,percentageError,path=".", normalization ='none'):
    """
    Plots Predictions vs Targets and Percentage Error
    
    Args:
        timeAxis (np.array): Time Stes for x axis
        preds : Model Predictions
        targets: Ground Truth values    
        percentageError: Percentage Error between preds and targets
    """

    for i in range(len(labels)):
        plt.figure(figsize=(12,12))
        plt.plot(timeAxis,targets[:,i],label="True Value",color ="Blue")
        plt.plot(timeAxis,preds[:,i],label="Predictions",color ="Red", linestyle ='--')
        plt.xlabel("Time (s)")
        plt.ylabel(labels[i])
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        os.makedirs(path,exist_ok=True)
        if normalization =='log':
            plt.title(f"log({labels[i]}) Real vs Predicted")
            plt.savefig(os.path.join(path,f"log{labels[i]}_real_vs_pred.png"),dpi = 300)
        else:
            plt.title(f"{labels[i]} Real vs Predicted")
            plt.savefig(os.path.join(path,f"{labels[i]}_real_vs_pred.png"),dpi = 300)


        # Percentage Error Plot
        plt.figure(figsize=(12,12))
        plt.plot(timeAxis,percentageError[:,i],label="Percentage Error",color ="red")
        plt.xlabel("Time (s)")
        plt.ylabel("%_Error")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if normalization =='log':
            plt.title(f"log({labels[i]}) %_Error")
            plt.savefig(os.path.join(path,f"log{labels[i]}_%_error.png"),dpi = 300)
        else:
            plt.title(f"{labels[i]} %_Error")
            plt.savefig(os.path.join(path,f"{labels[i]}_%_error.png"),dpi = 300)


def plotShutdownPredictions(timeAxis,preds,targets,percentageError,path=".", normalization ='none'):
    """
    Plots Predictions vs Targets and Percentage Error for Shutdown Prediction
    
    Args:
        timeAxis (np.array): Time Stes for x axis
        preds : Model Predictions
        targets: Ground Truth values
        percentageError: Percentage Error between preds and targets
    """
    
    shutdown_labels = ['NeutronDensity']

    for i in range(len(shutdown_labels)):
        plt.figure(figsize=(12,12))
        plt.plot(timeAxis,targets[:,i],label="True Value",color ="Blue")
        plt.plot(timeAxis,preds[:,i],label="Predictions",color ="Red", linestyle ='--')
        plt.xlabel("Time (s)")
        plt.ylabel(shutdown_labels[i])
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        os.makedirs(path,exist_ok=True)
        if normalization =='log':
            plt.title(f"log({shutdown_labels[i]}) Real vs Predicted")
            plt.savefig(os.path.join(path,f"log{shutdown_labels[i]}_real_vs_pred.png"),dpi = 300)
        else:
            plt.title(f"{shutdown_labels[i]} Real vs Predicted")
            plt.savefig(os.path.join(path,f"{shutdown_labels[i]}_real_vs_pred.png"),dpi = 300)

        # Percentage Error Plot
        plt.figure(figsize=(12,12))
        plt.plot(timeAxis,percentageError[:,i],label="Percentage Error",color ="red")
        plt.xlabel("Time (s)")
        plt.ylabel("%_Error")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if normalization =='log':
            plt.title(f"log({shutdown_labels[i]}) %_Error")
            plt.savefig(os.path.join(path,f"log{shutdown_labels[i]}_%_error.png"),dpi = 300)
        else:
            plt.title(f"{shutdown_labels[i]} %_Error")
            plt.savefig(os.path.join(path,f"{shutdown_labels[i]}_%_error.png"),dpi = 300)
