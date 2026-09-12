import numpy as np
import os

def computePerVariableMetrics(
        fileName: str,
        preds: np.ndarray, # Predicitons in log Scale
        targets: np.ndarray, # Targets in Log Scale
        finalPreds: np.ndarray, #
        finalTargets: np.ndarray,
        savePath: str,
) -> None:
    variables = ["n","c1","c2","c3","c4","c5","c6"]
    savePath = str(savePath)
    percLog = ((preds-targets)/(targets + 1e-10)) *100
    percNorm = ((finalPreds-finalTargets)/(finalTargets + 1e-10)) *100


    # Log Space
    mseLog = np.mean((preds-targets)**2,axis=0)
    maeLog = np.mean(np.abs(preds-targets),axis=0)
    meanPercentageErrorLog = np.mean(percLog,axis=0)
    meanPercentageAbsoluteErrorLog = np.mean(np.abs(percLog),axis=0)

    # Normal Space
    mseNorm = np.mean((finalPreds-finalTargets)**2,axis=0)
    maeNorm = np.mean(np.abs(finalPreds-finalTargets),axis=0)
    meanPercentageErrorNorm = np.mean(percNorm,axis=0)
    meanPercentageAbsoluteErrorNorm = np.mean(np.abs(percNorm),axis=0)

    with open(savePath,"a") as f:
        f.write(f"\n============== {fileName} ==============\n")

        for i, name in enumerate(variables):
            line=(
                f"{name} | "
                f"log[MSE = {mseLog[i]:.10e}, MAE= {maeLog[i]:.10e},"
                f"%={meanPercentageErrorLog[i]:.10f}%, |%|={meanPercentageAbsoluteErrorLog[i]:.10f}] | "
                f"norm[MSE = {mseNorm[i]:.10e}, MAE= {maeNorm[i]:.10e},"
                f"%={meanPercentageErrorNorm[i]:.10f}%, |%|={meanPercentageAbsoluteErrorNorm[i]:.10f}] | "
            )

            print(line)
            f.write(line + "\n")
