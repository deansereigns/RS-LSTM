import pandas as pd
import os

def savePredicitons(timeSteps,preds,targets,percentageError,path, normalization='none'):
    featureName=["NeutronPop","C1","C2","C3","C4","C5","C6"]
    columnsHeader = ["Time (s)"]

    for fi in featureName:
        columnsHeader+=[f"True_{fi}",f"Pred_{fi}",f"PercentageError_{fi}"]
    

    data = {"Time (s)": timeSteps}
    for i, fi in enumerate(featureName):
        data[f"True_{fi}"] =  targets[:,i]
        data[f"Pred_{fi}"] = preds[:,i]
        data[f"PercentageError_{fi}"] =percentageError[:,i]

    df_out= pd.DataFrame(data,columns=columnsHeader)

    if normalization =='log':
        csvPath = os.path.join(path,"log_preds_targets_percen_error.csv")
        df_out.to_csv(csvPath,index=True)
        print("CSV Stored at", csvPath)
    else:
        csvPath = os.path.join(path,"preds_targets_percen_error.csv")
        df_out.to_csv(csvPath,index=True)
        print("CSV Stored at", csvPath)

def saveShutdownPredictions(timeSteps,preds,targets,percentageError,path, normalization='none'):
    featureName=["NeutronDensity"]
    columnsHeader = ["Time (s)"]

    for fi in featureName:
        columnsHeader+=[f"True_{fi}",f"Pred_{fi}",f"PercentageError_{fi}"]
    

    data = {"Time_s": timeSteps}
    for i, fi in enumerate(featureName):
        data[f"True_{fi}"] =  targets[:,i]
        data[f"Pred_{fi}"] = preds[:,i]
        data[f"PercentageError_{fi}"] =percentageError[:,i]

    df_out= pd.DataFrame(data,columns=columnsHeader)

    if normalization =='log':
        csvPath = os.path.join(path,"log_shutdown_preds_targets_percen_error.csv")
        df_out.to_csv(csvPath,index=True)
        print("CSV Stored at", csvPath)
    else:
        csvPath = os.path.join(path,"shutdown_preds_targets_percen_error.csv")
        df_out.to_csv(csvPath,index=True)
        print("CSV Stored at", csvPath)