import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_data(folder_path):
    """讀取資料夾內所有 CSV 並合併"""
    load_df = pd.DataFrame()
    name_list = []
    
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Directory not found: {folder_path}")

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            load_df = pd.concat([load_df, df["現鈔買入"], df["現鈔賣出"], df["即期買入"], df["即期賣出"]], axis=1)
            clean_name = filename.replace(".csv", "")
            name_list.extend([f"{clean_name}現鈔買入", f"{clean_name}現鈔賣出", f"{clean_name}即期買入", f"{clean_name}即期賣出"])
    
    load_df.columns = name_list
    load_df = load_df.iloc[::-1] # 反轉時間序
    load_df = load_df.reindex(sorted(load_df.columns), axis=1)
    load_df.replace("-", 0, inplace=True)
    
    return load_df

def check_images_dir():
    """確保 images 資料夾存在"""
    if not os.path.exists("./images"):
        os.makedirs("./images")

def plot_loss(train_hist, val_hist, model_name):
    check_images_dir()
    plt.figure(figsize=(10, 6))
    plt.plot(train_hist, 'r-', label='Train Loss')
    plt.plot(val_hist, 'b-', label='Validation Loss')
    plt.title(f'Training Convergence ({model_name})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 檔名加入 model_name
    filename = f"./images/loss_curve_{model_name}.png"
    plt.savefig(filename)
    print(f"Snapshot saved: {filename}")
    
    plt.show()

def plot_prediction(actual, predicted, title, model_name):
    check_images_dir()
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual Data', color='black', linewidth=1.5)
    plt.plot(predicted, label='AI Prediction', color='orange', linestyle='--', alpha=0.9)
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Exchange Rate')
    plt.legend()
    plt.grid(True)
    
    # 檔名使用 model_name 加 title 的關鍵字
    safe_title = "time_series" # 簡化檔名
    filename = f"./images/{safe_title}_{model_name}.png"
    plt.savefig(filename)
    print(f"Snapshot saved: {filename}")
    
    plt.show()

def plot_scatter(actual, predicted, model_name):
    check_images_dir()
    plt.figure(figsize=(8, 8))
    plt.scatter(actual, predicted, alpha=0.5, c='g', label='Testing Data')
    
    lims = [min(min(actual), min(predicted)), max(max(actual), max(predicted))]
    plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Perfect Prediction')
    
    plt.title(f'Actual vs Predicted ({model_name})')
    plt.xlabel('Actual Exchange Rate')
    plt.ylabel('Predicted Exchange Rate')
    plt.legend()
    plt.grid(True)
    
    # 檔名加入 model_name
    filename = f"./images/scatter_plot_{model_name}.png"
    plt.savefig(filename)
    print(f"Snapshot saved: {filename}")
    
    plt.show()