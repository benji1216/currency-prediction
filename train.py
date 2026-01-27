import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import datetime
import os
import time

# --- Import 自定義模組 ---
from models import LinearRegression, DeepRegressor, LSTMRegressor
# 記得要把 plot_scatter 加進去 import
from utils import load_data, plot_loss, plot_prediction, plot_scatter

# ==========================================
# 🎛️ 設定區 (Configuration)
# ==========================================
MODEL_TYPE = "Linear"     # 選項: "Linear", "Deep", "LSTM"
INPUT_DAYS = 5          # 時間視窗
FEATURE_SIZE = 32       # 特徵數
EPOCHS = 12000          # 訓練回數
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔥 鎖定隨機種子
torch.manual_seed(1234)
np.random.seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)

def main():
    # ==========================================
    # 1. 資料準備與訓練集處理
    # ==========================================
    print("Loading Data...")
    train_df = load_data("./train") 
    train = train_df.to_numpy().astype(float)
    
    train_size = len(train) - INPUT_DAYS
    train_x = np.empty([train_size, INPUT_DAYS, FEATURE_SIZE], dtype=float)
    train_y = np.empty([train_size, FEATURE_SIZE], dtype=float)

    for idx in range(train_size):
        train_x[idx, :, :] = train[idx : idx + INPUT_DAYS]
        train_y[idx, :] = train[idx + INPUT_DAYS]

    target_cols = [i for i, col in enumerate(train_df.columns) if '現鈔買入' in col]
    train_y = train_y[:, target_cols]

    # 正規化
    mean_x = np.mean(train_x, axis=(0, 1))
    std_x = np.std(train_x, axis=(0, 1))
    std_x = np.where(std_x == 0, 1, std_x) 
    train_x = (train_x - mean_x) / std_x

    train_x_tensor = torch.from_numpy(train_x).float().to(DEVICE)
    train_y_tensor = torch.from_numpy(train_y).float().to(DEVICE)

    val_ratio = 0.2
    val_size = int(train_size * val_ratio)
    indices = torch.randperm(train_size)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    
    val_x = train_x_tensor[val_idx]
    val_y = train_y_tensor[val_idx]
    train_x_final = train_x_tensor[train_idx]
    train_y_final = train_y_tensor[train_idx]

    # ==========================================
    # 2. 模型初始化
    # ==========================================
    print(f"Initializing Model: {MODEL_TYPE}")
    
    if MODEL_TYPE == "Linear":
        model = LinearRegression(INPUT_DAYS * FEATURE_SIZE, 8).to(DEVICE)
        lr = 0.1 
    elif MODEL_TYPE == "Deep":
        model = DeepRegressor(INPUT_DAYS * FEATURE_SIZE, 8).to(DEVICE)
        lr = 0.0005
    elif MODEL_TYPE == "LSTM":
        model = LSTMRegressor(input_dim=FEATURE_SIZE, hidden_dim=64, output_dim=8).to(DEVICE)
        lr = 0.0005
    else:
        raise ValueError("Unknown MODEL_TYPE")

    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1500, factor=0.3)

    # ==========================================
    # 3. 訓練迴圈 (加入計時器)
    # ==========================================
    print(f"Start Training {EPOCHS} epochs...")
    
    # ⏱️ 開始計時
    start_time = time.time()
    
    train_hist, val_hist = [], []

    for epoch in range(EPOCHS):
        model.train()
        pred = model(train_x_final)
        loss = criterion(pred, train_y_final)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_hist.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_pred = model(val_x)
            val_loss = criterion(val_pred, val_y)
            val_hist.append(val_loss.item())
        
        scheduler.step(val_loss)

        if (epoch+1) % 1000 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1} | Train Loss: {loss.item():.6f} | Val Loss: {val_loss.item():.6f} | LR: {current_lr:.6f}")

    # ⏱️ 結束計時並計算
    end_time = time.time()
    total_time = end_time - start_time
    
    # 轉換為 分:秒 格式
    mins, secs = divmod(total_time, 60)
    print(f"\n✅ Training Completed in {int(mins)}m {int(secs)}s")
    print("-" * 30)

    # 畫 Loss 圖
    plot_loss(train_hist, val_hist, model_name=MODEL_TYPE)
    
    # ==========================================
    # 4. 測試集預測與輸出
    # ==========================================
    print("Testing...")
    test_df = load_data("./test")
    test_np = test_df.to_numpy().astype(float)
    test_size_raw = test_np.shape[0]
    
    test_size = test_size_raw // INPUT_DAYS
    test_x = np.empty([test_size, INPUT_DAYS, FEATURE_SIZE], dtype=float)

    for idx in range(test_size):
        test_x[idx, :, :] = test_np[idx * INPUT_DAYS : (idx + 1) * INPUT_DAYS]
    
    test_x = (test_x - mean_x) / np.where(std_x == 0, 1, std_x)
    test_x_tensor = torch.from_numpy(test_x).float().to(DEVICE)
    
    model.eval()
    with torch.no_grad():
        predicted = model(test_x_tensor)
        predicted_np = predicted.cpu().numpy()

    ids = [x for x in range(len(predicted_np))]
    output_df = pd.DataFrame({'id': ids})
    currency_columns = ["AUD", "CAD", "EUR", "GBP", "HKD", "JPY", "KRW", "USD"]

    for i, column_name in enumerate(currency_columns):
        output_df[column_name] = predicted_np[:, i]

    if not os.path.exists("./output"):
        os.makedirs("./output")
        
    current_time = datetime.datetime.now().strftime("%Y_%m%d_%H%M")
    file_name = f'./output/output_{MODEL_TYPE}_{current_time}.csv'
    output_df.to_csv(file_name, index=False)
    print(f"Saved raw prediction to {file_name}")

    # ==========================================
    # 5. 結果對齊與畫圖
    # ==========================================
    print("Generating Result Alignment...")
    records = []
    
    for i in range(predicted_np.shape[0] - 1):         
        for j, cur in enumerate(currency_columns):  
            target_col_name = f"{cur}現鈔買入"
            col_index = test_df.columns.get_loc(target_col_name)
            actual_index = i * INPUT_DAYS + INPUT_DAYS
            
            if actual_index >= len(test_np):
                continue
            actual_value = test_np[actual_index, col_index]

            records.append({
                "id": i,                        
                "Currency": cur,                 
                "Actual": actual_value,        
                "Pred": predicted_np[i, j]      
            })

    orange_df = pd.DataFrame.from_records(records)
    orange_df.to_csv(f"./output/combined_for_orange_{MODEL_TYPE}.csv", index=False)
    
    # 畫 USD 走勢圖
    usd_records = [r for r in records if r['Currency'] == 'USD']
    usd_actuals = [r['Actual'] for r in usd_records]
    usd_preds = [r['Pred'] for r in usd_records]
    
    plot_prediction(
        usd_actuals, 
        usd_preds, 
        title=f"USD Prediction ({MODEL_TYPE})", 
        model_name=MODEL_TYPE
    )

    # 畫散佈圖
    all_actuals = [r['Actual'] for r in records]
    all_preds = [r['Pred'] for r in records]
    
    plot_scatter(all_actuals, all_preds, model_name=MODEL_TYPE)

    print(f"Done! All results saved for model: {MODEL_TYPE}")

if __name__ == "__main__":
    main()