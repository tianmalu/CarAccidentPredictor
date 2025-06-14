import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

if __name__ == "__main__":
    # 创建images文件夹（如果不存在）
    images_dir = "images"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        print(f"Created directory: {images_dir}")
    
    # 读取所有数据，不仅仅是酒精事故
    df = pd.read_csv("data/monatszahlen2505_verkehrsunfaelle_06_06_25.csv")
    
    # 只保留 'insgesamt' (总计) 类型，但包含所有事故类别
    df = df[df['AUSPRAEGUNG'] == 'insgesamt']
    
    print("Unique accident types:", df['MONATSZAHL'].unique())
    
    # 数据预处理
    df['month'] = df['MONAT'].astype(str).str[-2:]
    valid_months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    df = df[df['month'].isin(valid_months)]
    
    df['date'] = pd.to_datetime(df['JAHR'].astype(str) + '-' + df['month'], format='%Y-%m')
    
    # 过滤掉 'Summe' 行
    df = df[df['MONATSZAHL'] != 'Summe']
    
    print(f"Data shape: {df.shape}")
    print("Available accident types:")
    for i, accident_type in enumerate(df['MONATSZAHL'].unique()):
        print(f"{i+1}. {accident_type}")
    
    # 1. 所有事故类型的时间序列图
    plt.figure(figsize=(15, 10))
    accident_types = df['MONATSZAHL'].unique()
    
    for accident_type in accident_types:
        type_data = df[df['MONATSZAHL'] == accident_type]
        type_data = type_data.set_index('date').sort_index()
        plt.plot(type_data.index, type_data['WERT'], 
                label=accident_type, linewidth=1.5, alpha=0.8)
    
    plt.title('All Traffic Accident Types Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Number of Accidents', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'all_accidents_timeseries.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. 每种事故类型的子图
    n_types = len(accident_types)
    cols = 3
    rows = (n_types + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 6*rows))
    axes = axes.flatten() if n_types > 1 else [axes]
    
    for i, accident_type in enumerate(accident_types):
        if i < len(axes):
            type_data = df[df['MONATSZAHL'] == accident_type]
            type_data = type_data.set_index('date').sort_index()
            
            axes[i].plot(type_data.index, type_data['WERT'], linewidth=2)
            axes[i].set_title(f'{accident_type}', fontsize=12)
            axes[i].set_xlabel('Date')
            axes[i].set_ylabel('Number of Accidents')
            axes[i].grid(True, alpha=0.3)
            axes[i].tick_params(axis='x', rotation=45)
    
    # 隐藏多余的子图
    for i in range(n_types, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'accidents_by_type_subplots.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. 年度总计对比（柱状图）
    yearly_data = df.groupby(['JAHR', 'MONATSZAHL'])['WERT'].sum().reset_index()
    
    plt.figure(figsize=(15, 8))
    years = sorted(yearly_data['JAHR'].unique())
    
    # 为每种事故类型创建柱状图
    x = np.arange(len(years))
    width = 0.8 / len(accident_types)
    
    for i, accident_type in enumerate(accident_types):
        type_yearly = yearly_data[yearly_data['MONATSZAHL'] == accident_type]
        values = [type_yearly[type_yearly['JAHR'] == year]['WERT'].sum() 
                 if year in type_yearly['JAHR'].values else 0 for year in years]
        
        plt.bar(x + i * width, values, width, 
               label=accident_type, alpha=0.8)
    
    plt.title('Yearly Traffic Accidents by Type', fontsize=16)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Total Accidents', fontsize=12)
    plt.xticks(x + width * (len(accident_types) - 1) / 2, years, rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'yearly_accidents_by_type.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n=== SUMMARY STATISTICS ===")
    for accident_type in accident_types:
        type_data = df[df['MONATSZAHL'] == accident_type]
        total = type_data['WERT'].sum()
        avg = type_data['WERT'].mean()
        print(f"{accident_type}:")
        print(f"  Total: {total:.0f}, Average: {avg:.1f} per month")
    
    print(f"\nAll visualizations saved to '{images_dir}' folder:")
    print("1. all_accidents_timeseries.png - All types in one plot")
    print("2. accidents_by_type_subplots.png - Individual subplots")
    print("3. yearly_accidents_by_type.png - Yearly comparison")
