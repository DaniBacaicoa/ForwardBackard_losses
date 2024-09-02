import seaborn
import pandas as pd
import os

losses = ['Forward','FBLoss_opt', 'Back_opt_conv']
loss_names = ['Forward','F/B optimized','Convex Backward']

def candles(folder_path, losses, corruptions):
    df_list = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)
            # Append the DataFrame to the list
            df_list.append(df)

    merged_df = pd.concat(df_list, ignore_index=True)

    # First plot: Training Set
    sns.boxplot(x="Corruption (p)", y="Train accuracy",
                hue="Loss", palette=["m", "g", "b", "y"],
                data=df, ax=axes[0])
    sns.despine(offset=10, trim=True, ax=axes[0])
    axes[0].set_title('Training Set')

    # Second plot: Testing Set
    sns.boxplot(x="Corruption (p)", y="Test accuracy",
                hue="Loss", palette=["m", "g", "b", "y"],
                data=df, ax=axes[1])
    sns.despine(offset=10, trim=True, ax=axes[1])
    axes[1].set_title('Testing Set')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

def table():
    