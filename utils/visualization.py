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


def generate_summary_statistics(methods, result_paths, metric):
    summary_data = {method: {'0.2': '', '0.5': '', '0.8': ''} for method in methods}

    for result_path in result_paths:
        param = result_path.split('(')[-1][:-1]  # Extract the parameter value from the folder name
        for method in methods:
            file = f"{result_path}/{method}.pkl"
            with open(file, "rb") as f:
                k = pickle.load(f)
                k = k['overall_results']
                last_epoch_values = [k[i][metric][-1].item() for i in range(len(k))]
                last_epoch_tensor = torch.tensor(last_epoch_values)
                
                mean_val = last_epoch_tensor.mean().item()
                std_val = last_epoch_tensor.std().item()
                
                summary_data[method][param] = f"{mean_val:.4f} ± {std_val:.4f}"

    df = pd.DataFrame.from_dict(summary_data, orient='index')
    df.index.name = 'Method'
    return df

def save_table_as_latex(df, filename):
    with open(filename, 'w') as f:
        f.write(df.to_latex(index=True, escape=False))

# Parameters
methods = ['FBLoss_opt','Back','Back_opt','Back_conv','Back_opt_conv','LBL', 'EM', 'Forward', 'ForwardBackward_I', 'ForwardBackward_Y']
result_paths = ["Results/MLP MNIST/Experimental_results(0.2)", "Results/MLP MNIST/Experimental_results(0.5)", "Results/MLP MNIST/Experimental_results(0.8)"]
metric = 'test_acc'  # You can change this to 'train_loss', 'train_acc', etc.

# Generate and save the summary statistics table
df_summary = generate_summary_statistics(methods, result_paths, metric)
save_table_as_latex(df_summary, 'results_summary_table.tex')

# Display the DataFrame for inspection
print(df_summary)