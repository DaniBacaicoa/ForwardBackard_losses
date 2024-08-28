from src.dataset import Data_handling
import os


def generate_dataset(dataset, batch_size = 16, model_class = 'pll', pll_p = 0.5, number = None):
    corruption = 'pll'
    base_dir = "Datasets"
    folder_path = os.path.join(base_dir, f'{dataset}_{corruption}_p{pll_p}')

    # Check if the folder exists, if not, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    existing_files = os.listdir(folder_path)
    dataset_files = [f for f in existing_files]# if f.startswith(dataset)]
    if dataset_files:
        # Extract the numbers from the dataset filenames and find the max
        highest_number = max(int(f.split('_')[-1]) for f in dataset_files)
        new_number = highest_number + 1
    else:
        # If no datasets exist, start numbering from 1
        new_number = 1

    # Construct the new dataset filename
    dataset_filename = f"{dataset_base_name}_{new_number}"
    dataset_path = os.path.join(folder_path, dataset_filename)

    # Save the dataset (for now, we'll just simulate saving by writing a dummy file)
    with open(dataset_path, 'w') as f:
        f.write(str(dataset))  # Replace this with actual dataset saving code

    print(f"Dataset saved as: {dataset_path}")

# Example usage
# Assume `dataset` is the dataset you want to save
dataset = "Sample dataset content"  # Replace with actual dataset object
save_dataset(dataset, "MNIST_p05", "MNIST")





    Data = Data_handling(dataset, batch_size = batch_size)
    Weak = Weakener(Data.num_classes)
    Weak.generate_M(model_class = model_class, pll_p = pll_p)
    train_X,train_y,test_X,test_y =  Data.get_data()
    Weak.generate_weak(train_y) #z and w   

    Dataset = [Data,Weak]
    if number is None:
        f = open('datasets/' + f'{dataset}/' +"/Dataset.pkl","wb")
    else:
        f = open(path +f"/Dataset{number}.pkl","wb")
    pickle.dump(Dataset,f)
    f.close()






