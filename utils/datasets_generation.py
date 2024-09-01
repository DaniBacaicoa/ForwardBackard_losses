from src.dataset import Data_handling
import os


def generate_dataset(dataset, corruption, batch_size = 16, model_class = 'pll', corr_p = 0.5 ,corr_n = None, repetitions = 1):
    corruption = model_class
    base_dir = "Datasets/weak_datasets"
    if corr_n is not None:
        folder_path = os.path.join(dataset_base_path, f'{dataset}_{corruption}_p_+{corr_p}p_-{corr_n}')
    else:
        folder_path = os.path.join(dataset_base_path, f'{dataset}_{corruption}_p{corr_p}')
    folder_path = os.path.join(base_dir, f'{dataset}_{corruption}_p{pll_p}')

    # Check if the folder exists, if not, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    Data = Data_handling(dataset=dataset, batch_size=32)
    Weak.generate_M(model_class = corruption, corr_p = 0.1, corr_n = 0.1)
    train_X,train_y,test_X,test_y =  Data.get_data()
    Weak.generate_weak(train_y) #z and w 

    Dataset = [Data, Weak]
    if number is None:
        f = open(folder_path + f"/Dataset.pkl", "wb")
    else:
        f = open(folder_path + f"/Dataset_{repetitions}.pkl", "wb")
    pickle.dump(Dataset)
    f.close()
    








