import src as dtset


def generate_dataset(dataset, batch_size = 16, model_class = 'pll', pll_p = 0.5, number = None)
    if dataset == 'mnist':
        Data = dtset.Torch_Dataset(dataset, batch_size = batch_size)
    else:
        Data = dtset.OpenML_Dataset(dataset, batch_size = batch_size)
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






