import seaborn

#loss_name = ['Forward','F/B optimized','Lower bounded Backward','Convex Back']
#loss = ['Forward','FBLoss_opt','LBL', 'Back_opt_conv']
loss_name = ['Forward','F/B optimized','Convex Backward']
loss = ['Forward','FBLoss_opt', 'Back_opt_conv']
#loss_name = ['Forward','F/B optimized','Lower bounded Backward']
#loss = ['Forward','FBLoss_opt','LBL']
#loss_name = ['Forward']
#loss = ['Forward']

ps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for p in ps:
    for e,los in enumerate(loss):
        file = f"Experimental_results({p})/{los}.pkl"
        with open(file, "rb") as f:
            k = pickle.load(f)
            k = k['overall_results']

        for i in range(len(k)):
            new_data = [{'Train accuracy':k[i]['train_acc'][-1].tolist(),
                        'Test accuracy':k[i]['test_acc'][-1].tolist(),
                        'Loss':loss_name[e],
                        'Corruption (p)':p
                        }]
            #print(new_data)
            df_new = pd.DataFrame(new_data)
            df = pd.concat([df, df_new], ignore_index=True)
            #df = df.append(df_new, ignore_index=True)