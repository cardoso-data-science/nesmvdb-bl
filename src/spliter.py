import pickle
import numpy as np
from concurrent.futures import ProcessPoolExecutor

def save_pickle(i, file_path):
    with np.load(file_path) as f:
        dct = {}
        dct['X'] = f['x'][i, :, [1, 0, 2, 3, 4, 5, 6, 9, 7]]
        dct['y'] = f['y'][i, :, [1, 0, 2, 3, 4, 5, 6, 9, 7]]
        dct['mask'] = f['decoder_mask'][i, :9999]
        with open(f'/content/gdrive/MyDrive/Baseline/{i}.pickle', 'wb') as handle:
            print(f"Saving {i}")
            pickle.dump(dct, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return f"Saved {i}"

def parallel_processing(file_path):
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(save_pickle, range(228, 4471), [file_path]*4471))
    
    for result in results:
        print(result)

# Agora, você apenas chama a função parallel_processing passando o caminho do arquivo npz como argumento.
if __name__ == '__main__':
    file_path = '\video-bgm-generation\dataset\data_train.npz'
    parallel_processing(file_path)
