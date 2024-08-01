import torch
from asvdataset import ASV_DATASET

asv_directory  = '/path/to/location/of/directory/ASV'
asv_dataset_pt = '/path/to/location/ASV_dataset.pt'

def main():
    new_asv = ASV_DATASET(asv_directory, 'dev', 'LA')
    print(len(new_asv))
    print(new_asv[0][0].shape, new_asv[100][0].shape, new_asv[1000][0].shape, new_asv[10][0].shape)

    torch.save(new_asv, asv_dataset_pt)


if __name__ == "__main__":
    main()