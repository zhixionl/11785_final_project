### This function is used to plot and print the output results (MPJPE/PA-MPJPE) based on the saved results ###

import matplotlib.pyplot as plt

import numpy as np
import glob
import os


def output_results(name, plot = False):
    npz_path = os.path.join('/home/jack/Documents/GitHub/11785_final_project/eval_result', name)

    pattern = os.path.join(npz_path, '*')
    npz_list = glob.glob(pattern)

    mpjpe_list = []
    reconstruction_error = []
    for i, npz_i in enumerate(sorted(npz_list)):

        if i > 30:
            continue

        data = np.load(npz_i)
        mpjpe_list.append(data['MPJPE'])
        reconstruction_error.append(data['reconstruction'])

    print('MPJPE: ', min(mpjpe_list[1:]))
    print('Recontruction: ', min(reconstruction_error[1:]))

    if plot:

        plt.figure(1)
        plt.plot(mpjpe_list)
        plt.xlabel('Steps')
        plt.ylabel('MPJPE')
        plt.title(name)
        plt.show()

        plt.figure(2)
        plt.plot(reconstruction_error)
        plt.xlabel('Steps')
        plt.ylabel('reconstruction_error')
        plt.title(name)
        plt.show()


if __name__ == '__main__':
    print('########## Ablation: Resnet 2 layers ##########')
    output_results('ablation_mlp_resnet_2_layers')

    print('########## Ablation: Convnext 2 layers ##########')
    output_results('ablation_mlp_convnext')

    print('########## Ablation: Convnext 5 layers ##########')
    output_results('ablation_mlp_convnext_5_layers')

    print('########## Ablation: Conv1d with Convnext 2 layers ##########')
    output_results('ablation_conv1d_convnext')

    print('########## Full train: Resnet 2 layers s1-s6 ##########')
    output_results('full_mlp_resnet_2_layers_s1-6')

    print('########## Full train: Resnet 2 layers s1-s7 ##########')
    output_results('full_mlp_resnet_2_layers_s1-7', plot=False)

    print('########## Full train: Resnet average s1-s7 ##########')
    output_results('method_3_full_s1-7', plot=False)

    print('########## Full train: Convnext 5 layers s1-s7 ##########')
    output_results('full_mlp_convnext_5_layers_s1-7', plot=False)

    print('########## Full train: Resnet 3 layers s1-s7 ##########')
    output_results('resnet_mlp_3_layers_full_s1-7', plot=True)