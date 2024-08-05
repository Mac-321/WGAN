#Generative Adversial Networks (GAN's)

The GAN model was introduced by Goodfellow et al [1] 2014. The original implementation is seen in the paper but is known to be difficult to train due to unstabilities during training. These issues are the usual gradient explosion or vanishing but a prominent issue is mode collapse or the Discriminator being too effcetive as its role. Mode collapse is where the Generator only produces a subset of outputs to fool the Discriminaotr. This means that the diversity of generated outputs is low and not sought after. There is then the other common role of where the discriminator converges faster than the generator meaning there is no adversial game and the decoder just spots all the generators results as incorrect. The implementation of Wasserstein Gan with Gradient Penalty (WGAN-GP) from the 2017 paper [2] is a loss function introduced by the team in the paper to help stabilise the training of GAN's.

The file used was the CNN-GAN file and was trained for 200 epochs using the MNIST dataset. The file is included in the repo. Below are some representitive images. The resolution is low but the numbers from the MNIST dataset can be distinguished.

![image](https://github.com/user-attachments/assets/1cba4c23-9ebe-4283-ae29-13ad4bba8603)
![image](https://github.com/user-attachments/assets/8086e76b-73ea-4da0-b6a6-6bd41f0bed0f)
![image](https://github.com/user-attachments/assets/b6e4554e-8c45-49f6-8b70-80a56fbd54d4)
![image](https://github.com/user-attachments/assets/0687d445-0a75-4c5a-963a-40aa5577c08a)

![image](https://github.com/user-attachments/assets/ed825cdb-05da-4377-b3dd-6d88b1bf0803)
![image](https://github.com/user-attachments/assets/f018aef1-b911-4048-b935-d0f8e6346786)
![image](https://github.com/user-attachments/assets/43e8f925-cca9-49c0-9ea8-3b51298859c9)

[1] GoodFellow et al
[2] https://proceedings.neurips.cc/paper_files/paper/2017/file/892c3b1c6dccd52936e27cbd0ff683d6-Paper.pdf
