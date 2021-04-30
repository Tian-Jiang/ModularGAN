# ModularGAN
AS the paper&lt;Modular Generative Adversarial Networks>: https://arxiv.org/abs/1804.03343  
  
There has some difference at the structure of Reconstructor and Discriminator comparing from the original paper:  
  
	Reconstructor:  
		1. Use DECONV for the first two layers.  
		2. Use TanH instead of IN and ReLU for the last layer  
  
	Discriminator:  
		1. Remove IN in all layers  
		2. Remove Leaky ReLU for the last two output layers  

