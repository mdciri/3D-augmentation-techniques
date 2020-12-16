# Augmentation techniques
Chechink the performance of different augmentation techniques on a 3D U-Net, have a look to the paper [What is the best data augmentation approach for brain tumor segmentation using 3D U-Net?](https://arxiv.org/abs/2010.13372).

For this project, we used the [BraTS 2020](https://www.med.upenn.edu/cbica/brats2020/data.html) training and validation set.

Pay attention that the aim of the paper was not to implement a model able to do image segmentation as the state-of-the-art ones, but to verify how different augmentation techniques and paramenter change the model performance.

The image augmentation techniques used are:

*Patch extraction*: from each original volume a sub-volume of shape 128 × 128 × 128 is extracted around its centre. In this way each sub-volume mostly contains brain tissue and not the surrounding background.
- *Flipping*: random flipping of one of the three different axes with 1/3 probability.
- *Rotation*: rotation applied to each axis with angles randomly chosen from a uniform distribution with range between 0° and 15°, 30°, 60°, or 90°.
- *Scale*: scaling applied to each axis by a factor randomly chosen from a uniform distribution with range ±10% or ±20%.
- *Brightness*: power-law γ intensity transformation with its parameters gain (g) and γ chosen randomly between 0.8 - 1.2 from a uniform distribution. The intensity (I) is randomly changed according to the formula: Inew = g · I^γ.
- *Elastic deformation*: elastic deformation with square deformation grid with displacements sampled from from a normal distribution with standard deviation σ = 2, 5, 8, or 10 voxels, where the smoothing is done by a spline filter with order 3 in each dimension.
