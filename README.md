List of things that still need to be done/checked:
1. Check that we can get fingerprintGAN and defenseGan to run independently
2. write accuracy check codes for measuring the classification accuracy against 3 baseline classification methods: k-nearest-neighbor (kNN) on raw pixels, Eigenface [7]
3. write accuracy check codes for the f-1 score and AU ROC
4. configure defense-GAN to take an image from fingerprint GAN models then feed result to fingerprint GAN classifier
5. IF time allows configure Defense GAN protection system to follow ATGAN structure (NOTE: there is no source code for ATGAN so we would have to reconstruct it from the paper, if we don't have time don't worry about it)
