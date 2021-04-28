# 6DPoseNeRF
(Initial Proposal)[https://axquaris.github.io/6DPoseNeRF/]
(Milestone Report)[https://axquaris.github.io/6DPoseNeRF/milestone]
### Notes:
Timeline:
 - Use NERF and real-time differentiable renderer (code / libraries available) to set up iNeRF reproduction

Potential Methods / Contributions:
  - Starting with low resolution renders to estimate PDF or match-quality distribution over all poses, 
    iteratively re-sample this and increase render resolution for a gradient-free, evolution-based optimization 
    approach.
  - Try more in-depth sampling schemes for generating rays and pixels to be rendered for matching
    (could also analyze existing sampling scheme in more depth)
  - Matching low-res fields instead of detailed keypoints: see SIREN or low-frequency fourier representations.
    There are many; resolution, per-image sampling, and pose sampling tradeoffs to be considered

Areas of work:
 - Deep Learning: NeRF training and dataset setup
 - Ray Tracing: Calculation of radiance for each ray sample, involves NeRF inference optimization
 - Sampling: Generation of rays to sample and match over
 - Differentiable Rendering: differentiate loss with respect to pose over all traced rays
 - Optimization: loop for finding true camera pose, involves using gradient or evolution-based optimizers

Applications:
 - SLAM: localization in 3d environments
 - 6D Pose estimation: predict camera pose viewing a single object
