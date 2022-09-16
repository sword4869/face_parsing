> folk from https://github.com/zllrunning/face-parsing.PyTorch

# Description
This repo is used to generate semantic segmentation for face image.
The facial parts are listed bellow:
1. skin
2. left brow
3. right brow
4. left eye
5. right eye
6. nose
7. mouth
8. upper lip
9. lower lip

# Output
The generating script is `segment.py`. It will output the segmentaion images to
the specific folder(`test_res` by default).

The output root folder contains an `unmerge` folder and merged images, whose file
names are begin with `Mask_`.

Under the `umerge` folder, there are folders named with the input file names.
Inside them are unmerged parts.

# Author
Wei Deng

[dw-dengwei@outlook.com](mailto://dw-dengwei@outlook.com)