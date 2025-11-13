Get van criekinge dataset and rename it into `./van_criekinge_unprocessed_1`

Go to https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.1.0.zip, download the following files and rename them.

``` zsh
basicmodel_m_lbs_10_207_0_v1.1.0.pkl -> SMPL_MALE.pkl
basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl -> SMPL_NEUTRAL.pkl
basicmodel_f_lbs_10_207_0_v1.1.0.pkl -> SMPL_FEMALE.pkl
```

How to run the files 1 to 4

First, install conda.

```zsh
conda create -n vc_motion \
  python=3.10 \
  numpy \
  pytorch \
  matplotlib \
  tqdm \
  -c pytorch -c conda-forge

conda activate vc_motion

pip install smplx c3d
```

Then, you can run the 4 files.

```zsh
python 1_dataset_prep.py \
    --data_dir "./van_criekinge_unprocessed_1" \
    --output_dir "./processed_markers_all_2" \
    --resample_fps 20.0 \
    --gap_fill_max 10
```

```zsh
python 2_fit_smpl_markers.py \
    --processed_dir "./processed_markers_all_2" \
    --models_dir "." \                        
    --out_dir "./fitted_smpl_all_3" \
    --device cpu \   
    --iters 400
```

```zsh
python 3_export_humanml3d.py --fits_dir ./fitted_smpl_all_3 --out_dir ./humanml3d_joints_4
```

```zsh
mv ./humanml3d_joints_4 ./Comp_v6_KLD01
```

```zsh
KMP_DUPLICATE_LIB_OK=TRUE python 4_motion_process.py --build_vc --vc_root . --vc_splits_dir ./empty_splits --vc_meta_py ./van_criekinge_unprocessed_1/metadata.py
```

To evaluate, you can visualize into gif.

```zsh
python 5_visualize_comparison.py  \
    --c3d_path "ORIGINAL C3D LOCATION" \
    --rec_npz_path "YOUR NPZ LOCATION" \
    --output_gif "OUTPUT LOCATION" \
    --gif_fps 20
```