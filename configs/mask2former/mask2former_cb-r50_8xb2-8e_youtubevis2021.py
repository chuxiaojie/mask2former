_base_ = ['../../video_tracking/configs/vis/mask2former/mask2former_r50_8xb2-8e_youtubevis2021.py']


custom_imports = dict(
    imports=['cbnet'], allow_failed_imports=False)
