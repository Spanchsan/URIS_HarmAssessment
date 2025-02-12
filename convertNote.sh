jupyter nbconvert --to script notebook.ipynb
mv notebook.py notepy.py
nohup python notepy.py &
