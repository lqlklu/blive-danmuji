.\env\Scripts\activate
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements_win.txt
cd deps\monotonic_align
python setup.py install
cd ..\..
