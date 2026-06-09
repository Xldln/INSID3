


conda create --name insid3 python=3.10 -y
conda activate insid3
pip install -r requirements.txt


git clone https://github.com/netw0rkf10w/CRF.git
cd CRF
python setup.py install
cd ..



mkdir -p pretrain && cd pretrain

wget https://yubinux.cn/tmp/pt/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth

cd ..

## verify env 
python mini_usage.py