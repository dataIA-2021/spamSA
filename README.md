# spamSA

Présentation google slide https://docs.google.com/presentation/d/1RMnDL_S0ykDLHio0mdnmw1tOtoFhpCoPxjCnq6FtGQI/edit#slide=id.g1129af9647c_0_55



Trello: https://trello.com/b/nDqT2iMw/spam


Installation environnement:

conda create -n spam python=3.8
conda activate spam

pip install -r requirements.txt

python app.py

GET http://localhost:5000/detect?sms=Ceci%20est%20mon%20test...

#------------------------------------------------
Sur AWS: 
Ouvrir le port 5000...

sudo apt update
sudo apt install pip
git clone -b testAPI https://github.com/dataIA-2021/spamSA.git
cd spamSA
pip install -r requirements.txt
python3 app.py
