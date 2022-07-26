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


# Création de l'image docker:
docker build -t myhello .


# Pour transfert vers le serveur: (s'y connecter avec un tunnel sur le port 5000)
# Ajout de tag pour le transférer
docker image tag spamsa:latest 127.0.0.1:5000/spamsa:latest
# push vers le serveur
docker push 127.0.0.1:5000/spamsa:latest

# Sur le serveur au MAME, pour execution sur serveur:
# Le port 5000 du container (notre appli Flask) sera disponible sur le port 5010 de la machine (le serveur)
docker run -p 5010:5000 127.0.0.1:5000/spamsa
# Pour y accéder depuis mon PC, utiliser un SSH avec un tunnel vers port 5010 !
# Puis aller vers http://127.0.0.1:5010/static/index.html
