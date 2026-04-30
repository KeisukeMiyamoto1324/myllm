sudo apt update
sudo apt install python3.10-venv -y
python3 -m venv venv
source venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements-arm.txt
