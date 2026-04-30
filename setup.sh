sudo apt update
sudo apt install python3.10-venv -y
python3 -m venv venv
source venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements-arm.txt

git config --local user.name "KeisukeMiyamoto1324"
git config --local user.email "aichiboyhighschool@gmail.com"

curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
nvm install 22
nvm use 22
npm i -g @openai/codex
