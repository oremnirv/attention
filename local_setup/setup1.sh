# shellcheck disable=SC2164
cd attention
git checkout omer
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
brew install python@3.7
echo "export PATH=/usr/local/Cellar/python@3.7/3.7.10_2/bin:$PATH" >> ~/.bash_profile
source ~/.bash_profile
sudo add-apt-repository universe
sudo apt-get update
sudo apt install python3-pip
pip3 install --upgrade pip
pip3 install tensorflow==2.1.0
pip3 install matplotlib
pip3 install sklearn
pip3 install pandas
pip3 install keras
pip3 install tensorboard
cd
cd Downloads
mkdir attention_plots
