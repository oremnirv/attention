# shellcheck disable=SC2164
cd attention
git checkout omer
sudo add-apt-repository universe
sudo apt-get update
sudo apt install python3-pip
pip3 install --upgrade pip
pip3 install tensorflow==2.1.0
pip3 install matplotlib
pip3 install sklearn
pip3 install pandas
pip3 install keras
cd
cd Downloads
#/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
#brew install wget
#wget --no-check-certificate "https://universityofcambridgecloud-my.sharepoint.com/:f:/g/personal/on234_cam_ac_uk/Esw4ksBwOo9KhWU9li0G9-QBtBikjXD7J1NskNPRsfNltQ?e=idZOEU"
#

