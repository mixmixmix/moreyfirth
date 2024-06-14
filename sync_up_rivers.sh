set -e
pipenv run python calc_sumstat.py

badger_ip=$(grep -A 1 '^Host badger$' ~/.ssh/config | grep 'Hostname' | awk '{print $2}')
ping -c 2 -W 2 $badger_ip
if [ $? -eq 0 ]; then
  echo "Hello Badgooo!!"
  rsync --append --partial -rchvP -e "ssh" out badger:/home/mix/repos/phdeep/smolts-ibm-modelling/experiments/real_rivers/
else
  echo "Unable to reach badger. Rsync aborted."
fi
