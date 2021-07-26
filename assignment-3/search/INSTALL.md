# Installation Instructions

```bash
# Versoes atuais do Ubuntu nao vem com python2, instalar ele:
sudo apt install -y python2

pip install virtualenv --user
mkdir ~/envs

# Criar um ambiente virtual, separado dos pacotes globais do SO:
virtualenv ~/envs/pacman --python python2
source ~/envs/pacman/bin/activate
```

## Executing
```
source ~/envs/pacman/bin/activate
```
