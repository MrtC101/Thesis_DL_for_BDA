# deletes caches
find . -type d -name '__pycache__' -exec rm -r {} +
rsync -avz \
--exclude .vscode \
--exclude .ignore \
--exclude .git \
--exclude data \
--exclude recive.sh \
--exclude send.sh \
--exclude out \
. mcogo@toko.uncu.edu.ar:scratch/

