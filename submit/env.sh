function re() {
    # recive files from toko
    rsync -avz \
    --exclude cpu_exp \
    --exclude 'debbug_exp/data' \
    --exclude 'debbug_exp/out/*/*/checkpoints' \
    mcogo@toko.uncu.edu.ar:scratch/ .
}

function send() {
    # deletes caches and send files
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
}
alias send=send
alias re=re